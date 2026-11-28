"""Model selection and averaging over saved forecasts (notebook 07).

Every function takes the data it needs as arguments; nothing reads notebook variables.

- forecasts: the long forecast table (unique_id, cutoff, ds, model, yhat, experiment, information_tier).
- actuals: unique_id, ds, y.
- evidence: development forecasts of one candidate pool joined with actuals, on the rows every candidate
  forecasts (`selection_evidence`). The candidates of a selection are the models in its evidence.
- hierarchy: one row per series, one column per selection level (`hierarchy`).

A selection decision on a given date uses only the development runs whose whole horizon was observed by
then (`completed_runs`), so it never sees the weeks it forecasts.

Cost tables and top-three rankings are cached, because the experiments ask for the same ones repeatedly (the
portfolio fallback of every policy, the same decision in several summary tables). A cache entry is keyed on
the identity of the input tables, which it holds by weak reference, and on every parameter, including the
runs. A new or reassigned table never matches an old entry; do not modify an evidence table in place.
"""

import weakref

import numpy as np
import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae

from shared.evaluation import apc_blocks, common_rows, score, to_wide
from shared.leaderboard import combine_tiers, eligible
from shared.prepare_data import DEV_CUTOFFS, completed_runs

COLUMNS = ["unique_id", "cutoff", "ds", "model", "yhat", "experiment", "information_tier"]
EXPERIMENT = "07_model_selection"
LEVELS = ["portfolio", "department", "class", "subclass", "sku"]

_cache = {}


def _cached(kind, frames, params, compute):
    """compute(), memoized on the identity of `frames` and the hashable `params`."""
    key = (kind, tuple(id(f) for f in frames), params)
    hit = _cache.get(key)
    if hit is not None and all(ref() is f for ref, f in zip(hit[0], frames)):
        return hit[1]
    value = compute()
    _cache[key] = ([weakref.ref(f) for f in frames], value)
    return value


# ------------------------------------------------------------------------------------------ candidates
def usable_models(forecasts, future_drivers):
    """Models a deployable forecast may use, sorted by name.

    future_drivers=False: models that need only sales history, calendar, item attributes and past prices and
    promotions. future_drivers=True: also models that read the price and promotion plan. Leakage diagnostics
    are never usable.
    """
    tiers = forecasts.drop_duplicates("model").set_index("model").information_tier
    scenario = "future_drivers" if future_drivers else "no_future_drivers"
    return sorted(m for m, tier in tiers.items() if eligible(tier, scenario))


def information_of(forecasts, models):
    """Information tier of a strategy built from these models: the most any of them needs."""
    return combine_tiers(forecasts.loc[forecasts.model.isin(models), "information_tier"].unique())


def hierarchy(panel):
    """Selection level of every series: portfolio, department, class, subclass and SKU. Missing values go to 'Unknown'."""
    items = panel.drop_duplicates("unique_id").set_index("unique_id")
    return pd.DataFrame({
        "portfolio": "all",
        "department": items.department_name.fillna("Unknown"),
        "class": items.class_id.astype("string").fillna("Unknown"),
        "subclass": items.subclass_id.astype("string").fillna("Unknown"),
        "sku": items.index,
    }, index=items.index)


def selection_evidence(forecasts, actuals, models):
    """Development forecasts of these models with actuals, on the rows every one of them forecasts."""
    dev_rows = forecasts[forecasts.cutoff.isin(DEV_CUTOFFS)].merge(actuals, on=["unique_id", "ds"])
    return common_rows(dev_rows[dev_rows.model.isin(models)])


# ------------------------------------------------------------------------------------------ selection
def cost_table(evidence, hierarchy, runs, level="sku", aggregate="sku", lam=0.0, mu=0.0, forecasts=None):
    """Selection cost per group (rows) and candidate (columns) over the evidence runs.

    cost = MAE + lam * |mean error| (+ mu * revision size in units of demand, which needs `forecasts`).
    aggregate="hierarchy" sums actuals and forecasts to group-week before the MAE.
    """
    models = sorted(evidence.model.unique())
    runs = tuple(runs)

    def evidence_rows():
        ev = evidence[evidence.cutoff.isin(runs)].assign(group=lambda d: d.unique_id.map(hierarchy[level]))
        if aggregate == "hierarchy":
            ev = ev.groupby(["group", "model", "cutoff", "ds"], as_index=False)[["y", "yhat"]].sum().assign(unique_id=lambda d: d.group)
        return ev

    def error_cost():
        errors = evaluate(to_wide(evidence_rows(), extra=["group"]), metrics=[mae, bias], models=models, id_col="group").set_index(["metric", "group"])
        return errors.loc["mae"] + lam * errors.loc["bias"].abs()

    cost = _cached("cost", (evidence, hierarchy), (runs, level, aggregate, float(lam)), error_cost).copy()
    if mu:   # the revision penalty also depends on the forecasts; computed fresh every time
        ev = evidence_rows()
        fcst = forecasts[forecasts.model.isin(models) & forecasts.cutoff.isin(runs)]
        blocks = apc_blocks(fcst, list(runs)[1:])
        blocks["group"] = blocks.unique_id.map(hierarchy[level])
        revision = blocks.groupby(["group", "model"])[["change", "base"]].sum()
        revision = (revision.change / revision.base.where(revision.base > 0)).unstack("model")
        mean_y = ev[ev.model == models[0]].groupby("group").y.mean()
        cost += mu * revision.reindex_like(cost).fillna(0).mul(mean_y, axis=0)
    return cost


def choose(evidence, hierarchy, runs, level="sku", aggregate="sku", lam=0.0, mu=0.0, margin=0.0, default=None, forecasts=None):
    """The chosen model for every series, and the fallback model.

    Each group takes its lowest-cost candidate. The fallback is the portfolio winner unless `default` is given;
    a group keeps the fallback unless its winner lowers the cost by more than `margin`. Exact ties go to the
    first candidate alphabetically.
    """
    cost = cost_table(evidence, hierarchy, runs, level, aggregate, lam, mu, forecasts)
    default = default or cost_table(evidence, hierarchy, runs, "portfolio", aggregate, lam, mu, forecasts).iloc[0].idxmin()
    pick = cost.idxmin(axis=1)
    gain = 1 - cost.min(axis=1) / cost[default]
    pick = pick.where(gain > margin, default)
    return hierarchy[level].map(pick), default


def strategy(name, forecasts, evidence, hierarchy, decisions, picks=None, **policy):
    """Selected forecasts at each decision cutoff, chosen with only the runs completed by then.

    `policy` holds the options of `choose` (level, aggregate, lam, mu, margin, default). A series with no
    evidence, or whose chosen model has no forecast at that cutoff, gets the fallback model. If `picks` is a
    dict, the choice per series is stored in it under (name, cutoff).
    """
    models = sorted(evidence.model.unique())
    out = []
    for cutoff in decisions:
        choice, default = choose(evidence, hierarchy, completed_runs(cutoff), forecasts=forecasts, **policy)
        if picks is not None:
            picks[(name, cutoff)] = choice
        rows = forecasts[forecasts.model.isin(models) & (forecasts.cutoff == cutoff)]
        picked = rows[rows.model == rows.unique_id.map(choice)]
        fallback = rows[(rows.model == default) & ~rows.unique_id.isin(picked.unique_id)]
        out.append(pd.concat([picked, fallback]).assign(model=name))
    tier = information_of(forecasts, models)
    return pd.concat(out).assign(experiment=EXPERIMENT, information_tier=tier)[COLUMNS]


# ------------------------------------------------------------------------------------------ averages and references
def average(name, forecasts, models, decisions):
    """Equal-weight average of these models' forecasts at each decision cutoff."""
    rows = forecasts[forecasts.model.isin(models) & forecasts.cutoff.isin(decisions)]
    out = rows.groupby(["unique_id", "cutoff", "ds"], as_index=False).yhat.mean()
    return out.assign(model=name, experiment=EXPERIMENT, information_tier=information_of(forecasts, models))


def top3(forecasts, actuals, models, runs):
    """The three models with the lowest WAPE over these runs."""
    def ranking():
        rows = forecasts[forecasts.model.isin(models) & forecasts.cutoff.isin(runs)]
        return score(rows, actuals, rmsse=False).sort_values("wape").model.head(3).tolist()

    return list(_cached("top3", (forecasts, actuals), (tuple(sorted(models)), tuple(runs)), ranking))


def best_individual(name, forecasts, actuals, models, decision):
    """Forecasts at `decision` of the model with the lowest WAPE on the runs completed by then, and that model's name."""
    winner = top3(forecasts, actuals, models, completed_runs(decision))[0]
    rows = forecasts[(forecasts.model == winner) & (forecasts.cutoff == decision)]
    return rows.assign(model=name, experiment=EXPERIMENT)[COLUMNS], winner


def wape_by_cutoff(forecasts, actuals, cutoffs):
    """WAPE of every model (rows) at each cutoff (columns), each cutoff scored on its own common rows."""
    by_cutoff = pd.concat([score(forecasts[forecasts.cutoff == c], actuals, rmsse=False).assign(cutoff=c) for c in cutoffs])
    return by_cutoff.pivot(index="model", columns="cutoff", values="wape")


# ------------------------------------------------------------------------------------------ ensembles
def ensemble_evidence(name, forecasts, actuals, models, decision):
    """Evidence rows for a top-three ensemble of `models` at a decision date.

    Members for each completed run are the top three on the other completed runs (leave one out), so the
    ensemble's evidence is not scored on the runs that chose it. Needs at least two completed runs.
    """
    runs = completed_runs(decision)
    parts = []
    for run in runs:
        members = top3(forecasts, actuals, models, [r for r in runs if r != run])
        rows = forecasts[forecasts.model.isin(members) & (forecasts.cutoff == run)]
        parts.append(rows.groupby(["unique_id", "cutoff", "ds"], as_index=False).yhat.mean().assign(model=name))
    return pd.concat(parts).merge(actuals, on=["unique_id", "ds"])


def residuals(forecasts, actuals, models, runs):
    """Forecast errors (forecast - actual), one column per model, on the rows every model forecasts, and the actuals."""
    rows = common_rows(forecasts[forecasts.model.isin(models) & forecasts.cutoff.isin(runs)]).merge(actuals, on=["unique_id", "ds"])
    errors = rows.assign(error=rows.yhat - rows.y).pivot(index=["unique_id", "cutoff", "ds"], columns="model", values="error")
    y = rows.drop_duplicates(["unique_id", "cutoff", "ds"]).set_index(["unique_id", "cutoff", "ds"]).y.reindex(errors.index)
    return errors, y


def ensemble_rules(forecasts, actuals, family, runs):
    """Members of ensembles C to F, chosen on these runs. `family` maps each pool model to its family.

    C: the best model and the best statistical model.
    D: the best model and the best model from another machine-learning family.
    E: the best model, then twice the competitive model (WAPE within 3 points of the best) least correlated
       with the members so far.
    F: the best model and the best model of each of the two next-best families.
    """
    errors, y = residuals(forecasts, actuals, list(family), runs)
    single = errors.abs().sum() / y.sum()
    best = single.idxmin()
    best_of = lambda fams: single[[m for m in single.index if family[m] in fams]].idxmin()
    correlation = errors.corr()

    competitive = single[single <= single.min() + 0.03].index
    diverse = [best]
    while len(diverse) < 3:
        diverse.append(correlation.loc[competitive.difference(diverse), diverse].mean(axis=1).idxmin())

    other_families = single.groupby(family).min().drop(family[best]).nsmallest(2).index
    return {
        "Ensemble_C_best_plus_statistical": [best, best_of(["statistical"])],
        "Ensemble_D_best_plus_other_ml": [best, best_of({"lgbm_l1", "lgbm_other", "catboost"} - {family[best]})],
        "Ensemble_E_diverse": diverse,
        "Ensemble_F_family": [best] + [best_of([f]) for f in other_families],
    }


def mean_correlation(errors, members):
    """Mean pairwise error correlation between ensemble members (NaN for a single model)."""
    corr = errors[members].corr()
    return corr.where(~np.eye(len(members), dtype=bool)).stack().mean()
