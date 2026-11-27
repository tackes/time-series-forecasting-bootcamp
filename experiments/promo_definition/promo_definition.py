"""Exploratory experiment: does a business rule on the promotion flag help the forecast?

An observation counts as a promotional event only when it is flagged as promotional AND its
planned discount reaches a threshold:

    discount_pct   = 1 - sale_price / regular_price      (undefined when regular_price <= 0)
    promo_adjusted = is_promotional & (discount_pct >= threshold)

Nothing in day1_ml/, shared/ or results/day1/ is modified. Everything is written to
experiments/promo_definition/results/.

Stages (run from the repository root):

    python experiments/promo_definition/promo_definition.py definitions
    python experiments/promo_definition/promo_definition.py models A B10 C10      # any config names
    python experiments/promo_definition/promo_definition.py evaluate
    python experiments/promo_definition/promo_definition.py plots

Config names: A (control), B<t> (adjusted flag only), C<t> (original and adjusted flags), t in 5, 10, 15, 20.
Configuration D (discount-based flag regardless of the original flag) is characterised in `definitions`
but not modelled: see the report.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from shared.prepare_data import load_panel, H, FREQ, ALL_CUTOFFS, DEV_CUTOFFS, TEST_CUTOFF
from shared.features import (LAGS, LAG_TRANSFORMS, DATE_FEATURES, LGB_PARAMS, CATEGORICAL, PROMO_CALENDAR,
                             add_drivers, item_attributes, future_drivers, promo_features)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
EPS = 1e-9                                   # 9.00 / 10.00 must count as a 10% discount
PRICE_COLS = ["discount", "price_index", "regular_price", "planned_price"]
CONTROL_COLS = PROMO_CALENDAR + PRICE_COLS   # lgbm_l1_future_price_levels, notebook 06
CONTROL_MODEL = "lgbm_l1_future_price_levels"


# ----------------------------------------------------------------------------- definitions
def discount_pct(frame):
    """1 - sale_price / regular_price; NaN where the regular price is missing or not positive."""
    regular = frame.regular_price.where(frame.regular_price > 0)
    return 1 - frame.sale_price / regular


def adjusted_flag(frame, threshold):
    """Flagged as promotional and discounted by at least `threshold`. An undefined discount fails the rule."""
    return ((frame.is_promotional == 1) & (discount_pct(frame) >= threshold - EPS)).astype(int)


def discount_flag(frame, threshold):
    """Configuration D: discounted by at least `threshold`, whatever the promotion flag says (realized price)."""
    return (discount_pct(frame) >= threshold - EPS).astype(int)


def profile(frame, flag, original):
    """Descriptive statistics of one promotion definition."""
    d = discount_pct(frame)
    by_product = flag.groupby(frame.unique_id).agg(["min", "max"])
    return {
        "promo_share_pct": 100 * flag.mean(),
        "original_promo_reclassified_pct": 100 * ((original == 1) & (flag == 0)).sum() / (original == 1).sum(),
        "non_promo_now_flagged_pct": 100 * ((original == 0) & (flag == 1)).sum() / (original == 0).sum(),
        "always_promo_products": int((by_product["min"] == 1).sum()),
        "never_promo_products": int((by_product["max"] == 0).sum()),
        "sometimes_promo_products": int(((by_product["min"] == 0) & (by_product["max"] == 1)).sum()),
        "volume_weighted_promo_share_pct": 100 * (frame.y * flag).sum() / frame.y.sum(),
    }


def definitions():
    rows = []
    # Weeks up to the final cutoff: the period behind the figures quoted in the request.
    for label, sample in [("teaching sample (2,000 series)", True), ("full dataset", False)]:
        frame = load_panel(sample=sample)
        frame = frame[frame.ds <= TEST_CUTOFF].reset_index(drop=True)
        original = frame.is_promotional.astype(int)
        d = discount_pct(frame)
        invalid = int(d.isna().sum())
        base = {"data": label, "definition": "original flag", "threshold": np.nan, **profile(frame, original, original),
                "non_promo_discount_over_threshold_pct": np.nan, "invalid_regular_price_rows": invalid}
        rows.append(base)
        for t in THRESHOLDS:
            for name, flag in [("B: flag and discount", adjusted_flag(frame, t)), ("D: discount only (realized)", discount_flag(frame, t))]:
                stats = profile(frame, flag, original)
                stats["non_promo_discount_over_threshold_pct"] = 100 * (d[flag == 0] >= t - EPS).mean()
                rows.append({"data": label, "definition": name, "threshold": t, **stats, "invalid_regular_price_rows": invalid})
        # Discount distribution as described in the request, and the leakage check behind configuration D.
        promo = original == 1
        print(f"{label}: discount > 1% in {100 * (d[promo] > 0.01).mean():.1f}% of promotional and "
              f"{100 * (d[~promo] > 0.01).mean():.1f}% of non-promotional observations; {invalid} rows with invalid regular price")
        if sample:
            leak = pd.DataFrame({
                "promotional": promo.map({True: "flagged promotional", False: "not flagged"}),
                "sales": np.where(frame.y > 0, "sales > 0", "zero sales"),
                "discount_ge_10pct": d >= 0.10 - EPS,
            }).groupby(["promotional", "sales"]).discount_ge_10pct.agg(share_pct=lambda s: 100 * s.mean(), rows="size")
            leak.round(3).to_csv(OUT / "realized_discount_vs_zero_sales.csv")
            print(leak.round(3))
    table = pd.DataFrame(rows)
    table.round(3).to_csv(OUT / "definitions.csv", index=False)
    print(table.round(2).to_string())


# ----------------------------------------------------------------------------- models
def prepared():
    panel = add_drivers(load_panel())
    for t in THRESHOLDS:
        flag = adjusted_flag(panel, t)
        # The same rule from the planned-price proxy used by the control (discount is the planned discount).
        assert (flag == ((panel.promo == 1) & (panel.discount >= t - EPS)).astype(int)).all()
        panel[f"promo_adj{round(t * 100)}"] = flag
    # Placebo: drop the same share of promotion weeks as the 10% rule, chosen at random (fixed seed).
    rng = np.random.default_rng(0)
    share = ((panel.promo == 1) & (panel.promo_adj10 == 0)).sum() / (panel.promo == 1).sum()
    panel["promo_placebo10"] = ((panel.promo == 1) & (rng.random(len(panel)) >= share)).astype(int)
    return panel


def feature_frame(panel, config):
    """unique_id, ds and the plan features of one configuration, for every week (plan known for the horizon)."""
    control = future_drivers(panel)[["unique_id", "ds", *CONTROL_COLS]]
    config, _ = split_seed(config)
    if config == "A":                          # the control
        return control, CONTROL_COLS
    flag = "promo_placebo10" if config.endswith("placebo") else f"promo_adj{int(config[1:])}"
    t = "placebo" if config.endswith("placebo") else int(config[1:])
    calendar = promo_features(panel[["unique_id", "ds", flag]].rename(columns={flag: "promo"}))
    if config[0] == "B":                       # adjusted calendar replaces the original one
        frame = calendar.merge(panel[["unique_id", "ds", *PRICE_COLS]], on=["unique_id", "ds"])
        return frame, CONTROL_COLS
    if config[0] == "C":                       # adjusted calendar added next to the original one
        adj_cols = [f"{c}_adj{t}" for c in PROMO_CALENDAR]
        calendar = calendar.rename(columns=dict(zip(PROMO_CALENDAR, adj_cols)))
        return control.merge(calendar, on=["unique_id", "ds"]), CONTROL_COLS + adj_cols
    raise ValueError(config)


def split_seed(config):
    """"B10_seed2" -> ("B10", 2). A config without a seed suffix uses notebook 06's random_state."""
    base, _, seed = config.partition("_seed")
    return base, int(seed) if seed else None


def model_name(config):
    base, seed = split_seed(config)
    if base == "A":
        name = "A_control" if seed is None else "ref_control"
    elif base.endswith("placebo"):
        name = f"ref_{base[0]}_placebo"
    else:
        name = f"{base[0]}_adj{base[1:]}"
    return name if seed is None else f"{name}_seed{seed}"


def lgb_params(config):
    """Notebook 06 parameters; the seed references change random_state only."""
    _, seed = split_seed(config)
    return LGB_PARAMS if seed is None else {**LGB_PARAMS, "random_state": seed}


def run_models(configs):
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.target_transforms import LocalStandardScaler
    from shared.leaderboard import to_long

    panel = prepared()
    actuals = panel[["unique_id", "ds", "y"]]
    attributes = item_attributes(panel)
    as_category = {col: "category" for col in CATEGORICAL}
    for config in configs:
        frame, cols = feature_frame(panel, config)
        name = model_name(config)
        out, start = [], time.perf_counter()
        for cutoff in ALL_CUTOFFS:            # same cutoffs, sample, lags, scaling and parameters as notebook 06
            history = actuals[actuals.ds <= cutoff]
            end = cutoff + pd.Timedelta(weeks=H)
            train = (history.merge(frame[["unique_id", "ds", *cols]], on=["unique_id", "ds"])
                     .merge(attributes, on="unique_id").astype(as_category))
            X_df = frame.loc[(frame.ds > cutoff) & (frame.ds <= end), ["unique_id", "ds", *cols]]
            mlf = MLForecast(
                models={name: lgb.LGBMRegressor(**lgb_params(config), objective="l1")},
                freq=FREQ, lags=LAGS, lag_transforms=LAG_TRANSFORMS, date_features=DATE_FEATURES,
                target_transforms=[LocalStandardScaler()], num_threads=1,
            )
            mlf.fit(train, static_features=CATEGORICAL)
            out.append(to_long(mlf.predict(H, X_df=X_df), cutoff, "promo_definition", "future_drivers"))
        out = pd.concat(out)
        out.astype({"yhat": "float32"}).to_parquet(OUT / f"forecasts_{name}.parquet", index=False)
        print(f"{name}: {len(cols)} plan features, {time.perf_counter() - start:.0f}s", flush=True)
        if config == "A":
            check_control(out)


def check_control(control):
    saved = pd.read_parquet(ROOT / "results" / "day1" / "forecasts" / "06_l1_features.parquet").query("model == @CONTROL_MODEL")
    key = ["unique_id", "cutoff", "ds"]
    both = control.merge(saved, on=key, suffixes=("", "_saved"))
    assert len(both) == len(saved) == len(control)
    diff = (both.yhat.astype("float32") - both.yhat_saved).abs().max()
    print(f"control vs saved {CONTROL_MODEL}: {len(both):,} rows, max |difference| = {diff}")
    pd.Series({"rows": len(both), "max_abs_difference": float(diff)}).to_csv(OUT / "control_reproduction.csv")


# ----------------------------------------------------------------------------- evaluation
def load_all():
    files = sorted(OUT.glob("forecasts_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def week_segments(panel):
    """Per week: product type (original flag, weeks up to the final cutoff) and week type."""
    history = panel[panel.ds <= TEST_CUTOFF].groupby("unique_id").promo.agg(["min", "max"])
    product = np.select([history["min"] == 1, history["max"] == 0], ["always promotional", "never promotional"], "sometimes promotional")
    product = pd.Series(product, index=history.index, name="product_type")
    d = discount_pct(panel)
    weeks = panel[["unique_id", "ds"]].copy()
    weeks["product_type"] = weeks.unique_id.map(product)
    weeks["week_type"] = np.select(
        [(panel.promo == 1) & (panel.promo_adj10 == 1), (panel.promo == 1) & (panel.promo_adj10 == 0)],
        ["promotional, kept at 10%", "promotional, reclassified at 10%"], "not promotional")
    weeks["realized_discount"] = np.where(d >= 0.10 - EPS, "realized discount >= 10%", "realized discount < 10%")
    return weeks


def base_and_seed(model):
    """("B_adj10", 0) for B_adj10, ("B_adj10", 2) for B_adj10_seed2; the control family is "A_control"."""
    base, _, seed = model.partition("_seed")
    base = "A_control" if base == "ref_control" else base
    return base, int(seed) if seed else 0


def evaluate():
    from shared.evaluation import score, common_rows
    from shared.leaderboard import leaderboard

    panel = prepared()
    actuals = panel[["unique_id", "ds", "y"]]
    fcst = common_rows(load_all())
    families = pd.DataFrame([base_and_seed(m) for m in fcst.model.unique()], index=fcst.model.unique(), columns=["config", "seed"])
    order = lambda m: (not m.startswith("A_"), m.startswith("ref_"), m)
    seed0 = sorted(families[families.seed == 0].index, key=order)

    dev = leaderboard(fcst, actuals, scope="dev").set_index("model")
    final = leaderboard(fcst, actuals, scope="test").set_index("model")
    by_origin = pd.concat([score(fcst[fcst.cutoff == c], actuals, rmsse=False).assign(cutoff=c) for c in DEV_CUTOFFS]
                          ).pivot(index="model", columns="cutoff", values="wape")
    control_of = {m: ("A_control" if s == 0 else f"ref_control_seed{s}") for m, s in families.seed.items()}
    control_origin = by_origin.loc[[control_of[m] for m in by_origin.index]].set_axis(by_origin.index)
    table = pd.DataFrame({
        "dev_wape": dev.wape, "dev_bias": dev.bias, "dev_mean_rmsse": dev.mean_rmsse, "dev_median_rmsse": dev.median_rmsse,
        "dev_apc": dev.apc,
        "dev_wape_change_pts": 100 * (dev.wape - dev.wape.loc[[control_of[m] for m in dev.index]].to_numpy()),
        "origins_better_than_control": by_origin.lt(control_origin).sum(axis=1),
        "final_wape": final.wape, "final_bias": final.bias,
    }).join(families)
    table.loc[table.config == "A_control", "origins_better_than_control"] = np.nan

    # 1) The comparison as specified: notebook 06's random_state, one fit per configuration.
    summary = table.loc[seed0].drop(columns=["config", "seed"])
    summary.round(4).to_csv(OUT / "summary.csv")
    print("Seed 0 (notebook 06 random_state):")
    print(summary.round(4).to_string())
    (100 * (by_origin.loc[seed0] - by_origin.loc["A_control"])).round(2).to_csv(OUT / "dev_wape_change_by_origin.csv")
    print((100 * (by_origin.loc[seed0] - by_origin.loc["A_control"])).round(2).to_string())

    # 2) Over random seeds: each configuration against the control fitted with the same seed.
    seeds = table.groupby("config").agg(
        seeds=("seed", "nunique"), dev_wape_mean=("dev_wape", "mean"), dev_wape_min=("dev_wape", "min"), dev_wape_max=("dev_wape", "max"),
        change_pts_mean=("dev_wape_change_pts", "mean"), change_pts_min=("dev_wape_change_pts", "min"),
        change_pts_max=("dev_wape_change_pts", "max"), seeds_better=("dev_wape_change_pts", lambda c: int((c < 0).sum())),
        origins_better_mean=("origins_better_than_control", "mean"), dev_bias_mean=("dev_bias", "mean"),
        dev_apc_mean=("dev_apc", "mean"), final_wape_mean=("final_wape", "mean"))
    seeds = seeds.loc[sorted(seeds.index, key=order)]
    seeds.round(4).to_csv(OUT / "summary_over_seeds.csv")
    print()
    print("Over seeds (paired with the control of the same seed):")
    print(seeds.round(4).to_string())

    # 3) Segments on the development cutoffs, WAPE and bias averaged over seeds.
    rows = fcst[fcst.cutoff.isin(DEV_CUTOFFS)].merge(week_segments(panel), on=["unique_id", "ds"])
    parts = [score(rows, actuals, by=[by], rmsse=False).rename(columns={by: "segment"}).assign(split=by)
             for by in ["product_type", "week_type", "realized_discount"]]
    seg = pd.concat(parts).merge(families, left_on="model", right_index=True)
    for metric in ["wape", "bias"]:
        wide = seg.pivot_table(index=["split", "segment"], columns="config", values=metric, aggfunc="mean")
        wide = wide[sorted(wide.columns, key=order)]
        wide.insert(0, "rows", seg.drop_duplicates(["split", "segment"]).set_index(["split", "segment"]).rows)
        if metric == "wape":
            change = 100 * wide.drop(columns="rows").sub(wide["A_control"], axis=0)
            change.round(2).to_csv(OUT / "dev_segments_wape_change_pts.csv")
            print()
            print("Segment WAPE change vs control, points (mean over seeds):")
            print(change.round(2).to_string())
        wide.round(4).to_csv(OUT / f"dev_segments_{metric}.csv")
        print()
        print(f"Segment {metric} (mean over seeds):")
        print(wide.round(4).to_string())


# ----------------------------------------------------------------------------- plots
def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shared.evaluation import score

    panel = prepared()
    actuals = panel[["unique_id", "ds", "y"]]
    weeks = week_segments(panel)
    fcst = load_all().query("model in ['A_control', 'B_adj10', 'C_adj10']")

    # Products chosen on the definition only: most reclassified promotion weeks inside development horizons,
    # one per volume tercile among sometimes-promotional products, plus one always-promotional product.
    in_dev = np.zeros(len(panel), bool)
    for c in DEV_CUTOFFS:
        in_dev |= (panel.ds > c).to_numpy() & (panel.ds <= c + pd.Timedelta(weeks=H)).to_numpy()
    panel["reclassified"] = (panel.promo == 1) & (panel.promo_adj10 == 0)
    stats = panel[in_dev].groupby("unique_id").agg(reclassified=("reclassified", "sum"))
    stats["volume"] = panel[panel.ds <= TEST_CUTOFF].groupby("unique_id").y.mean()
    stats["tercile"] = pd.qcut(stats.volume.rank(method="first"), 3, labels=["low volume", "mid volume", "high volume"])
    stats["product_type"] = weeks.drop_duplicates("unique_id").set_index("unique_id").product_type
    picks = {}
    sometimes = stats[stats.product_type == "sometimes promotional"]
    for tercile, group in sometimes.groupby("tercile", observed=True):
        picks[group.reclassified.idxmax()] = f"sometimes promotional, {tercile}"
    always = stats[stats.product_type == "always promotional"]
    picks[always.reclassified.idxmax()] = "always promotional"

    report = []
    for uid, why in picks.items():
        p = panel[panel.unique_id == uid]
        # the development cutoff whose horizon holds most of this product's reclassified weeks
        counts = {c: p[(p.ds > c) & (p.ds <= c + pd.Timedelta(weeks=H))].reclassified.sum() for c in DEV_CUTOFFS}
        cutoff = max(counts, key=counts.get)
        f = fcst[(fcst.unique_id == uid) & (fcst.cutoff == cutoff)]
        err = score(fcst[(fcst.unique_id == uid) & fcst.cutoff.isin(DEV_CUTOFFS)], actuals, rmsse=False).set_index("model")
        report.append({"unique_id": uid, "why": why, "plot_cutoff": cutoff.date(),
                       "promo_weeks": int(p.promo.sum()), "reclassified_weeks": int(p.reclassified.sum()),
                       **{f"dev_mae_{m}": err.mae[m] for m in err.index}})

        fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.2]})
        axes[0].plot(p.ds, p.y, color="black", lw=1, label="sales")
        for model, style in [("A_control", dict(color="tab:blue")), ("B_adj10", dict(color="tab:orange")),
                             ("C_adj10", dict(color="tab:green", ls="--"))]:
            g = f[f.model == model]
            axes[0].plot(g.ds, g.yhat, lw=1.6, label=f"{model} forecast", **style)
        axes[0].axvline(cutoff, color="grey", ls=":", lw=1)
        axes[0].set_title(f"{uid} ({why}); forecasts from {cutoff.date()}")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[1].plot(p.ds, p.regular_price, color="tab:gray", label="regular price")
        axes[1].plot(p.ds, p.sale_price, color="tab:red", lw=1, label="selling price")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[2].fill_between(p.ds, 0, p.promo, step="mid", color="tab:blue", alpha=0.25, label="original flag")
        axes[2].step(p.ds, p.promo_adj10 * 0.8, where="mid", color="tab:orange", label="adjusted flag (10%), drawn at 0.8")
        axes[2].set_yticks([0, 1])
        axes[2].legend(loc="upper left", fontsize=8)
        for ax in axes:
            ax.axvspan(cutoff, cutoff + pd.Timedelta(weeks=H), color="grey", alpha=0.08)
        fig.tight_layout()
        fig.savefig(OUT / f"product_{uid}.png", dpi=110)
        plt.close(fig)
    report = pd.DataFrame(report)
    report.round(1).to_csv(OUT / "product_examples.csv", index=False)
    print(report.round(1).to_string())


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "definitions":
        definitions()
    elif stage == "models":
        run_models(sys.argv[2:])
    elif stage == "evaluate":
        evaluate()
    elif stage == "plots":
        plots()
    else:
        raise SystemExit(__doc__)
