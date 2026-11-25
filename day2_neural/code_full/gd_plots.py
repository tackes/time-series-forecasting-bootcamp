"""Figures for the full-data Day 2 notebooks: one function per chart, the same look everywhere.

Every function draws and **shows** its figure (``plt.show()``), so the notebooks display
it rather than just saving it. The catalogue, and where each figure appears:

====  ==========================  ==========================================================
F1    ``plot_series_price``       units with price on a second axis, promo weeks, forecasts
F1b   ``plot_series_panels``      the same as two stacked panels (the twin-axis alternative)
F2    ``plot_error_dist``         per-series error: histogram, ECDF, box plot by class
F3    ``plot_class_heatmap``      median error, sales class × model (``plot_heatmap`` draws it)
F4    ``plot_wins``               marginal wins against a baseline
F5    ``plot_elasticity``         elasticity histogram, by class, and against shift size
F6    ``plot_elasticity_strength``  % of series per elasticity category, per arm
F7    ``plot_elasticity_by_class``  median elasticity by class, per arm, against empirical
F8    ``plot_training``           loss curves with warm-up end and best-weights step
F9    ``plot_arm_comparison``     seeds and arms: curves, oscillation, accuracy, elasticity
F10   ``plot_pareto``             accuracy against elasticity, the front and the knees
F11   ``plot_probe``              lift by shock size, sustained vs one-week pulse
F13   ``plot_lift_by_depth``      lift by cut depth (random calendars), against references
====  ==========================  ==========================================================

F12 (forecasts at -20% / base / +20% price) is ``plot_series_price`` with scenario columns.
"""

from __future__ import annotations

import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import gd_helpers as gd
import gd_price as gp

bc = gd.bc

# Okabe-Ito, colour-blind safe. Fixed roles so a colour means the same thing everywhere.
COLOURS = {"actual": "#000000", "price": "#CC79A7", "regular": "#999999", "promo": "#56B4E9",
           "test": "#F2F2F2", "good": "#009E73", "bad": "#D55E00", "band": "#009E73"}
MODEL_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
CATEGORY_COLOURS = {"Highly elastic": "#08306B", "Elastic": "#2171B5", "Inelastic": "#9ECAE1",
                    "Flat": "#F0E442", "Perverse": "#D55E00"}
TIERS = {"high": ["A", "B"], "medium": ["D", "E", "F"], "low": ["I", "J"]}
CLASS_ORDER = list("ABCDEFGHIJ")


def use_style() -> None:
    """One look for every figure: inline PNG at a modest dpi keeps notebooks small."""
    plt.rcParams.update({
        "figure.figsize": (12, 4.5), "figure.dpi": 90, "savefig.dpi": 90,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
        "axes.prop_cycle": plt.cycler(color=MODEL_COLOURS),
    })
    try:
        from IPython import get_ipython

        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'png'")
    except Exception:  # not in a notebook
        pass


def _colour(i: int) -> str:
    """The ``i``-th model colour, cycling through ``MODEL_COLOURS``."""
    return MODEL_COLOURS[i % len(MODEL_COLOURS)]


def example_ids(examples: dict) -> list[str]:
    """The example series from ``00_baselines`` (tier -> ids) as one flat list, in tier order."""
    ids = []
    for tier_ids in examples.values():
        ids += tier_ids
    return ids


def series_table(panel: pl.DataFrame) -> pd.DataFrame:
    """Per-series attributes for slicing charts: class, demand type, mean weekly units."""
    return bc.series_info(panel).to_pandas().set_index("unique_id")


# ============================================================================
# EXAMPLE SERIES (chosen once in 00, reused everywhere)
# ============================================================================

def pick_examples(panel: pl.DataFrame, n_per_tier: int = 2, seed: int = bc.RANDOM_SEED) -> dict:
    """Two series per volume tier (high A-B, medium D-F, low I-J), saved for every notebook.

    Picks sit near the tier's median volume rather than at the top, are still selling in the
    last 14 training weeks, and the first pick of each tier has a price cut in the test window,
    so price response is visible.
    """
    if gd.EXAMPLES_PATH.exists():
        return json.loads(gd.EXAMPLES_PATH.read_text())
    ids = set(gd.eval_ids())
    stats = (panel.filter(pl.col("unique_id").is_in(list(ids)))
             .group_by("unique_id")
             .agg(pl.col("sale_class").first(),
                  pl.col("y").filter(pl.col("ds") < bc.TEST_START).mean().alias("mean_y"),
                  pl.col("y").filter((pl.col("ds") < bc.TEST_START)
                                     & (pl.col("ds") >= bc.VALID_START)).mean().alias("recent_y"),
                  (pl.col("discount_depth").filter(pl.col("ds") >= bc.TEST_START) >= 0.10)
                  .any().alias("cut_in_test"))
             .to_pandas())
    stats = stats[stats["recent_y"] >= 0.5 * stats["mean_y"]]     # still selling recently
    rng = np.random.default_rng(seed)
    picks = {}
    for tier, classes in TIERS.items():
        g = stats[stats["sale_class"].isin(classes)].copy()
        g["dist"] = (np.log1p(g["mean_y"]) - np.log1p(g["mean_y"].median())).abs()
        g = g.sort_values("dist").head(max(10, 3 * n_per_tier))   # near the middle
        if g.empty:
            continue
        with_cut = g[g["cut_in_test"]]
        first = with_cut.sample(1, random_state=rng.integers(1e9)) if len(with_cut) else g.head(1)
        others = g.drop(first.index)
        rest = others.sample(min(n_per_tier - 1, len(others)), random_state=rng.integers(1e9))
        picks[tier] = first["unique_id"].tolist() + rest["unique_id"].tolist()
    gd.EXAMPLES_PATH.write_text(json.dumps(picks, indent=2))
    return picks


# ============================================================================
# F1 / F1b / F12: series with price
# ============================================================================

def _promo_spans(ds: pd.Series, flag: pd.Series) -> list[tuple]:
    """Runs of consecutive promo weeks as (start, end) pairs, for shading."""
    spans, start = [], None
    for d, on in zip(ds, flag):
        if on and start is None:
            start = d
        if not on and start is not None:
            spans.append((start, d))
            start = None
    if start is not None:
        spans.append((start, ds.iloc[-1] + pd.Timedelta(weeks=1)))
    return spans


def _one_series(ax, s: pd.DataFrame, f: pd.DataFrame | None, models: list[str],
                cutoff, bands: dict | None, events: bool):
    """Draw one series: units on ``ax``, price on a twin axis. Returns (bias text, twin)."""
    ax.axvspan(pd.Timestamp(cutoff), s["ds"].max() + pd.Timedelta(weeks=1),
               color=COLOURS["test"], zorder=0, lw=0)
    for a, b in _promo_spans(s["ds"], s["promo"] > 0):
        ax.axvspan(a, b, color=COLOURS["promo"], alpha=0.18, lw=0, zorder=0)
    ax.plot(s["ds"], s["y"], color=COLOURS["actual"], lw=1.4, marker="o", ms=2, label="actual")
    promo = s[s["promo"] > 0]
    ax.scatter(promo["ds"], promo["y"], c=promo["discount_depth"], cmap="viridis", vmin=0,
               vmax=0.5, s=14, zorder=3, label="promo week (colour = discount)")
    ymax = s["y"].max()
    bias = []
    for i, m in enumerate(models):
        ax.plot(f["ds"], f[m], color=_colour(i), lw=1.5, ls="--", label=m)
        if bands and m in bands:
            lo, hi = bands[m]
            ax.fill_between(f["ds"], f[lo], f[hi], color=_colour(i), alpha=0.15, lw=0)
        ymax = max(ymax, f[m].max())
        actual = s[s["ds"] >= pd.Timestamp(cutoff)]["y"].sum()
        if actual > 0:
            bias.append(f"{m} {f[m].clip(lower=0).sum() / actual - 1:+.0%}")
    if events:
        for c in gp.EVENT_COLS:
            for d in s.loc[s.get(c, pd.Series(0, index=s.index)) > 0, "ds"]:
                ax.axvline(d, color="0.4", ls=":", lw=0.8)
                ax.text(d, ymax * 2.1, c.replace("ev_", ""), rotation=90, fontsize=7,
                        va="top", ha="right", color="0.3")
    ax.set_ylim(0, max(ymax, 1) * 2.2)          # units in the lower half
    ax.set_ylabel("units / week")

    ax2 = ax.twinx()
    ax2.step(s["ds"], s["sale_price"], where="mid", color=COLOURS["price"], lw=1.2,
             label="sale price")
    if "regular_price" in s:
        ax2.step(s["ds"], s["regular_price"], where="mid", color=COLOURS["regular"], lw=1.0,
                 ls="--", label="regular price")
    pmax = np.nanmax(s[["sale_price"] + (["regular_price"] if "regular_price" in s else [])])
    ax2.set_ylim(0, pmax * 1.15)               # price near the top
    ax2.set_ylabel("price", color=COLOURS["price"])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return " | ".join(bias), ax2


def plot_series_price(panel: pl.DataFrame, ids, fcst: pd.DataFrame | None = None,
                      models: list[str] | None = None, cutoff=bc.TEST_START,
                      history_weeks: int = 52, bands: dict | None = None,
                      events: bool = False, title: str = "", ncols: int = 2) -> None:
    """F1: units (left) and sale/regular price (right, step lines), with promo weeks shaded.

    ``ids`` is a list or the ``pick_examples`` dict (one row per volume tier). Forecast columns
    in ``models`` are drawn dashed over the test window, with bias % in each title. ``bands``
    maps a model to its (lo, hi) interval columns. ``events`` marks event weeks.
    """
    if isinstance(ids, dict):
        labels = {u: tier for tier, us in ids.items() for u in us}
        ids = [u for us in ids.values() for u in us]
    else:
        labels = {}
    models = models or ([] if fcst is None else bc.model_columns(fcst))
    start = pd.Timestamp(cutoff) - pd.Timedelta(weeks=history_weeks)
    data = gd.only(panel, ids).to_pandas()
    data["ds"] = pd.to_datetime(data["ds"])
    if fcst is not None:
        fcst = fcst.reset_index() if "unique_id" not in fcst.columns else fcst
        fcst = fcst.assign(ds=pd.to_datetime(fcst["ds"]))

    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.1 * nrows), squeeze=False)
    for ax, uid in zip(axes.flat, ids):
        s = data[(data["unique_id"] == uid) & (data["ds"] >= start)].sort_values("ds")
        f = fcst[fcst["unique_id"] == uid].sort_values("ds") if fcst is not None else None
        bias, ax2 = _one_series(ax, s, f, models if f is not None else [], cutoff, bands, events)
        cls = s["sale_class"].iloc[0]
        head = f"{labels.get(uid, '')} volume · class {cls} · {uid}".strip(" ·")
        ax.set_title(head + (f"\nbias: {bias}" if bias else ""), fontsize=9)
    for ax in axes.flat[len(ids):]:
        ax.set_visible(False)
    h1, l1 = axes.flat[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(title or "Example series: units (left), price (right); grey = test window")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()


def plot_series_panels(panel: pl.DataFrame, uid: str, history_weeks: int = 52) -> None:
    """F1b: units above, price below, sharing the date axis. No twin-axis scale to misread."""
    s = gd.only(panel, [uid]).to_pandas()
    s["ds"] = pd.to_datetime(s["ds"])
    s = s[s["ds"] >= pd.Timestamp(bc.TEST_START) - pd.Timedelta(weeks=history_weeks)]
    fig, (au, ap) = plt.subplots(2, 1, sharex=True, figsize=(12, 5),
                                 gridspec_kw={"height_ratios": [3, 1.3]})
    for a, b in _promo_spans(s["ds"], s["promo"] > 0):
        for ax in (au, ap):
            ax.axvspan(a, b, color=COLOURS["promo"], alpha=0.18, lw=0)
    au.plot(s["ds"], s["y"], color=COLOURS["actual"], lw=1.4, marker="o", ms=2)
    au.set_ylabel("units / week")
    ap.step(s["ds"], s["sale_price"], where="mid", color=COLOURS["price"], label="sale price")
    ap.step(s["ds"], s["regular_price"], where="mid", color=COLOURS["regular"], ls="--",
            label="regular price")
    ap.set_ylabel("price")
    ap.legend(loc="upper left")
    for ax in (au, ap):
        ax.axvline(pd.Timestamp(bc.TEST_START), color="0.4", ls=":")
    fig.suptitle(f"{uid}: the same data as two panels (promo weeks shaded)")
    fig.tight_layout()
    plt.show()


# ============================================================================
# F2 / F3 / F4: accuracy
# ============================================================================

def plot_error_dist(wide: pd.DataFrame, info: pd.DataFrame, models: list[str] | None = None,
                    metric: str = "RMSSE") -> None:
    """F2: per-series error as a histogram (log x), an ECDF, and a box plot by sales class.

    ``wide`` is ``gd.per_series(metrics)``: one row per series, one column per model.
    """
    models = models or list(wide.columns)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    pos = wide[models].clip(lower=1e-3)
    bins = np.logspace(np.log10(pos.min().min()), np.log10(pos.max().max()), 60)
    for i, m in enumerate(models):
        v = pos[m].dropna()
        axes[0].hist(v, bins=bins, histtype="step", lw=1.5, color=_colour(i), label=m)
        axes[0].axvline(v.median(), color=_colour(i), ls="--", lw=1)
        x = np.sort(v)
        axes[1].plot(x, np.arange(1, len(x) + 1) / len(x), color=_colour(i), lw=1.5, label=m)
    axes[0].set_xscale("log")
    axes[0].set(title=f"{metric} per series (dashed = median)", xlabel=metric, ylabel="series")
    axes[1].set_xscale("log")
    axes[1].axvline(1, color="0.5", ls=":")
    axes[1].set(title="ECDF: share of series at or below", xlabel=metric, ylabel="share")
    axes[1].legend()
    joined = wide[models].join(info["sale_class"])
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        data = [joined.loc[joined["sale_class"] == c, m].dropna() for c in CLASS_ORDER]
        bp = axes[2].boxplot(data, positions=np.arange(10) + i * width, widths=width * 0.9,
                             showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set(facecolor=_colour(i), alpha=0.5)
    axes[2].set_xticks(np.arange(10) + 0.4 - width / 2, CLASS_ORDER)
    handles = []
    for i in range(len(models)):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=_colour(i), alpha=0.5))
    axes[2].legend(handles, models)
    axes[2].set(title=f"{metric} by sales class (A = top sellers)", xlabel="class")
    fig.tight_layout()
    plt.show()


def plot_heatmap(table: pd.DataFrame, title: str, fmt: str = ".2f", cmap: str = "RdYlGn_r",
                 label: str = "", text: pd.DataFrame | None = None, figsize=None,
                 xlabel: str = "", ylabel: str = "", vmin=None, vmax=None,
                 xrotation: int = 0) -> None:
    """A labelled heatmap of ``table`` (rows down, columns across), each cell written in.

    Cells show ``format(value, fmt)``, or the matching cell of ``text`` if given (e.g.
    "mean ± sd"). NaN cells are left blank. With the default colour map greener is lower.
    """
    values = table.to_numpy(dtype=float)
    if figsize is None:
        figsize = (max(6, 1.1 * table.shape[1] + 3), 0.5 * table.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ha = "right" if xrotation else "center"
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns], rotation=xrotation,
                  ha=ha)
    ax.set_yticks(range(table.shape[0]), [str(r) for r in table.index])
    for (r, c), v in np.ndenumerate(values):
        if np.isnan(v):
            continue
        cell = text.iloc[r, c] if text is not None else format(v, fmt)
        ax.text(c, r, cell, ha="center", va="center", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    plt.show()


def plot_class_heatmap(wide: pd.DataFrame, info: pd.DataFrame,
                       models: list[str] | None = None, metric: str = "RMSSE") -> None:
    """F3: median error by sales class (columns) and model (rows); greener is better."""
    models = models or list(wide.columns)
    table = wide[models].join(info["sale_class"]).groupby("sale_class")[models].median().T
    table = table[[c for c in CLASS_ORDER if c in table.columns]]
    plot_heatmap(table, f"Median {metric} by sales class (A = top 10% of units)",
                 label=f"median {metric}", figsize=(10, 0.5 * len(models) + 1.5))


def win_table(wide: pd.DataFrame, info: pd.DataFrame, model: str, baseline: str) -> pd.DataFrame:
    """Share of series where ``model`` beats ``baseline``, by class: by count and by volume."""
    d = wide[[model, baseline]].dropna().join(info[["sale_class", "mean_weekly_units"]])
    d["win"] = d[model] < d[baseline]
    d["vol"] = d["mean_weekly_units"].fillna(0)
    d["vol_won"] = d["vol"] * d["win"]
    g = d.groupby("sale_class")
    out = pd.DataFrame({"win_share": g["win"].mean(),
                        "win_share_volume": g["vol_won"].sum() / g["vol"].sum().clip(lower=1e-9),
                        "n": g.size()})
    out.loc["all"] = [d["win"].mean(), (d["vol"] * d["win"]).sum() / d["vol"].sum(), len(d)]
    return out


def plot_wins(wide: pd.DataFrame, info: pd.DataFrame, model: str, baseline: str,
              metric: str = "RMSSE") -> pd.DataFrame:
    """F4: marginal wins of ``model`` over ``baseline``; returns the win table.

    (a) share of series won, by class, by count and weighted by volume;
    (b) per-series improvement ``(baseline - model) / baseline``;
    (c) head to head, one point per series (below the line, ``model`` is better).
    """
    table = win_table(wide, info, model, baseline)
    d = wide[[model, baseline]].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    by_class = table.drop(index="all").reindex(CLASS_ORDER).dropna(how="all")
    x = np.arange(len(by_class))
    axes[0].bar(x - 0.2, by_class["win_share"], 0.4, color=_colour(0), label="by count")
    axes[0].bar(x + 0.2, by_class["win_share_volume"], 0.4, color=_colour(1), label="by volume")
    axes[0].axhline(0.5, color="0.3", ls="--", lw=1)
    axes[0].set_xticks(x, by_class.index)
    axes[0].set(ylim=(0, 1), title=f"{model} beats {baseline}: share of series",
                xlabel="class", ylabel="share won")
    axes[0].legend()
    gain = ((d[baseline] - d[model]) / d[baseline].replace(0, np.nan)).clip(-1, 1).dropna()
    bins = np.linspace(-1, 1, 41)
    axes[1].hist(gain[gain > 0], bins=bins, color=COLOURS["good"], label=f"{model} better")
    axes[1].hist(gain[gain <= 0], bins=bins, color=COLOURS["bad"], label=f"{baseline} better")
    axes[1].axvline(gain.median(), color="0.2", ls="--", lw=1,
                    label=f"median {gain.median():+.0%}")
    axes[1].set(title="Per-series improvement (clipped to ±100%)", xlabel="(base - model) / base")
    axes[1].legend()
    lim = (max(d.min().min(), 1e-2), d.max().max())
    axes[2].scatter(d[baseline], d[model], s=6, alpha=0.4, color=_colour(0))
    axes[2].plot(lim, lim, color="0.3", ls="--", lw=1)
    axes[2].set(xscale="log", yscale="log", xlabel=f"{baseline} {metric}",
                ylabel=f"{model} {metric}", title="Head to head (below the line: model better)")
    fig.tight_layout()
    plt.show()
    return table.round(3)


# ============================================================================
# F5 / F6 / F7: price response
# ============================================================================

def plot_elasticity(el: pd.DataFrame, info: pd.DataFrame, band=None,
                    title: str = "") -> None:
    """F5: (a) arc elasticity at ±20% (``gp.report_shift``), (b) by class, (c) median against
    shift size.

    ``band`` (e.g. ``gp.empirical_band(emp)``) is shaded on every panel when given.
    """
    at = gp.report_shift(el)
    top = el[np.isclose(el["shift"], at)].set_index("unique_id")
    e = top["e_arc"].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].hist(e.clip(-5, 2), bins=70, color=_colour(0), alpha=0.85)
    if band:
        axes[0].axvspan(*band, color=COLOURS["band"], alpha=0.15, label="empirical band")
    axes[0].axvline(e.median(), color=COLOURS["bad"], ls="--", label=f"median {e.median():.2f}")
    resp = e[e.abs() > gp.FLAT_TOL]
    if len(resp):
        axes[0].axvline(resp.median(), color=COLOURS["price"], ls=":",
                        label=f"median responsive {resp.median():.2f}")
    axes[0].axvline(0, color="0.2", lw=0.8)
    axes[0].set(title=f"Arc elasticity at ±{at:.0%}", xlabel="elasticity",
                ylabel="series")
    axes[0].legend()
    joined = top.join(info["sale_class"])
    order = [c for c in gp.REPORT_CLASSES if c in set(joined["sale_class"])]
    axes[1].boxplot([joined.loc[joined["sale_class"] == c, "e_arc"].dropna() for c in order],
                    showfliers=False)
    axes[1].set_xticks(range(1, len(order) + 1), order)
    if band:
        axes[1].axhspan(*band, color=COLOURS["band"], alpha=0.15)
    axes[1].axhline(0, color=COLOURS["bad"], lw=0.8)
    axes[1].set(title="By sales class", xlabel="class", ylabel="elasticity")
    med = el.groupby("shift")[["e_arc", "e_down", "e_up"]].median()
    for i, c in enumerate(med.columns):
        axes[2].plot(med.index * 100, med[c], "o-", color=_colour(i), label=c)
    if band:
        axes[2].axhspan(*band, color=COLOURS["band"], alpha=0.15)
    axes[2].set(title="Median against shift size (flat = invariant)", xlabel="price shift, %",
                ylabel="median elasticity")
    axes[2].legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_elasticity_strength(arms: dict[str, pd.Series], empirical: pd.Series | None = None,
                             title: str = "") -> pd.DataFrame:
    """F6: % of series in each elasticity category, one bar per arm; returns the table.

    ``arms`` maps an arm name to per-series elasticities (at one shift size). Rows are sorted
    by perverse share; an "empirical" row is added if given.
    """
    rows = {name: gp.categorise(e.dropna()).value_counts(normalize=True) * 100
            for name, e in arms.items()}
    table = pd.DataFrame(rows).T.reindex(columns=list(CATEGORY_COLOURS)).fillna(0)
    table = table.sort_values("Perverse", ascending=False).rename_axis(None)
    if empirical is not None:
        emp = gp.categorise(empirical.dropna()).value_counts(normalize=True) * 100
        table.loc["empirical"] = emp.reindex(table.columns).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(table) + 1.6))
    left = np.zeros(len(table))
    for cat, colour in CATEGORY_COLOURS.items():
        ax.barh(table.index, table[cat], left=left, color=colour, edgecolor="white", label=cat)
        left += table[cat].to_numpy()
    ax.set(xlim=(0, 100), xlabel="% of series", title=title or "Strength of price response")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(False)
    fig.tight_layout()
    plt.show()
    return table.round(1)


def plot_elasticity_by_class(arms: dict[str, pd.Series], info: pd.DataFrame,
                             empirical: pd.DataFrame | None = None) -> None:
    """F7: median elasticity by sales class, one line per arm; empirical dashed, zero dotted."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, e) in enumerate(arms.items()):
        med = e.to_frame("e").join(info["sale_class"]).groupby("sale_class")["e"].median()
        med = med.reindex(gp.REPORT_CLASSES)
        ax.plot(med.index, med.values, "o-", color=_colour(i), label=name)
    if empirical is not None:
        emp = empirical.set_index("sale_class")["elasticity"].reindex(gp.REPORT_CLASSES)
        ax.plot(emp.index, emp.values, "s--", color="black", lw=2, label="empirical")
    ax.axhline(0, color=COLOURS["bad"], ls=":")
    ax.set(xlabel="sales class", ylabel="median elasticity",
           title="Median elasticity by class (more negative = stronger response)")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ============================================================================
# F8 / F9: training
# ============================================================================

def plot_training(runs: list[dict], warmup: int | None = None, title: str = "") -> None:
    """F8: train and validation loss (log scale) against step, one panel per run.

    Each run is a ``gd.fit_seed`` result. Marks the end of warm-up and the best-weights step.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.6), squeeze=False)
    for ax, run in zip(axes.flat, runs):
        t, s = run["trajectory"], run["summary"]
        ax.plot(t["step"], t["train_loss"], color=_colour(0), lw=1, alpha=0.7, label="train")
        va = t.dropna(subset=["valid_loss"])
        ax.plot(va["step"], va["valid_loss"], color=_colour(1), lw=1.6, label="valid")
        if warmup:
            ax.axvline(warmup, color="0.5", ls=":", label="warm-up ends")
        if s.get("best_step") is not None:
            ax.axvline(s["best_step"], color=COLOURS["good"], ls="--", label="weights kept")
        ax.set(yscale="log", xlabel="step", title=f"{s['alias']} · seed {s['seed']}")
    axes.flat[0].set_ylabel("loss")
    axes.flat[0].legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_arm_comparison(arms: dict[str, list[dict]], metrics: dict[str, pd.Series] | None = None,
                        elasticity: dict[str, list[float]] | None = None, band=None) -> None:
    """F9: every seed's validation curve, then box-and-jitter by arm.

    ``arms`` maps an arm name to its ``gd.fit_seeds`` runs. ``metrics`` maps arm name to
    per-seed median RMSSE; ``elasticity`` to per-seed median elasticity (optional panel).
    """
    panels = 3 if elasticity else 2
    fig, axes = plt.subplots(1, panels + 1, figsize=(4.6 * (panels + 1), 3.8))
    for i, (name, runs) in enumerate(arms.items()):
        for j, run in enumerate(runs):
            va = run["trajectory"].dropna(subset=["valid_loss"])
            axes[0].plot(va["step"], va["valid_loss"], color=_colour(i), lw=1, alpha=0.8,
                         label=name if j == 0 else None)
    axes[0].set(yscale="log", xlabel="step", ylabel="valid loss", title="Validation, every seed")
    axes[0].legend()

    def box(ax, values: dict, label: str):
        names = list(values)
        ax.boxplot([values[n] for n in names], showfliers=False)
        for k, n in enumerate(names):
            jitter = np.random.default_rng(k).uniform(-0.08, 0.08, len(values[n]))
            ax.scatter(np.full(len(values[n]), k + 1) + jitter, values[n], color="black", s=12)
        ax.set_xticks(range(1, len(names) + 1), names, rotation=15)
        ax.set_title(label)

    osc = {n: [r["summary"].get("oscillation", np.nan) for r in runs] for n, runs in arms.items()}
    box(axes[1], osc, "Oscillation (lower = steadier)")
    if metrics:
        box(axes[2], {n: list(v) for n, v in metrics.items()}, "Median RMSSE per seed")
    if elasticity:
        box(axes[3], elasticity, "Median elasticity per seed")
        if band:
            axes[3].axhspan(*band, color=COLOURS["band"], alpha=0.15)
    fig.tight_layout()
    plt.show()


# ============================================================================
# F10 / F11: selection and probes
# ============================================================================

def plot_pareto(trials: pd.DataFrame, front: pd.DataFrame, knees: dict,
                incumbent=None, x: str = "accuracy", y: str = "elasticity") -> None:
    """F10: every trial (infeasible grey), the Pareto front joined, knees and incumbent marked.

    ``trials`` needs ``x``, ``y`` and a boolean ``feasible`` column; ``knees`` maps a method
    name to a row index of ``front``.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bad = trials[~trials["feasible"]]
    ok = trials[trials["feasible"]]
    ax.scatter(bad[x], bad[y], color="0.75", s=18, label="infeasible")
    ax.scatter(ok[x], ok[y], color=_colour(0), s=22, alpha=0.6, label="feasible")
    ax.plot(front[x], front[y], "o-", color=COLOURS["bad"], lw=1.5, label="Pareto front")
    markers = ["*", "P", "X"]
    for k, (method, idx) in enumerate(knees.items()):
        ax.scatter(front.loc[idx, x], front.loc[idx, y], marker=markers[k % 3], s=220,
                   color="black", label=f"knee ({method})", zorder=5)
    if incumbent is not None:
        ax.scatter(*incumbent, marker="D", s=80, facecolor="none", edgecolor="black",
                   label="incumbent")
    ax.set(xlabel=f"{x} (lower is better)", ylabel=f"{y} (more negative = stronger response)",
           title="Accuracy against price response")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_probe(summary: pd.DataFrame, title: str = "") -> None:
    """F11: median lift by shock size, sustained vs pulse; spill for the pulse."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colour = {mode: _colour(i) for i, mode in enumerate(sorted(summary["mode"].unique()))}
    for mode, g in summary.groupby("mode"):
        g = g.sort_values("shock")
        axes[0].plot(g["shock"] * 100, g["median_lift"] * 100, "o-", color=colour[mode],
                     label=mode)
    axes[0].axhline(0, color="0.3", lw=0.8)
    axes[0].axvline(0, color="0.3", lw=0.8)
    axes[0].set(xlabel="price shock, %", ylabel="median lift in units, %",
                title="Lift by shock (a cut should lift sales)")
    axes[0].legend()
    pulse_modes = [m for m in summary["mode"].unique() if str(m).startswith("pulse")]
    for mode in pulse_modes:
        g = summary[summary["mode"] == mode].sort_values("shock")
        axes[1].plot(g["shock"] * 100, g["median_spill"] * 100, "o-", color=colour[mode],
                     label=mode)
    axes[1].axhline(0, color="0.3", lw=0.8)
    if len(pulse_modes) > 1:
        axes[1].legend()
    axes[1].set(xlabel="price shock, %", ylabel="median change in other weeks, %",
                title="Pulse spill into weeks not shocked (ideally ~0)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_lift_by_depth(curves: dict[str, pd.Series], references: dict[str, pd.Series] | None = None,
                       title: str = "Lift by cut depth (random promotion calendars)") -> None:
    """F13: median lift in the cut week against the depth of a one-week cut.

    ``curves``: model -> Series of median lift indexed by depth (``calendar_report``), drawn
    solid. ``references``: name -> Series of lift by depth (the planted lift, the lift the
    empirical elasticity implies), drawn dashed in greys.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, lift) in enumerate(curves.items()):
        ax.plot(lift.index * 100, lift.values * 100, "o-", color=_colour(i), lw=1.8, label=name)
    for i, (name, lift) in enumerate((references or {}).items()):
        ax.plot(lift.index * 100, lift.values * 100, "s--", color=("0.15", "0.55")[i % 2],
                lw=1.5, label=name)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set(xlabel="one-week price cut, %", ylabel="median lift in that week, %", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
