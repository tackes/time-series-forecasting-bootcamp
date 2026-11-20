"""Price, promotion and event helpers for the full-data Day 2 notebooks.

What is here, and the research finding behind it (``docs/gfm_learnings.md``):

- ``deflate_prices``: CPI deflation, so inflation does not look like a price rise (§3).
- ``add_events``: week dummies for Easter, Anzac Day, Christmas and Mother's Day. Smooth
  Fourier terms cannot place a moving holiday such as Easter (§7).
- ``add_discount_flag`` and ``oversample``: copies of series with real discount weeks, so the
  network sees more price variation (§3). Copies are for training only.
- ``price_sweep`` and ``arc_elasticity``: the standard sweep. Move **one** price channel by
  ±5/10/20%, keep everything else as observed, and report midpoint elasticities (§5).
- ``elasticity_report``, ``categorise`` and ``empirical_elasticity``: how we read a sweep.
- ``probe``: sustained and one-week (pulse) price shocks (§5, phase 7).
- ``add_synthetic_cuts``: one-week cuts with a planted lift, added to the copies only (05b).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
from dateutil.easter import easter

import gd_helpers as gd

bc = gd.bc

SHIFTS = (0.05, 0.10, 0.20)          # sweep magnitudes
MIN_UNITS = 3.0                      # report elasticity only for series selling >= 3 units/week
REPORT_CLASSES = list("ABCDEFGHI")   # class J is mostly noise at these volumes
FLAT_TOL = 0.05                      # |e| below this counts as flat
RESEARCH_BAND = (-0.68, -0.40)       # empirical range on the 100k set, classes A-I (not this data)

# Elasticity categories for the strength chart (F6). Upper edges, in order.
CATEGORIES = [("Highly elastic", -3.0), ("Elastic", -1.0), ("Inelastic", -FLAT_TOL),
              ("Flat", 0.0), ("Perverse", np.inf)]

# Oversampling (research phases 2-6): copies by number of discount weeks, × class weight,
# × multiplier, capped. "flat_core" weights: none for A, I, J; most for the middle classes.
COPY_TIERS = [(10, 25, 4), (25, 50, 2), (50, 10_000, 1)]
FLAT_CORE = {"A": 0.0, "B": 1.0, "C": 1.5, "D": 1.5, "E": 1.5, "F": 1.5, "G": 1.5,
             "H": 1.0, "I": 0.0, "J": 0.0}
COPY_MULTIPLIER = 5
COPY_CAP = 15
COPY_TAG = "__copy"

# Planted lift for synthetic cuts (05b): discount depth -> extra share of units.
SYNTHETIC_LIFT = {0.10: 0.35, 0.15: 0.45, 0.20: 0.60, 0.25: 1.00, 0.30: 1.50}


# ============================================================================
# PRICE FEATURES
# ============================================================================

def _price_index(df: pl.DataFrame) -> pl.DataFrame:
    """Price relative to the series' median training price (as in ``bootcamp.load_panel``)."""
    ref = (df.filter(pl.col("ds") < bc.TEST_START).group_by("unique_id")
           .agg(pl.col("sale_price").median().alias("_ref")))
    return (df.drop("price_index", strict=False).join(ref, on="unique_id")
            .with_columns((pl.col("sale_price") / pl.col("_ref")).alias("price_index"))
            .drop("_ref"))


def deflate_prices(df: pl.DataFrame, path=gd.CPI_PATH) -> pl.DataFrame:
    """Divide prices by the CPI deflator for their week, then rebuild the price features.

    The deflator is 1 in the base week, so prices are in base-week money.
    """
    cpi = pl.read_csv(path, try_parse_dates=True).select(
        pl.col("ds").cast(pl.Date), pl.col("deflator").cast(pl.Float64))
    out = df.join(cpi, on="ds", how="left")
    missing = out["deflator"].null_count()
    if missing:
        raise ValueError(f"{missing} rows have no CPI deflator; extend {path.name}.")
    out = out.with_columns(
        (pl.col("sale_price") / pl.col("deflator")).alias("sale_price"),
        (pl.col("regular_price") / pl.col("deflator")).alias("regular_price"),
    ).drop("deflator")
    out = bc._add_price_features(out.drop("price_ratio_to_rolling_mean", "price_pct_change_1w",
                                          "discount_depth"))
    return _price_index(out)


def add_discount_flag(df: pl.DataFrame, rule: str = "rolling", depth: float = 0.10) -> pl.DataFrame:
    """Add ``is_discount``: a real price cut of at least ``depth``.

    ``rule="rolling"`` (default) compares with the 4-week rolling mean (``discount_depth``);
    ``rule="regular"`` with the regular price, which is unreliable here (about 30% of weeks
    are priced above it). The promo flag alone is a poor guide: in the research
    it barely correlated with real discounts.
    """
    if rule == "regular":
        cut = pl.col("sale_price") < (1 - depth) * pl.col("regular_price")
    else:
        cut = pl.col("discount_depth") >= depth
    return df.with_columns(cut.fill_null(False).alias("is_discount"))


# ============================================================================
# EVENTS
# ============================================================================

def week_start(d: date) -> date:
    """The Wednesday that starts the bootcamp week (``W-WED``) containing ``d``."""
    return d - timedelta(days=(d.weekday() - 2) % 7)


def event_dates(year: int) -> dict[str, date]:
    """Retail events used as dummies. Easter and Mother's Day move every year."""
    may_first = date(year, 5, 1)
    mothers_day = may_first + timedelta(days=(6 - may_first.weekday()) % 7 + 7)  # 2nd Sunday
    return {"ev_easter": easter(year) - timedelta(days=2),        # Good Friday
            "ev_anzac": date(year, 4, 25),
            "ev_mothers_day": mothers_day,
            "ev_christmas": date(year, 12, 25)}


EVENT_COLS = list(event_dates(2024))


def add_events(df: pl.DataFrame) -> pl.DataFrame:
    """Add one 0/1 column per event: 1 in the week that contains it."""
    years = range(df["ds"].min().year, df["ds"].max().year + 1)
    rows = [{"ds": week_start(d), "event": name}
            for y in years for name, d in event_dates(y).items()]
    weeks = (pl.DataFrame(rows).with_columns(pl.lit(1.0).alias("on"))
             .pivot(on="event", index="ds", values="on", aggregate_function="max"))
    out = df.join(weeks.with_columns(pl.col("ds").cast(df.schema["ds"])), on="ds", how="left")
    return out.with_columns([pl.col(c).fill_null(0.0) for c in EVENT_COLS])


def add_fourier(df: pl.DataFrame, k: int = 2, period: float = 52.18) -> pl.DataFrame:
    """Week-of-year sine and cosine pairs (``k`` harmonics), known for any future week."""
    weeks = pl.col("ds").dt.ordinal_day().cast(pl.Float64) / 7.0
    cols = []
    for h in range(1, k + 1):
        angle = 2 * np.pi * h * weeks / period
        cols += [angle.sin().alias(f"fourier_sin{h}"), angle.cos().alias(f"fourier_cos{h}")]
    return df.with_columns(cols)


def fourier_cols(k: int) -> list[str]:
    """Column names ``add_fourier`` creates."""
    return [f"fourier_{name}{h}" for h in range(1, k + 1) for name in ("sin", "cos")]


def hierarchy_statics(panel: pl.DataFrame, level: str = "department_id",
                      top: int = 12) -> pd.DataFrame:
    """One-hot statics for the ``top`` largest groups at ``level`` (the rest share "other").

    Raw ids are labels, not quantities, so they are one-hot encoded rather than fed as numbers.
    """
    size = (panel.group_by(level).agg(pl.col("unique_id").n_unique().alias("n"))
            .sort("n", descending=True))
    keep = size[level].head(top).to_list()
    ids = panel.select("unique_id", level).unique("unique_id").to_pandas()
    ids[level] = ids[level].where(ids[level].isin(keep), other=-1)
    dummies = pd.get_dummies(ids[level], prefix=level.replace("_id", "")).astype("float32")
    return pd.concat([ids[["unique_id"]], dummies], axis=1)


# ============================================================================
# FROM A CONFIG TO TRAINING DATA
# ============================================================================

def prepare(cfg: dict, panel: pl.DataFrame | None = None, synthetic_seed: int | None = None):
    """Rebuild a notebook's training data from its config.

    Reads these optional config keys (all default off):

    - ``features``: ``{"deflate": bool, "events": bool, "fourier_k": int,
      "hierarchy_level": str | None}``;
    - ``futr_exog``: future covariates (default ``bc.FUTR_EXOG_COLS``);
    - ``oversample``: ``{"rule": "rolling" | "regular", "cap": int, "multiplier": int}``;
    - ``synthetic``: ``{"cuts_per_copy": int}`` (needs ``oversample``);
    - ``y_shift``: target shift (default ``gd.Y_SHIFT``).

    Returns ``(panel, train, test, data)``. ``panel``, ``train`` and ``test`` hold the original
    series only; ``data.fit_df`` also holds any copies.
    """
    feats = cfg.get("features", {})
    panel = panel if panel is not None else gd.train_panel()
    if feats.get("deflate"):
        panel = deflate_prices(panel)
    if feats.get("events"):
        panel = add_events(panel)
    if feats.get("fourier_k"):
        panel = add_fourier(panel, feats["fourier_k"])
    over = cfg.get("oversample")
    panel = add_discount_flag(panel, rule=(over or {}).get("rule", "rolling"))
    train, test = bc.split(panel, cutoff=bc.TEST_START)

    fit = train
    if over:
        copies = copies_per_series(train, multiplier=over.get("multiplier", COPY_MULTIPLIER),
                                   cap=over.get("cap", COPY_CAP))
        fit = oversample(train, copies)
        if cfg.get("synthetic"):
            fit = add_synthetic_cuts(fit, cfg["synthetic"].get("cuts_per_copy", 2),
                                     seed=synthetic_seed or bc.RANDOM_SEED)
    extra = None
    if feats.get("hierarchy_level"):
        extra = hierarchy_statics(panel, feats["hierarchy_level"])
        if over:                     # copies share their original series' statics
            ids = fit.select("unique_id").unique().to_pandas()
            ids["orig"] = ids["unique_id"].str.split(COPY_TAG).str[0]
            extra = (ids.merge(extra.rename(columns={"unique_id": "orig"}), on="orig")
                     .drop(columns="orig"))
    data = gd.nixtla_data(fit, test, futr_exog=cfg.get("futr_exog", bc.FUTR_EXOG_COLS),
                          y_shift=cfg.get("y_shift", gd.Y_SHIFT), extra_statics=extra)
    return panel, train, test, data


# ============================================================================
# OVERSAMPLING
# ============================================================================

def copies_per_series(train: pl.DataFrame, weights: dict = FLAT_CORE,
                      multiplier: int = COPY_MULTIPLIER, cap: int = COPY_CAP) -> pl.DataFrame:
    """How many extra copies each series gets, from its discount weeks in **training** only."""
    counts = (train.filter(pl.col("ds") < bc.TEST_START)
              .group_by("unique_id")
              .agg(pl.col("is_discount").sum().alias("discount_weeks"),
                   pl.col("sale_class").first()))
    tier = pl.lit(0)
    for low, high, n in COPY_TIERS:
        tier = pl.when(pl.col("discount_weeks").is_between(low, high, closed="left")).then(n) \
                 .otherwise(tier)
    weight = pl.col("sale_class").replace_strict(weights, default=0.0)
    n_copies = (tier * weight * multiplier).round().clip(0, cap).cast(pl.Int32)
    return counts.with_columns(n_copies.alias("copies"))


def oversample(train: pl.DataFrame, copies: pl.DataFrame) -> pl.DataFrame:
    """Append the copies to the training frame. Copy ids end in ``__copy<k>``."""
    frames = [train]
    for k in range(1, int(copies["copies"].max() or 0) + 1):
        ids = copies.filter(pl.col("copies") >= k)["unique_id"]
        frames.append(train.filter(pl.col("unique_id").is_in(ids.implode()))
                      .with_columns((pl.col("unique_id") + f"{COPY_TAG}{k}").alias("unique_id")))
    return pl.concat(frames)


def drop_copies(df):
    """Remove oversampled copies (before scoring or plotting)."""
    if isinstance(df, pl.DataFrame):
        return df.filter(~pl.col("unique_id").str.contains(COPY_TAG, literal=True))
    return df[~df["unique_id"].astype(str).str.contains(COPY_TAG, regex=False)]


def add_synthetic_cuts(train: pl.DataFrame, cuts_per_copy: int = 2, seed: int = bc.RANDOM_SEED,
                       lift: dict = SYNTHETIC_LIFT) -> pl.DataFrame:
    """Plant one-week price cuts in ordinary weeks of the **copies** (originals untouched).

    Each chosen week gets a cut of a random depth from ``lift``, its price features updated,
    and its units raised by the planted lift. ``synthetic_depth`` records what was planted.
    """
    rng = np.random.default_rng(seed)
    is_copy = pl.col("unique_id").str.contains(COPY_TAG, literal=True)
    ordinary = is_copy & (pl.col("promo") == 0) & (~pl.col("is_discount")) & \
        (pl.col("ds") < bc.TEST_START)
    candidates = train.with_row_index("_row").filter(ordinary).select("_row", "unique_id")
    picked = (candidates.with_columns(pl.Series("_r", rng.random(candidates.height)))
              .sort("_r").group_by("unique_id").head(cuts_per_copy)["_row"])
    depths = rng.choice(list(lift), size=len(picked))
    plan = pl.DataFrame({"_row": picked, "synthetic_depth": depths}).with_columns(
        pl.col("synthetic_depth").replace_strict(lift, return_dtype=pl.Float64).alias("_lift"))
    out = train.with_row_index("_row").join(plan, on="_row", how="left").with_columns(
        pl.col("synthetic_depth").fill_null(0.0), pl.col("_lift").fill_null(0.0))
    cut = 1 - pl.col("synthetic_depth")
    return out.with_columns(
        (pl.col("sale_price") * cut).alias("sale_price"),
        (pl.col("price_index") * cut).alias("price_index"),
        pl.max_horizontal(pl.col("discount_depth"), pl.col("synthetic_depth")).alias("discount_depth"),
        (pl.col("y") * (1 + pl.col("_lift"))).round().alias("y"),
    ).drop("_row", "_lift")


# ============================================================================
# PRICE SWEEP AND ELASTICITY
# ============================================================================

def report_ids(train: pl.DataFrame, ids: list[str] | None = None) -> list[str]:
    """Series we report elasticity for: classes A-I selling at least ``MIN_UNITS`` a week."""
    ids = ids if ids is not None else gd.eval_ids()
    keep = (train.filter(pl.col("unique_id").is_in(ids) & (pl.col("ds") < bc.TEST_START))
            .group_by("unique_id")
            .agg(pl.col("y").mean().alias("mean_y"), pl.col("sale_class").first())
            .filter((pl.col("mean_y") >= MIN_UNITS) & pl.col("sale_class").is_in(REPORT_CLASSES)))
    return sorted(keep["unique_id"].to_list())


def shift_prices(futr: pd.DataFrame, shift: float, channel: str = "price_index",
                 compound: bool = False, weeks=None) -> pd.DataFrame:
    """Future covariates with price moved by ``shift`` (e.g. -0.10 for a 10% cut).

    Single channel (the default and the research's final method) moves ``channel`` only.
    ``compound=True`` also recomputes ``discount_depth`` from the moved price, as Lei's lab 03
    does; the research found that version can flip the sign. ``weeks`` limits the move to
    some forecast weeks (0-based), for a one-week pulse.
    """
    out = futr.copy()
    step = out.groupby("unique_id").cumcount()
    on = step.isin(weeks) if weeks is not None else np.ones(len(out), dtype=bool)
    out.loc[on, channel] = out.loc[on, channel] * (1 + shift)
    if compound and "discount_depth" in out:
        ratio = (1 - out["discount_depth"]) * np.where(on, 1 + shift, 1.0)
        out["discount_depth"] = np.maximum(0.0, 1 - ratio)
    return out


def price_sweep(nf, data, ids: list[str], shifts=SHIFTS, channel: str = "price_index",
                compound: bool = False) -> pd.DataFrame:
    """Total forecast units over the horizon at base price and at ±each shift.

    Returns long format: ``unique_id``, ``shift`` (0, ±m), ``units``. Forecasts are unclipped
    and have the target shift removed, as the research's method requires.
    """
    sub = gd.NixtlaData(gd.only(data.fit_df, ids), gd.only(data.predict_df, ids),
                        gd.only(data.futr_df, ids), data.static_df, data.futr_exog,
                        data.stat_exog, data.y_shift)
    rows = []
    for s in [0.0] + [sign * m for m in shifts for sign in (-1, 1)]:
        futr = shift_prices(sub.futr_df, s, channel, compound)
        fcst = gd.predict(nf, sub, futr_df=futr)
        col = bc.model_columns(fcst)[0]
        total = fcst.groupby("unique_id")[col].sum().rename("units").reset_index()
        rows.append(total.assign(shift=s))
    return pd.concat(rows, ignore_index=True)


def arc_elasticity(sweep: pd.DataFrame) -> pd.DataFrame:
    """Midpoint (arc) elasticities per series and shift size from a ``price_sweep``.

    ``e_arc = (q_up - q_down) / mean(q_up, q_down) / (2m)``; ``e_down`` and ``e_up`` are the
    one-sided midpoint versions against the base price.
    """
    wide = sweep.pivot(index="unique_id", columns="shift", values="units")
    q0 = wide[0.0]
    rows = []
    for m in sorted({abs(s) for s in wide.columns if s != 0}):
        q_dn, q_up = wide[-m], wide[m]
        e_arc = (q_up - q_dn) / ((q_up + q_dn) / 2) / (2 * m)
        e_down = ((q_dn - q0) / ((q_dn + q0) / 2)) / (-m / (1 - m / 2))
        e_up = ((q_up - q0) / ((q_up + q0) / 2)) / (m / (1 + m / 2))
        rows.append(pd.DataFrame({"unique_id": wide.index, "shift": m, "e_arc": e_arc.values,
                                  "e_down": e_down.values, "e_up": e_up.values}))
    return pd.concat(rows, ignore_index=True)


def categorise(e: pd.Series) -> pd.Series:
    """Label each elasticity with its strength category (see ``CATEGORIES``)."""
    edges = [-np.inf] + [edge for _, edge in CATEGORIES]
    labels = [name for name, _ in CATEGORIES]
    return pd.cut(e, bins=edges, labels=labels, right=True)


def elasticity_report(el: pd.DataFrame, info: pd.DataFrame | None = None) -> dict:
    """The numbers we always quote together for one model or arm (research §5).

    ``n_estimable`` (non-zero response), median arc elasticity per shift, share perverse
    (e > 0), share flat (|e| < 0.05) and the invariance ratio (median at 5% ÷ median at 20%;
    about 1 for a smooth response).
    """
    by_shift = el.groupby("shift")["e_arc"].median()
    at20 = el[el["shift"] == max(el["shift"])]["e_arc"]
    out = {"n": int(at20.notna().sum()),
           "n_estimable": int((at20.abs() > 1e-6).sum()),
           "median_e20": float(at20.median()),
           "share_perverse": float((at20 > 0).mean()),
           "share_flat": float((at20.abs() < FLAT_TOL).mean()),
           "invariance": float(by_shift.iloc[0] / by_shift.iloc[-1]) if by_shift.iloc[-1] else None}
    for m, v in by_shift.items():
        out[f"median_e{int(round(m * 100))}"] = float(v)
    return out


def empirical_elasticity(train: pl.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Elasticity estimated from history: a two-way within regression of log(y+1) on log price.

    Series and week averages are removed from both sides (alternating demeaning), so the slope
    comes from price changes within a series, net of what all series did that week.
    Returns one row overall and one per sales class.
    """
    d = (train.filter(pl.col("unique_id").is_in(ids) & (pl.col("ds") < bc.TEST_START))
         .select("unique_id", "ds", "sale_class",
                 pl.col("y").log1p().alias("ly"), pl.col("price_index").log().alias("lp"))
         .to_pandas())

    def slope(g: pd.DataFrame) -> float:
        y, x = g["ly"].to_numpy(copy=True), g["lp"].to_numpy(copy=True)
        for _ in range(10):
            for key in ("unique_id", "ds"):
                y = y - pd.Series(y).groupby(g[key].to_numpy()).transform("mean").to_numpy()
                x = x - pd.Series(x).groupby(g[key].to_numpy()).transform("mean").to_numpy()
        return float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else np.nan

    rows = [{"sale_class": "all", "elasticity": slope(d), "n_series": d["unique_id"].nunique()}]
    for sc, g in d.groupby("sale_class"):
        rows.append({"sale_class": sc, "elasticity": slope(g.reset_index(drop=True)),
                     "n_series": g["unique_id"].nunique()})
    return pd.DataFrame(rows)


def empirical_band(emp: pd.DataFrame) -> tuple[float, float]:
    """Inter-quartile range of the per-class empirical elasticities (classes A-I): the band
    drawn on the elasticity charts."""
    v = emp[emp["sale_class"].isin(REPORT_CLASSES)]["elasticity"].dropna()
    return float(v.quantile(0.25)), float(v.quantile(0.75))


# ============================================================================
# PROBES (simulated discounts on a trained model)
# ============================================================================

PROBE_SHOCKS = (-0.10, -0.20, -0.30, 0.20)
PULSE_WEEK = 4


def probe(nf, data, ids: list[str], shocks=PROBE_SHOCKS, channel: str = "price_index",
          pulse_week: int = PULSE_WEEK) -> pd.DataFrame:
    """Lift from sustained and one-week (pulse) price shocks, per series.

    - sustained: the price moves in every forecast week; ``lift`` is the change in total units;
    - pulse: the price moves in week ``pulse_week`` only; ``lift`` is the change in that week
      and ``spill`` the change in all other weeks (ideally about zero).
    """
    sub = gd.NixtlaData(gd.only(data.fit_df, ids), gd.only(data.predict_df, ids),
                        gd.only(data.futr_df, ids), data.static_df, data.futr_exog,
                        data.stat_exog, data.y_shift)
    base = gd.predict(nf, sub)
    col = bc.model_columns(base)[0]
    base["step"] = base.groupby("unique_id").cumcount()
    rows = []
    for shock in shocks:
        for mode, weeks in (("sustained", None), ("pulse", [pulse_week])):
            futr = shift_prices(sub.futr_df, shock, channel, weeks=weeks)
            f = gd.predict(nf, sub, futr_df=futr)
            f["step"] = f.groupby("unique_id").cumcount()
            m = base.merge(f, on=["unique_id", "step"], suffixes=("_0", "_1"))
            on = m["step"] == pulse_week if weeks else np.ones(len(m), dtype=bool)
            q0 = m[on].groupby("unique_id")[f"{col}_0"].sum()
            q1 = m[on].groupby("unique_id")[f"{col}_1"].sum()
            r = pd.DataFrame({"lift": (q1 - q0) / q0.where(q0.abs() > 1e-9)})
            if weeks:
                s0 = m[~on].groupby("unique_id")[f"{col}_0"].sum()
                s1 = m[~on].groupby("unique_id")[f"{col}_1"].sum()
                r["spill"] = (s1 - s0) / s0.where(s0.abs() > 1e-9)
            r["elasticity"] = r["lift"] / shock
            rows.append(r.reset_index().assign(shock=shock, mode=mode))
    return pd.concat(rows, ignore_index=True)


def probe_summary(resp: pd.DataFrame) -> pd.DataFrame:
    """Median lift, elasticity and spill by shock and mode, with the share of right-signed series."""
    right_sign = np.sign(resp["lift"]) == -np.sign(resp["shock"])
    return (resp.assign(right_sign=right_sign)
            .groupby(["mode", "shock"])
            .agg(median_lift=("lift", "median"), median_elasticity=("elasticity", "median"),
                 share_right_sign=("right_sign", "mean"), median_spill=("spill", "median"),
                 n=("lift", "count"))
            .reset_index())
