"""Feature and model settings shared by the Day 1 LightGBM notebooks.

Every ML notebook starts from these settings and changes one thing at a time.
Fourier terms are anchored to the calendar (day of year), so a given week gets the
same value for every series and every cutoff.

The price and promotion helpers below build three kinds of driver (notebook 03):

- planned drivers, known for the forecast weeks when a retailer forecasts its own sales:
  the promotion flag, the regular price and the promotional discount;
- historical drivers, built only from weeks at least H weeks before the target week,
  so they are observed for every week of a 14-week forecast;
- realized selling-price features, kept only as leakage diagnostics: a week with no
  sales gets the regular price, so the realized price reveals the target.
"""

import numpy as np
import pandas as pd
from mlforecast.lag_transforms import RollingMean

from shared.prepare_data import H

LAGS = [1, 2, 3, 4, 8, 13]
LAG_TRANSFORMS = {1: [RollingMean(window_size=4), RollingMean(window_size=13), RollingMean(window_size=26)]}

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbosity=-1)

CATEGORICAL = ["department", "class", "subclass"]


def sin1(dates):
    return np.sin(2 * np.pi * dates.dayofyear / 365.25)


def cos1(dates):
    return np.cos(2 * np.pi * dates.dayofyear / 365.25)


def sin2(dates):
    return np.sin(4 * np.pi * dates.dayofyear / 365.25)


def cos2(dates):
    return np.cos(4 * np.pi * dates.dayofyear / 365.25)


DATE_FEATURES = ["month", sin1, cos1, sin2, cos2]


def item_attributes(panel):
    """Department, class and subclass per series, as strings ("Unknown" when missing)."""
    items = panel.drop_duplicates("unique_id").set_index("unique_id")
    attributes = pd.DataFrame({col: items[f"{col}_id"].astype("Int64").astype("string").fillna("Unknown") for col in CATEGORICAL})
    return attributes.rename_axis("unique_id").reset_index()


def in_horizon(frame, cutoff, horizon=H):
    """Rows of the `horizon` weeks after cutoff."""
    return frame[(frame.ds > cutoff) & (frame.ds <= cutoff + pd.Timedelta(weeks=horizon))]


def past(values, ids, window, how, lag=H, min_periods=1):
    """Rolling statistic per series over the `window` weeks ending `lag` weeks before each row's week."""
    return values.groupby(ids).transform(lambda s: s.shift(lag).rolling(window, min_periods=min_periods).agg(how))


def add_drivers(panel):
    """Price and promotion columns, added to a copy of the prepared panel (sorted by series and week).

    promo           promotion flag
    regular_price   shelf price
    discount        promotional discount: 1 - sale_price / regular_price in promotion weeks, 0 otherwise.
                    In promotion weeks the realized price has the same distribution with or without sales,
                    so it stands in for the planned promotional price.
    planned_price   regular_price * (1 - discount)
    price_index     planned_price over the median regular price of the 52 weeks ending H weeks earlier
    rel_price       realized average selling price / regular price, every week (an outcome, not a plan)
    """
    panel = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    ids = panel.unique_id
    panel["promo"] = panel.is_promotional.astype(int)
    panel["rel_price"] = (panel.sale_price / panel.regular_price.where(panel.regular_price > 0)).fillna(1).clip(0, 3)
    panel["discount"] = ((1 - panel.rel_price) * panel.promo).clip(0, 0.9)
    panel["planned_price"] = panel.regular_price * (1 - panel.discount)
    reference = past(panel.regular_price, ids, 52, "median").fillna(panel.groupby("unique_id").regular_price.transform("first"))
    panel["price_reference"] = reference
    panel["price_index"] = panel.planned_price / reference
    return panel


def promo_features(frame):
    """Promotion-calendar features from a frame of unique_id, ds, promo (history followed by future weeks).

    Each feature at week t uses promotion flags up to week t only.
    """
    frame = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    g = frame.groupby("unique_id")
    frame["promo_lag1"] = g.promo.shift(1).fillna(0)
    frame["promo_lag2"] = g.promo.shift(2).fillna(0)
    frame["promo_freq_13"] = g.promo.transform(lambda s: s.shift(1).rolling(13, min_periods=1).mean()).fillna(0)
    last_promo = frame.ds.where(frame.promo == 1).groupby(frame.unique_id).shift(1).groupby(frame.unique_id).ffill()
    frame["weeks_since_promo"] = ((frame.ds - last_promo).dt.days / 7).fillna(26).clip(upper=26)
    frame["promo_x_gap"] = frame.promo * frame.weeks_since_promo
    return frame


PROMO_CALENDAR = ["promo", "promo_lag1", "promo_lag2", "promo_freq_13", "weeks_since_promo", "promo_x_gap"]


def future_drivers(panel):
    """Planned drivers and promotion-calendar features for every week, as a retailer with a plan would know them."""
    frame = promo_features(panel[["unique_id", "ds", "promo"]])
    cols = ["discount", "price_index", "regular_price", "planned_price"]
    return frame.merge(panel[["unique_id", "ds"] + cols], on=["unique_id", "ds"])


def partial_plan(panel, cutoff, plan_weeks):
    """Planned drivers up to H weeks after cutoff, with the plan known only for the first plan_weeks weeks.

    Later weeks assume no promotion at the regular price observed on the cutoff. Calendar features are
    recomputed from these flags. Weeks up to cutoff + plan_weeks match future_drivers.
    """
    cols = ["unique_id", "ds", "promo", "discount", "regular_price", "price_reference"]
    frame = panel.loc[panel.ds <= cutoff + pd.Timedelta(weeks=H), cols].copy()
    unplanned = frame.ds > cutoff + pd.Timedelta(weeks=plan_weeks)
    last_regular = frame[frame.ds <= cutoff].groupby("unique_id").regular_price.last()
    frame.loc[unplanned, "promo"] = 0
    frame.loc[unplanned, "discount"] = 0.0
    frame.loc[unplanned, "regular_price"] = frame.loc[unplanned, "unique_id"].map(last_regular)
    frame["planned_price"] = frame.regular_price * (1 - frame.discount)
    frame["price_index"] = frame.planned_price / frame.price_reference
    calendar = promo_features(frame[["unique_id", "ds", "promo"]])
    return calendar.merge(frame[["unique_id", "ds", "discount", "price_index", "regular_price", "planned_price"]], on=["unique_id", "ds"])


HISTORICAL = ["promo_lag14", "promo_share_26", "weeks_since_promo_14", "rel_price_lag14", "discount_26",
              "rel_price_mean_13", "rel_price_std_13", "regular_price"]


def historical_drivers(panel):
    """Drivers that are observed for every week of a 14-week forecast.

    Every column except regular_price describes the weeks up to H weeks before the row's week, so for any
    target week up to H weeks after the cutoff it uses data on or before the cutoff. Past realized prices are
    history, so rel_price may be used here. regular_price is the current shelf price; for forecast weeks,
    `historical_future` replaces it with the last value observed on the cutoff.
    """
    ids = panel.unique_id
    rel = panel.rel_price
    last_promo = panel.ds.where(panel.promo == 1).groupby(ids).ffill().groupby(ids).shift(H)
    out = panel[["unique_id", "ds", "regular_price"]].copy()
    out["promo_lag14"] = panel.groupby("unique_id").promo.shift(H)
    out["promo_share_26"] = past(panel.promo.astype(float), ids, 26, "mean")
    out["weeks_since_promo_14"] = ((panel.ds - last_promo).dt.days / 7 - H).clip(upper=52).fillna(52)
    out["rel_price_lag14"] = rel.groupby(ids).shift(H)
    out["discount_26"] = past((1 - rel).clip(lower=0), ids, 26, "mean")
    out["rel_price_mean_13"] = past(rel, ids, 13, "mean")
    out["rel_price_std_13"] = past(rel, ids, 13, "std", min_periods=4)
    return out[["unique_id", "ds"] + HISTORICAL]


def historical_future(hist, cutoff):
    """X_df for the H weeks after cutoff: the lagged columns as they are, regular price carried forward."""
    future = in_horizon(hist, cutoff).copy()
    last_regular = hist[hist.ds <= cutoff].groupby("unique_id").regular_price.last()
    future["regular_price"] = future.unique_id.map(last_regular)
    return future


def realized_drivers(panel):
    """Leakage diagnostic: the same price features as future_drivers, from the realized selling price.

    realized_discount   1 - sale_price / regular_price in every week (clipped to [0, 0.9])
    realized_index      sale_price over the same price reference as price_index
    """
    frame = promo_features(panel[["unique_id", "ds", "promo"]])
    out = panel[["unique_id", "ds"]].copy()
    out["realized_discount"] = (1 - panel.rel_price).clip(0, 0.9)
    out["realized_index"] = panel.sale_price / panel.price_reference
    return frame.merge(out, on=["unique_id", "ds"])


def day2_covariates(panel, cutoff):
    """Day 2's future covariates (day2_neural/code/bootcamp.py), with promo renamed and a cutoff-safe reference.

    day2_discount_depth  max(0, 1 - sale_price / 4-week rolling mean of sale_price, current week included)
    day2_price_index     sale_price / median sale_price of the weeks up to cutoff
    At the final cutoff the reference is Day 2's (weeks before 2025-04-16). Both use the realized price.
    """
    ids = panel.unique_id
    rolling = panel.sale_price.groupby(ids).transform(lambda s: s.rolling(4, min_periods=1).mean())
    reference = panel[panel.ds <= cutoff].groupby("unique_id").sale_price.median()
    out = panel[["unique_id", "ds", "promo"]].copy()
    out["day2_discount_depth"] = (1 - panel.sale_price / rolling).clip(lower=0)
    out["day2_price_index"] = panel.sale_price / ids.map(reference)
    return out
