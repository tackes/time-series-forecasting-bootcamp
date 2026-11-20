"""Feature and model settings shared by the Day 1 LightGBM notebooks.

Every ML notebook starts from these settings and changes one thing at a time.
Fourier terms are anchored to the calendar (day of year), so a given week gets the
same value for every series and every cutoff.
"""

import numpy as np
import pandas as pd
from mlforecast.lag_transforms import RollingMean

LAGS = [1, 2, 3, 4, 8, 13]
LAG_TRANSFORMS = {1: [RollingMean(window_size=4), RollingMean(window_size=13), RollingMean(window_size=26)]}

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbosity=-1)


def sin1(dates):
    return np.sin(2 * np.pi * dates.dayofyear / 365.25)


def cos1(dates):
    return np.cos(2 * np.pi * dates.dayofyear / 365.25)


def sin2(dates):
    return np.sin(4 * np.pi * dates.dayofyear / 365.25)


def cos2(dates):
    return np.cos(4 * np.pi * dates.dayofyear / 365.25)


DATE_FEATURES = ["month", sin1, cos1, sin2, cos2]


def future_promo(drivers, cutoff, plan_weeks, horizon=14):
    """Promotion flags for the forecast weeks after cutoff.

    The first plan_weeks weeks use the realized flag (a stand-in for a committed promotion plan,
    which this dataset does not contain). Later weeks assume no promotion.
    plan_weeks=0 uses history only; plan_weeks=horizon is the oracle.
    """
    future = drivers[(drivers.ds > cutoff) & (drivers.ds <= cutoff + pd.Timedelta(weeks=horizon))].copy()
    future.loc[future.ds > cutoff + pd.Timedelta(weeks=plan_weeks), "promo"] = 0
    return future[["unique_id", "ds", "promo"]]


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
