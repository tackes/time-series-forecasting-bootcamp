"""Leakage diagnostics: forecasts that read the realized selling price (notebook 03, optional section 12).

`sale_price` is a realized average selling price. A week with no sales has no transactions and gets the
regular price, so outside promotions the realized price reveals the zero-sales weeks: it is derived from the
target, not a plan. These two models measure what that does to a forecast, and give the Day 2 neural models
that use realized-price covariates a Day 1 counterpart with the same information:

- lgbm_realized_price_leaky:  the promotion-calendar features, with discount and price index computed from
                              the realized selling price in every week (`realized_drivers`).
- lgbm_day2_covariates_leaky: the promotion-calendar features with Day 2's `discount_depth` and `price_index`
                              (`day2_covariates`).

Same setup as the notebook 03 models: the 02 LightGBM (L2) on the per-series standardized target, the 02 lags,
rolling means and calendar terms, and department, class and subclass as categorical features. Forecasts carry
`information_tier="leakage_diagnostic"`, so they are never eligible for model selection or ensembles.
"""

import lightgbm as lgb
import pandas as pd
from mlforecast import MLForecast
from mlforecast.target_transforms import LocalStandardScaler

from shared.features import (CATEGORICAL, DATE_FEATURES, LAG_TRANSFORMS, LAGS, LGB_PARAMS, PROMO_CALENDAR, add_drivers,
                             day2_covariates, future_drivers, in_horizon, item_attributes, realized_drivers)
from shared.leaderboard import to_long
from shared.prepare_data import FREQ, H

LEAKAGE_MODELS = ["lgbm_realized_price_leaky", "lgbm_day2_covariates_leaky"]
REALIZED_COLUMNS = PROMO_CALENDAR + ["realized_discount", "realized_index"]
DAY2_COLUMNS = PROMO_CALENDAR + ["day2_discount_depth", "day2_price_index"]


def _fit_predict(name, actuals, attributes, drivers, columns, cutoff):
    train = actuals[actuals.ds <= cutoff].merge(drivers[["unique_id", "ds", *columns]], on=["unique_id", "ds"])
    train = train.merge(attributes, on="unique_id").astype({col: "category" for col in CATEGORICAL})
    model = MLForecast(models={name: lgb.LGBMRegressor(**LGB_PARAMS)}, freq=FREQ, lags=LAGS, lag_transforms=LAG_TRANSFORMS,
                       date_features=DATE_FEATURES, target_transforms=[LocalStandardScaler()], num_threads=1)
    model.fit(train, static_features=CATEGORICAL)
    X_df = in_horizon(drivers, cutoff)[["unique_id", "ds", *columns]]
    return to_long(model.predict(H, X_df=X_df), cutoff, "03_exogenous", "leakage_diagnostic")


def leakage_diagnostics(panel, cutoffs):
    """Forecasts of both leakage diagnostics at each cutoff, from the raw panel columns."""
    panel = add_drivers(panel)
    actuals = panel[["unique_id", "ds", "y"]]
    attributes = item_attributes(panel)
    calendar = future_drivers(panel)[["unique_id", "ds", *PROMO_CALENDAR[1:]]]
    realized = realized_drivers(panel)

    out = []
    for cutoff in cutoffs:
        out.append(_fit_predict("lgbm_realized_price_leaky", actuals, attributes, realized, REALIZED_COLUMNS, cutoff))
        day2 = day2_covariates(panel, cutoff).merge(calendar, on=["unique_id", "ds"])
        out.append(_fit_predict("lgbm_day2_covariates_leaky", actuals, attributes, day2, DAY2_COLUMNS, cutoff))
    return pd.concat(out)
