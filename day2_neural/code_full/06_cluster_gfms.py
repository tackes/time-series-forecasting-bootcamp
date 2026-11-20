# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# # Day 2 · 06 (full data) — Lab: one global model, or one per cluster? (20 min)
#
# A global model pools every series. That is its strength (more examples) and its risk (a fast
# seller and a slow, lumpy one get the same weights). A middle road is to **group similar series**
# and either tell the model the group, or train one model per group. The research only tried
# cluster arms with LightGBM and a cluster static in NBEATSx, so this notebook is new work.
#
# | Arm | What it is |
# |---|---|
# | pooled | the selected model (05 or 05b), trained on every series |
# | pooled + cluster static | the same model, with the cluster as one-hot statics |
# | per cluster | one model per cluster, trained only on that cluster's series |
#
# This notebook
#
# 1. describes every series with a few demand features from the **training weeks only**,
# 2. groups them with k-means, choosing the number of groups by silhouette,
# 3. fits the two cluster arms (3 seeds each) and scores all three on the test window.

# %%
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import gd_helpers as gd
import gd_plots as gplt
import gd_price as gp

bc = gd.bc
gd.status()
gplt.use_style()

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Intuition: classmates, not strangers
#
# Think of a teacher with one big class. Teaching everyone together is efficient, but the fast
# readers and the beginners need different things. You can tell the teacher who is in which
# reading group (a cluster static), or split the class and give each group its own teacher (one
# model per cluster). Splitting helps if the groups really learn differently, and hurts if each
# teacher now has too few pupils to learn from.

# %% [markdown]
# ## 1. The selected model and its data

# %%
cfg = gd.selected_config()
noise_floor = gd.load_config("02_shortlist")["noise_floor"]
panel, train, test, data = gp.prepare(cfg, synthetic_seed=cfg.get("synthetic_seed"))
info = gplt.series_table(panel)
eval_ids = gd.eval_ids()
params = cfg["params"]
pooled_name = cfg.get("forecasts", "05_selected")
print(f"selected: {pooled_name} | {panel['unique_id'].n_unique():,} series | "
      f"{data.fit_df['unique_id'].nunique():,} incl. copies | {cfg['max_steps']:,} steps")

# %% [markdown]
# ## 2. Series features from the training weeks
#
# Each series is described by numbers computed **before the test window**. The data provider's
# `series_features.parquet` also has clusters, but they use the full history, test weeks
# included, so we do not use them as an input.
#
# | Feature | Meaning |
# |---|---|
# | `log_mean_units` | how much it sells: log(1 + mean weekly units) |
# | `zero_share` | share of weeks with no sales (the same information as ADI, the average gap) |
# | `log_cv2_nonzero` | how erratic the non-zero weeks are: log(1 + CV²) |
# | `promo_share` | share of weeks on promotion |
# | `discount_share` | share of weeks with a real price cut (≥10% below the 4-week mean) |
# | `recent_trend` | mean log units in the last 26 training weeks minus the first 26 |
# | `yearly_acf` | correlation of log units with the same week a year earlier |

# %%
FEATURES = ["log_mean_units", "zero_share", "log_cv2_nonzero", "promo_share", "discount_share",
            "recent_trend", "yearly_acf"]


def train_features(train: pl.DataFrame) -> pd.DataFrame:
    """Per-series demand features from the training weeks only (no test-window leakage)."""
    t = train.filter(pl.col("ds") < bc.TEST_START).sort("unique_id", "ds")
    nonzero = pl.col("y").filter(pl.col("y") > 0)
    log_y = pl.col("y").log1p()
    feats = t.group_by("unique_id").agg(
        pl.col("y").mean().log1p().alias("log_mean_units"),
        (pl.col("y") == 0).mean().alias("zero_share"),
        (nonzero.std() / nonzero.mean()).pow(2).fill_null(0.0).log1p().alias("log_cv2_nonzero"),
        (pl.col("promo") > 0).mean().alias("promo_share"),
        pl.col("is_discount").mean().alias("discount_share"),
        (log_y.tail(26).mean() - log_y.head(26).mean()).alias("recent_trend"),
    )
    lagged = (t.with_columns(log_y.alias("ly"), log_y.shift(52).over("unique_id").alias("ly52"))
              .drop_nulls("ly52"))
    season = lagged.group_by("unique_id").agg(
        pl.corr("ly", "ly52").fill_nan(0.0).alias("yearly_acf"))
    out = feats.join(season, on="unique_id", how="left").fill_null(0.0).fill_nan(0.0)
    return out.sort("unique_id").to_pandas()


features = train_features(train)
scaled = StandardScaler().fit_transform(features[FEATURES])
features[FEATURES].describe().T.round(3)

# %% [markdown]
# ## 3. How many clusters? A silhouette sweep
#
# The silhouette compares each series' distance to its own cluster with its distance to the
# next one (1 = well separated, 0 = on the border). The rule, fixed before looking: **take the K
# in 4–8 with the highest silhouette**. It is computed on a sample of 5,000 series for speed.

# %%
K_RANGE = [3] if gd.SMOKE else [4, 5, 6, 7, 8]


def silhouette_sweep() -> pd.DataFrame:
    """Silhouette score of k-means for each K in ``K_RANGE``."""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=bc.RANDOM_SEED).fit(scaled)
        score = silhouette_score(scaled, km.labels_, sample_size=min(5_000, len(scaled)),
                                 random_state=bc.RANDOM_SEED)
        rows.append({"k": k, "silhouette": score, "inertia": km.inertia_})
    return pd.DataFrame(rows)


sweep = gd.cached("06_silhouette", silhouette_sweep)
best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(sweep["k"], sweep["silhouette"], "o-")
ax.axvline(best_k, color=gplt.COLOURS["good"], ls="--", label=f"chosen K = {best_k}")
ax.set(xlabel="number of clusters K", ylabel="silhouette (higher = better separated)",
       title="Choosing K")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# The final clustering is cached, so every later run uses the same labels. Clusters are numbered
# from the highest-selling (0) to the lowest.

# %%
def make_clusters() -> pd.DataFrame:
    """K-means labels for every training series, renumbered by mean volume (0 = highest)."""
    km = KMeans(n_clusters=best_k, n_init=10, random_state=bc.RANDOM_SEED).fit(scaled)
    out = features.assign(raw=km.labels_)
    order = out.groupby("raw")["log_mean_units"].mean().sort_values(ascending=False).index
    out["cluster"] = out["raw"].map({raw: i for i, raw in enumerate(order)})
    return out.drop(columns="raw").assign(k=best_k)


clusters = gd.cached("06_clusters", make_clusters)
cluster_of = clusters.set_index("unique_id")["cluster"]
info["cluster"] = cluster_of

# %% [markdown]
# ### Cluster profiles
#
# Feature means per cluster, with the size, how many scored series fall in it and its most
# common sales class.

# %%
profile = clusters.groupby("cluster")[FEATURES].mean()
profile.insert(0, "series", clusters.groupby("cluster").size())
profile.insert(1, "scored", info.loc[eval_ids].groupby("cluster").size())
profile.insert(2, "main_class", info.groupby("cluster")["sale_class"].agg(
    lambda s: s.value_counts().index[0]))
profile.round(2)

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## 4. The rule, written before the results
#
# The three arms are scored once on the test window. A cluster arm **beats pooled** only if its
# median RMSSE is lower by more than the 02 noise floor; otherwise pooled stays, because it is
# one model instead of several. This notebook does not change the model later notebooks use; it
# reports whether clustering is worth pursuing.

# %% [markdown]
# ## 5. Arm 1: pooled + cluster static
#
# The selected model's data, with a one-hot cluster column per cluster added to the statics.
# Oversampled copies share their original series' cluster.

# %%
def with_cluster_statics(data: gd.NixtlaData, cluster_of: pd.Series) -> gd.NixtlaData:
    """``data`` with one-hot cluster statics; copies take their original series' cluster."""
    ids = data.static_df[["unique_id"]].copy()
    original = ids["unique_id"].str.split(gp.COPY_TAG).str[0]
    ids["cluster"] = original.map(cluster_of)
    onehot = pd.get_dummies(ids["cluster"], prefix="cluster").astype("float32")
    static = data.static_df.merge(pd.concat([ids[["unique_id"]], onehot], axis=1),
                                  on="unique_id")
    return replace(data, static_df=static, stat_exog=data.stat_exog + list(onehot.columns))


static_data = with_cluster_statics(data, cluster_of)


def cluster_static_model(seed):
    return gd.build_nbeatsx(params, static_data, cfg["max_steps"], seed, "GFM_cluster_static")


static_runs = gd.fit_seeds(cluster_static_model, static_data, "06_cluster_static")
static_fcst = gd.seed_forecasts(static_runs)

# %% [markdown]
# ## 6. Arm 2: one model per cluster
#
# Each cluster gets its own model, trained only on its series (and their copies), with the step
# budget scaled to its size (`gd.scaled_max_steps`). Clusters without a scored series are not
# fitted. The forecasts are then stacked into one frame.

# %%
def cluster_subset(data: gd.NixtlaData, ids: set) -> gd.NixtlaData:
    """``data`` restricted to the series in ``ids`` and their oversampled copies."""
    keep = {u for u in data.fit_df["unique_id"].unique() if u.split(gp.COPY_TAG)[0] in ids}
    fit_df = data.fit_df[data.fit_df["unique_id"].isin(keep)]
    static = data.static_df[data.static_df["unique_id"].isin(set(fit_df["unique_id"]))]
    return replace(data, fit_df=fit_df, predict_df=gd.only(data.predict_df, ids),
                   futr_df=gd.only(data.futr_df, ids), static_df=static)


per_cluster_fcst = []
per_cluster_runs = {}
cluster_steps = {}
for k in sorted(clusters["cluster"].unique()):
    ids = set(clusters.loc[clusters["cluster"] == k, "unique_id"])
    if not ids & set(eval_ids):
        continue
    sub = cluster_subset(data, ids)
    steps = gd.scaled_max_steps(sub.fit_df["unique_id"].nunique(),
                                batch_size=params.get("batch_size", 256))
    cluster_steps[int(k)] = steps

    def per_cluster_model(seed, sub=sub, steps=steps):
        return gd.build_nbeatsx(params, sub, steps, seed, "GFM_per_cluster")

    with bc.Timer() as t:
        runs = gd.fit_seeds(per_cluster_model, sub, f"06_per_cluster_c{k}")
    per_cluster_runs[int(k)] = runs
    per_cluster_fcst.append(gd.seed_forecasts(runs))
    print(f"cluster {k}: {sub.fit_df['unique_id'].nunique():,} series incl. copies, "
          f"{steps:,} steps, {t.seconds:.0f}s")
per_cluster_fcst = pd.concat(per_cluster_fcst, ignore_index=True)

# %% [markdown]
# ## 7. Scores on the test window

# %%
pooled = gd.selected_forecasts(cfg).rename(columns={"GFM_selected": "GFM_pooled"})
arms = {"GFM_pooled": "GFM_pooled", "GFM_cluster_static": "GFM_cluster_static_mean",
        "GFM_per_cluster": "GFM_per_cluster_mean"}
both = (pooled.merge(static_fcst, on=["unique_id", "ds"])
        .merge(per_cluster_fcst, on=["unique_id", "ds"]))
metrics = gd.score(both, test, train)
rmsse_all = gd.per_series(metrics)
rmsse = rmsse_all[list(arms.values())].rename(columns={v: k for k, v in arms.items()})

seed_rows = []
for arm in ("GFM_cluster_static", "GFM_per_cluster"):
    seed_cols = [c for c in rmsse_all.columns if c.startswith(f"{arm}_s")]
    for col in seed_cols:
        seed_rows.append({"arm": arm, "seed": col.rsplit("_s", 1)[-1],
                          "median_rmsse": rmsse_all[col].median()})
seed_table = pd.DataFrame(seed_rows)
summary = pd.DataFrame({"median_rmsse": rmsse.median(), "mean_rmsse": rmsse.mean()})
seed_range = seed_table.groupby("arm")["median_rmsse"]
summary["seed_spread"] = seed_range.max() - seed_range.min()
summary.round(4)

# %%
board = both[["unique_id", "ds", "GFM_cluster_static_mean", "GFM_per_cluster_mean"]]
board = board.rename(columns={"GFM_cluster_static_mean": "GFM_cluster_static",
                              "GFM_per_cluster_mean": "GFM_per_cluster"})
per_cluster_seconds = 0.0
for runs in per_cluster_runs.values():
    per_cluster_seconds += gd.summaries(runs)["fit_seconds"].mean()
fit_seconds = {"GFM_cluster_static": gd.summaries(static_runs)["fit_seconds"].mean(),
               "GFM_per_cluster": per_cluster_seconds}
gd.save_forecasts(board, "06_clusters")
gd.record(board, test, train, lab=f"06 clusters ({len(gd.SEEDS)}-seed mean)",
          fit_seconds=fit_seconds,
          notes="per seed; per cluster = sum over clusters")
gd.median_board()

# %% [markdown]
# ### Marginal wins over pooled (F4)

# %%
gplt.plot_wins(rmsse, info, model="GFM_per_cluster", baseline="GFM_pooled")

# %%
gplt.plot_wins(rmsse, info, model="GFM_cluster_static", baseline="GFM_pooled")

# %% [markdown]
# ### Where each arm wins: by cluster and by sales class

# %%
by_cluster = rmsse.join(info["cluster"]).groupby("cluster")[list(arms)].median().T
fig, ax = plt.subplots(figsize=(10, 0.5 * len(arms) + 1.8))
im = ax.imshow(by_cluster.to_numpy(), cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(by_cluster.shape[1]), [f"cluster {c}" for c in by_cluster.columns])
ax.set_yticks(range(by_cluster.shape[0]), by_cluster.index)
for (r, c), v in np.ndenumerate(by_cluster.to_numpy()):
    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)
ax.grid(False)
fig.colorbar(im, ax=ax, label="median RMSSE")
ax.set_title("Median RMSSE by cluster (0 = highest volume)")
fig.tight_layout()
plt.show()

# %%
gplt.plot_class_heatmap(rmsse, info)

# %%
gplt.plot_error_dist(rmsse, info)

# %% [markdown]
# ## 8. Apply the rule

# %%
pooled_median = summary.loc["GFM_pooled", "median_rmsse"]
gains = pooled_median - summary["median_rmsse"].drop("GFM_pooled")
beats_pooled = gains[gains > noise_floor]
winner = beats_pooled.idxmax() if len(beats_pooled) else "GFM_pooled"
print(f"gain over pooled (median RMSSE): {gains.round(4).to_dict()} | "
      f"noise floor {noise_floor} | winner: {winner}")

# %%
gd.save_config("06_clusters", {
    "selected_from": pooled_name,
    "features": FEATURES,
    "k_range": K_RANGE,
    "k": best_k,
    "silhouette": sweep.round(4).to_dict("records"),
    "cluster_sizes": clusters["cluster"].value_counts().sort_index().to_dict(),
    "cluster_max_steps": cluster_steps,
    "seeds": gd.SEEDS,
    "median_rmsse": summary["median_rmsse"].round(4).to_dict(),
    "mean_rmsse": summary["mean_rmsse"].round(4).to_dict(),
    "seed_medians": seed_table.round(4).to_dict("records"),
    "median_rmsse_by_cluster": by_cluster.round(4).to_dict("index"),
    "noise_floor": noise_floor,
    "rule": "a cluster arm wins only if its test median RMSSE beats pooled by > noise floor",
    "winner": winner,
})

# %% [markdown]
# <!-- DRAFT: Graeme to edit -->
# ## Discussion
#
# * Per-cluster models see fewer series each. Look at the smallest clusters in the heatmap: is
#   that where splitting costs most?
# * The cluster static gives the pooled model the same information without splitting the data.
#   If it does no better than pooled, the model may already infer the group from the statics and
#   the history it sees.
# * The features describe demand, not price response. Would clusters built on discount depth and
#   lift group the series differently?
#
# **Try it:** add `log_mean_price` or the department one-hot to `FEATURES`, delete
# `06_clusters.parquet` and `06_silhouette.parquet`, and rerun with `GD_TRAIN=1`.
