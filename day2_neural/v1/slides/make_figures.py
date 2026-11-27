"""Intuition figures for the Day 2 slides.

Theory figures use small synthetic examples. Result figures use the numbers from
the reference run of the labs (2,000 series, CPU, 500 steps, 27 Oct 2026); rerun the
labs and edit the RESULTS block if your numbers differ.

    python slides/make_figures.py   # writes slides/img/*.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "img"
OUT.mkdir(exist_ok=True)

INK, ORANGE, BLUE, GREEN, GREY, SKY = "#0B2A3A", "#D55E00", "#0072B2", "#009E73", "#9AA7AE", "#56B4E9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold",
    "axes.titlesize": 16, "axes.titlecolor": INK, "savefig.dpi": 160,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})
rng = np.random.default_rng(46335)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


# ------------------------------------------------------------------ results
# Reference-run numbers, collected from the executed notebooks by collect_results.py into
# reference_results.json (rerun that script after rerunning the labs, then this one).
import json

RES = json.loads((Path(__file__).parent / "reference_results.json").read_text())
_lb = {r[0]: r for r in RES["lb"]}
_m = lambda k: _lb[k][1]
_md = lambda k: _lb[k][2]
_NM = {"DLinear": "DLinear", "NBEATS_interpretable": "N-BEATS interp.", "NBEATS_generic": "N-BEATS generic",
       "NHITS": "NHITS", "LightGBM": "LightGBM (Day 1)"}
_l6 = RES["l6"]
_types = RES["l7_types"]; _gi = _types["models"].index("LightGBM")
_ev = RES["l7_eval"]; _best = RES["l7_heldout"]["best_single"]
_cv = RES["cv"]
RESULTS = dict(
    day1={k: _m(k) for k in ["SeasonalNaive", "Naive", "AutoETS", "LightGBM"]},
    day1_median={k: _md(k) for k in ["SeasonalNaive", "Naive", "AutoETS", "LightGBM"]},
    lab01={_NM[k]: r for k, t, r, f in RES["l1"]},
    lab01_fit={_NM[k]: t for k, t, r, f in RES["l1"]},
    lab01_cv={f"fold 1\n(to {_cv[0][0][:7]})": _cv[0][1], f"fold 2\n(to {_cv[1][0][:7]})": _cv[1][1]},
    lab02_spread=dict(RES["l2_spread"]),
    lab02_seed=RES["l2_seed"],
    lab02_loss=RES["l2_loss"],
    lab02_cost={"NHITS default": tuple(RES["l2_cost"][0][1:]), "NHITS tuned (8 Optuna trials)": tuple(RES["l2_cost"][1][1:])},
    lab02_lgb=(next(t for k, t, r, f in RES["l1"] if k == "LightGBM"), _m("LightGBM")),
    lab03_nhits={k: (RES["l3_cov"]["NHITS_noexog"][i], RES["l3_cov"]["NHITSx"][i])
                 for i, k in enumerate(["mean\nRMSSE", "median\nRMSSE", "WAPE"])},
    lab03_lift={("Data, raw\n(upper ref.)" if k == "raw" else k): v for k, v in RES["l3_lift"]},
    lab03_mono={k: RES["l3_resp"][k]["monotone"] for k in ["NHITSx", "LightGBMx", "LightGBMx_mono"]},
    lab04=RES.get("l4"),
    lab04_refs={k: _m(k) for k in ["NHITSx", "LightGBM"]},
    lab05_levels=RES["l5"]["levels"],
    lab05={"Base (incoherent)": RES["l5"]["base"], **RES["l5"]["m"]},
    lab06_cov={f"NHITS_MQ (quantile loss)\ninterval score {_l6['NHITS_MQ'][2]:.0f}": tuple(_l6["NHITS_MQ"][:2]),
               f"LightGBM, per-series conformal\ninterval score {_l6['LightGBM_conf'][2]:.0f}": tuple(_l6["LightGBM_conf"][:2]),
               f"LightGBM, pooled conformal\ninterval score {_l6['LightGBM_pooled'][2]:.0f}": tuple(_l6["LightGBM_pooled"][:2]),
               f"AutoETS, per-series conformal\ninterval score {_l6['ETS_conf'][2]:.0f}": tuple(_l6["ETS_conf"][:2])},
    lab07_types={t: (n, v[_gi]) for t, v, n in _types["rows"]},
    lab07_holdout={"ENS mean top-3": _ev["ENS_mean_top3"], f"{_best[0]} (best single)": _best[1],
                   "ENS mean of all": _ev["ENS_mean_all"], "Best model per class": _ev["SEL_by_class"]},
)


# ------------------------------------------------------------------ theory
def rmsse_ruler():
    """RMSSE as 'my error divided by the error of copying last year'."""
    t = np.arange(104)
    season = 40 + 25 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 6, t.size)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(t, season, color=INK, lw=2)
    ax.plot(t[52:], season[:52], color=ORANGE, lw=2, ls="--", label="copy of last year (seasonal naive)")
    ax.fill_between(t[52:], season[52:], season[:52], color=ORANGE, alpha=0.15)
    ax.set_title("The yardstick: how wrong is 'copy last year'?")
    ax.set_xlabel("week"); ax.set_yticks([]); ax.legend(frameon=False, loc="upper left", fontsize=13)
    ax = axes[1]
    vals = {"0.5": 0.5, "1.0": 1.0, "1.5": 1.5}
    cols = [GREEN, INK, ORANGE]
    ax.barh(list(vals), list(vals.values()), color=cols)
    for k, (lab, v) in enumerate(vals.items()):
        txt = ["half the yardstick error", "as wrong as the yardstick", "50 % worse than the yardstick"][k]
        ax.text(v + 0.03, k, txt, va="center", fontsize=13, color=INK)
    ax.set_xlim(0, 2.9); ax.set_title("RMSSE = my error ÷ yardstick error"); ax.set_xticks([])
    ax.set_ylabel("RMSSE")
    save(fig, "rmsse_ruler")


def global_vs_local():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax = axes[0]
    ax.set_title("Local: one model per series")
    for r in range(3):
        for c in range(3):
            x0, y0 = 0.3 + c * 3.2, 0.4 + r * 1.9
            s = np.cumsum(rng.normal(0, 1, 20))
            s = (s - s.min()) / (np.ptp(s) + 1e-9)
            ax.plot(x0 + np.linspace(0, 1.4, 20), y0 + 0.2 + 0.9 * s, color=INK, lw=1.5)
            ax.add_patch(plt.Rectangle((x0 + 1.6, y0 + 0.25), 1.1, 0.9, color=GREY, alpha=0.5))
            ax.text(x0 + 2.15, y0 + 0.7, "model", ha="center", va="center", fontsize=10)
    ax.text(5, -0.35, "each model sees ~90 weeks", ha="center", fontsize=13, color=ORANGE)
    ax = axes[1]
    ax.set_title("Global: one model, windows from every series")
    for r in range(6):
        y0 = 0.3 + r * 0.95
        s = np.cumsum(rng.normal(0, 1, 30))
        s = (s - s.min()) / (np.ptp(s) + 1e-9)
        ax.plot(np.linspace(0.2, 3.6, 30), y0 + 0.65 * s, color=INK, lw=1.5)
        ax.add_patch(plt.Rectangle((1.2 + 0.2 * r % 1.5, y0 - 0.05), 1.2, 0.75, fill=False, ec=ORANGE, lw=1.5))
        ax.annotate("", xy=(6.3, 3.0), xytext=(3.8, y0 + 0.3), arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    ax.add_patch(plt.Rectangle((6.3, 2.1), 3.2, 1.8, color=BLUE))
    ax.text(7.9, 3.0, "one shared\nmodel", ha="center", va="center", color="white", fontsize=15, weight="bold")
    ax.text(5, -0.35, "the model sees ~180,000 weeks", ha="center", fontsize=13, color=ORANGE)
    save(fig, "global_vs_local")


def nbeats_relay():
    """Doubly residual stacking as a relay: each block explains what it can and passes on the rest."""
    t = np.arange(80)
    trend = 0.04 * t + 2
    season = 1.2 * np.sin(2 * np.pi * t / 26)
    noise = rng.normal(0, 0.35, t.size)
    y = trend + season + noise
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True, gridspec_kw={"wspace": 0.25})
    panels = [("① input window", y, INK), ("② block 1: trend", trend, ORANGE),
              ("③ block 2: season", season + trend.mean(), BLUE), ("④ left over: noise", noise + trend.mean(), GREY)]
    for ax, (title, s, c) in zip(axes, panels):
        ax.plot(t, s, color=c, lw=2.2)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("N-BEATS: each block explains part of the input, the next block sees only the residual; forecasts add up",
                 y=-0.02, fontsize=14, color=INK)
    save(fig, "nbeats_relay")


def window_scaling():
    t = np.arange(40)
    shape = 1 + 0.4 * np.sin(2 * np.pi * t / 13) + (t % 10 == 7) * 0.8
    big, small = 150_000 * shape * rng.normal(1, 0.03, t.size), 600 * shape * rng.normal(1, 0.08, t.size)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    axes[0].plot(t, big, color=BLUE, lw=2, label="class A item, ~150k/week")
    axes[0].plot(t, small, color=ORANGE, lw=2, label="class J item, ~600/week")
    axes[0].set_title("Raw: the small item is a flat line"); axes[0].legend(frameon=False, fontsize=12)
    z = lambda s: (s - s.mean()) / s.std()
    axes[1].plot(t, z(big), color=BLUE, lw=2); axes[1].plot(t, z(small), color=ORANGE, lw=2)
    axes[1].set_title("Scaled per window: the same shape")
    for ax in axes:
        ax.set_xticks([])
    save(fig, "window_scaling")


def loss_shapes():
    e = np.linspace(-3, 3, 200)
    huber = np.where(np.abs(e) < 1, 0.5 * e**2, np.abs(e) - 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    ax.plot(e, e**2, color=ORANGE, lw=2.5, label="MSE: big misses hurt a lot")
    ax.plot(e, np.abs(e), color=BLUE, lw=2.5, label="MAE: every unit costs the same")
    ax.plot(e, huber, color=GREEN, lw=2.5, ls="--", label="Huber: MAE in the tails")
    ax.set_ylim(0, 5); ax.set_xlabel("forecast error"); ax.set_title("How each loss punishes an error")
    ax.legend(frameon=False, fontsize=12)
    ax = axes[1]
    weeks = np.array([0, 0, 3, 0, 0, 0, 14, 0, 2, 0, 0, 0])
    ax.bar(np.arange(weeks.size), weeks, color=GREY)
    ax.axhline(weeks.mean(), color=ORANGE, lw=2.5, label=f"MSE answer = mean = {weeks.mean():.1f}")
    ax.axhline(np.median(weeks), color=BLUE, lw=2.5, label="MAE answer = median = 0")
    ax.set_title("An intermittent item: what should we forecast?"); ax.set_xlabel("week")
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    save(fig, "loss_shapes")


def response_sketch():
    p = np.linspace(-25, 25, 50)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(p, 100 * (1 - 0.015 * p), color=GREEN, lw=3, label="plausible: cheaper → more sales")
    ax.plot(p, 100 + 0 * p, color=GREY, lw=3, ls="--", label="flat: ignores price")
    ax.plot(p, 100 * (1 + 0.01 * p), color=ORANGE, lw=3, ls=":", label="perverse: cheaper → fewer sales")
    ax.set_xlabel("price change (%)"); ax.set_ylabel("forecast units (index)")
    ax.set_title("Ask the model 'what if?' and check the answer")
    ax.legend(frameon=False, fontsize=12)
    save(fig, "response_sketch")


def hierarchy_tree():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    def node(x, y, label, val, c):
        ax.add_patch(plt.Rectangle((x - 1.1, y - 0.42), 2.2, 0.84, color=c, alpha=0.9))
        ax.text(x, y, f"{label}\n{val}", ha="center", va="center", color="white", fontsize=13, weight="bold")
    node(5, 4.1, "Department", "forecast 120", BLUE)
    for x, v in zip([1.8, 5, 8.2], [30, 25, 50]):
        ax.plot([5, x], [3.68, 2.5], color=GREY, lw=1.5)
        node(x, 2.08, "Item", f"forecast {v}", INK)
    ax.text(5, 0.75, "30 + 25 + 50 = 105 ≠ 120", ha="center", fontsize=18, color=ORANGE, weight="bold")
    ax.text(5, 0.15, "Reconciliation decides whom to believe and adjusts the numbers so they add up",
            ha="center", fontsize=12, color=INK)
    save(fig, "hierarchy_tree")


def pinball():
    e = np.linspace(-3, 3, 200)  # y - q
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    for tau, c in [(0.5, BLUE), (0.9, ORANGE)]:
        ax.plot(e, np.maximum(tau * e, (tau - 1) * e), color=c, lw=2.5, label=f"τ = {tau}")
    ax.set_xlabel("actual − forecast"); ax.set_title("Pinball loss: missing low on q90 costs 9×")
    ax.legend(frameon=False)
    ax = axes[1]
    x = np.arange(20)
    truth = 10 + rng.normal(0, 2, x.size)
    for k, (lo, hi, c, lab) in enumerate([(-6, 6, GREY, "calibrated but wide"), (-1, 1, ORANGE, "sharp but overconfident"),
                                         (-2.6, 2.6, GREEN, "calibrated and sharp")]):
        off = k * 16
        ax.fill_between(x, 10 + lo + off, 10 + hi + off, color=c, alpha=0.3)
        ax.plot(x, truth + off, "o", color=INK, ms=3.5)
        ax.text(20.5, 10 + off, lab, va="center", fontsize=12, color=INK)
    ax.set_xlim(0, 33); ax.axis("off"); ax.set_title("Calibrated first, then sharp")
    save(fig, "pinball")


def conformal():
    res = np.abs(rng.standard_t(4, 400)) * 10
    q = np.quantile(res, 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    ax.hist(res, bins=40, color=GREY)
    ax.axvline(q, color=ORANGE, lw=3)
    ax.text(q + 1, ax.get_ylim()[1] * 0.85, f"80 % of past misses\nare smaller than {q:.0f}", color=ORANGE, fontsize=13)
    ax.set_title("1. Collect the model's past misses"); ax.set_xlabel("|actual − forecast| on calibration windows")
    ax = axes[1]
    t = np.arange(14)
    f = 100 + 5 * np.sin(t / 2)
    ax.plot(t, f, color=BLUE, lw=2.5, label="point forecast")
    ax.fill_between(t, f - q, f + q, color=BLUE, alpha=0.2, label=f"forecast ± {q:.0f}")
    ax.set_title("2. Put that miss around every new forecast"); ax.set_xlabel("weeks ahead")
    ax.legend(frameon=False, fontsize=12)
    save(fig, "conformal")


def elo_curve():
    d = np.linspace(-400, 400, 200)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(d, 1 / (1 + 10 ** (-d / 400)), color=BLUE, lw=3)
    for x in [18, 100, 200]:
        p = 1 / (1 + 10 ** (-x / 400))
        ax.plot([x], [p], "o", color=ORANGE)
        ax.text(x + 12, p - 0.04, f"+{x} → wins {p:.0%}", fontsize=12, color=INK)
    ax.axhline(0.5, color=GREY, lw=1, ls="--")
    ax.set_xlabel("rating difference"); ax.set_ylabel("chance to win a series")
    ax.set_title("Ratings are win chances, like chess")
    save(fig, "elo_curve")


def rolling_origin():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    weeks = 93
    for k, (start_test, lab) in enumerate([(65, "CV window 1 (from Sep 2024)"), (79, "CV window 2 = validation (from Jan 2025)"),
                                           (93, "test, scored once (from Apr 2025)")]):
        y = 2 - k
        ax.barh(y, start_test, color=SKY if k < 2 else GREY, height=0.55)
        ax.barh(y, 14, left=start_test, color=ORANGE if k < 2 else INK, height=0.55)
        ax.text(start_test + 15, y, lab, va="center", fontsize=13, color=INK)
    ax.axvline(weeks, color=INK, ls=":", lw=1)
    ax.set_yticks([]); ax.set_xlim(0, 160)
    ax.set_xlabel("weeks since 2023-07-05   (blue = train on, orange = forecast and score)")
    ax.set_title("Rolling origin: train on the past, forecast the next 14 weeks, slide forward")
    save(fig, "rolling_origin")


def fm_patches():
    t = np.arange(64)
    y = 50 + 15 * np.sin(2 * np.pi * t / 26) + rng.normal(0, 4, t.size) + (t % 16 == 9) * 25
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.3, 1.3, 1], "wspace": 0.3})
    axes[0].plot(t, y, color=INK, lw=2); axes[0].set_title("① history (any scale)")
    z = (y - y.mean()) / y.std()
    for k in range(8):
        sl = slice(8 * k, 8 * k + 8)
        axes[1].plot(t[sl], z[sl], color=[BLUE, ORANGE][k % 2], lw=2.5)
        axes[1].axvspan(8 * k, 8 * k + 8, color=[BLUE, ORANGE][k % 2], alpha=0.08)
    axes[1].set_title("② scaled, cut into patches = tokens")
    ax = axes[2]
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.add_patch(plt.Rectangle((-0.5, 1.6), 5.2, 2.8, color=BLUE, clip_on=False))
    ax.text(2.1, 3.0, "pretrained\ntransformer", ha="center", va="center", color="white", fontsize=12, weight="bold")
    ax.annotate("", xy=(6.0, 3.0), xytext=(4.8, 3.0), arrowprops=dict(arrowstyle="->", color=INK, lw=2))
    for k, (lo, hi) in enumerate([(2.2, 3.8), (1.6, 4.4), (1.0, 5.0)]):
        ax.add_patch(plt.Rectangle((6.3 + 1.1 * k, lo), 0.8, hi - lo, color=ORANGE, alpha=0.35 + 0.2 * (2 - k)))
    ax.text(7.8, 0.4, "③ next patches\nas quantiles", ha="center", fontsize=12, color=INK)
    for a in axes[:2]:
        a.set_xticks([]); a.set_yticks([])
    save(fig, "fm_patches")


def crps():
    x = np.linspace(0, 20, 400)
    from math import erf
    cdf = lambda m, s: 0.5 * (1 + np.vectorize(erf)((x - m) / (s * np.sqrt(2))))
    actual = 11.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
    for ax, (m, sd, title) in zip(axes, [(10.5, 1.2, "centred and sharp: small area"), (7, 3.5, "off-centre and vague: big area")]):
        F = cdf(m, sd)
        step = (x >= actual).astype(float)
        ax.plot(x, F, color=BLUE, lw=2.5, label="forecast CDF")
        ax.step(x, step, color=INK, lw=2, where="post", label=f"truth: step at {actual:.0f}")
        ax.fill_between(x, F, step, color=ORANGE, alpha=0.35, label="CRPS = this area")
        ax.set_title(title); ax.set_xlabel("weekly units")
    axes[0].set_ylabel("P(sales ≤ x)"); axes[0].legend(frameon=False, fontsize=11, loc="upper left")
    save(fig, "crps")


def demand_types():
    """Syntetos–Boylan quadrants for every eligible series (training weeks)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import bootcamp as bc
    info = bc.series_info(bc.load_panel()).to_pandas()
    colors = {"smooth": GREEN, "erratic": BLUE, "intermittent": ORANGE, "lumpy": INK}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for t, g in info.groupby("demand_type"):
        ax.scatter(g["adi"].clip(upper=12), g["cv2_nonzero"].clip(upper=4), s=4, alpha=0.3, color=colors[t], rasterized=True,
                   label=f"{t} ({len(g):,} items)")
    ax.axvline(1.32, color=GREY, ls="--"); ax.axhline(0.49, color=GREY, ls="--")
    ax.set_xscale("log"); ax.set_xticks([1, 1.32, 2, 4, 8], ["1", "1.32", "2", "4", "8+"])
    ax.set_xlabel("ADI: average weeks between sales (log scale)"); ax.set_ylabel("CV² of non-zero weeks")
    ax.set_title(f"How often it sells × how steady it is ({len(info):,} items, training weeks)")
    ax.legend(frameon=False, fontsize=11, loc="upper right", markerscale=2.5)
    save(fig, "demand_types")


# ------------------------------------------------------------------ results
def barh(ax, d, colors=None, fmt="{:.3f}", ref=None):
    names, vals = list(d), list(d.values())
    ax.barh(names, vals, color=colors or INK)
    for k, v in enumerate(vals):
        ax.text(v, k, " " + fmt.format(v), va="center", fontsize=12, color=INK)
    if ref is not None:
        ax.axvline(ref, color=GREY, ls="--", lw=1)
    ax.invert_yaxis()


def res_day1():
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    c = [GREY, GREY, GREY, BLUE]
    barh(axes[0], RESULTS["day1"], c, ref=1.0); axes[0].set_title("Mean RMSSE: pulled up by a few bad series")
    barh(axes[1], RESULTS["day1_median"], c, ref=1.0); axes[1].set_title("Median RMSSE: the typical series")
    for ax in axes:
        ax.set_xlim(0, 1.9)
    save(fig, "res_day1")


def res_lab01():
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    names = list(RESULTS["lab01"])
    cols = [GREY if n == "DLinear" else (BLUE if "LightGBM" in n else INK) for n in names]
    barh(axes[0], RESULTS["lab01"], cols); axes[0].set_xlim(1.0, max(RESULTS["lab01"].values()) * 1.1)
    axes[0].set_title("Test RMSSE: the deep models land on top of each other")
    cv = RESULTS["lab01_cv"]
    axes[1].bar(list(cv), list(cv.values()), color=[ORANGE, ORANGE])
    for k, v in enumerate(cv.values()):
        axes[1].text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    axes[1].set_title("Same NHITS, two CV folds"); axes[1].set_ylabel("RMSSE")
    save(fig, "res_lab01")


def res_lab02():
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    sp = dict(sorted(RESULTS["lab02_spread"].items(), key=lambda kv: -kv[1]))
    barh(axes[0], sp, [ORANGE] + [INK] * (len(sp) - 1))
    axes[0].set_title("Validation RMSSE swing per knob"); axes[0].set_xlim(0, max(sp.values()) * 1.3)
    axes[0].axvline(RESULTS["lab02_seed"], color=ORANGE, ls="--", lw=1.5, label="seed noise (same settings, 3 seeds)")
    axes[0].legend(frameon=False, fontsize=11, loc="lower right")
    for k, (name, (sec, r)) in enumerate(RESULTS["lab02_cost"].items()):
        axes[1].scatter(sec, r, s=180, color=[INK, ORANGE][k], zorder=3)
        axes[1].text(sec, r + 0.003, name, ha="center" if k == 0 else "right", va="bottom", fontsize=12)
    (ls, lr), cost = RESULTS["lab02_lgb"], list(RESULTS["lab02_cost"].values())
    axes[1].scatter(ls, lr, s=180, color=BLUE, zorder=3)
    axes[1].text(ls, lr - 0.004, "LightGBM", ha="left", va="top", fontsize=12)
    ys = [lr] + [r for _, r in cost]
    axes[1].set_xscale("log"); axes[1].set_xlim(ls / 3, cost[1][0] * 4)
    axes[1].set_ylim(min(ys) - 0.02, max(ys) + 0.02)
    axes[1].set_xlabel("fit time (s, log scale)"); axes[1].set_ylabel("test RMSSE")
    axes[1].set_title(f"{cost[1][0] / cost[0][0]:.0f}× the compute for {100 * (1 - cost[1][1] / cost[0][1]):.1f} % better")
    save(fig, "res_lab02")


def res_lab03():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw={"wspace": 0.85})
    ax = axes[0]
    w = 0.38
    keys = list(RESULTS["lab03_nhits"])
    no = [RESULTS["lab03_nhits"][k][0] for k in keys]; yes = [RESULTS["lab03_nhits"][k][1] for k in keys]
    ax.bar(np.arange(3) - w / 2, no, w, color=GREY, label="NHITS, no covariates")
    ax.bar(np.arange(3) + w / 2, yes, w, color=BLUE, label="NHITS + covariates")
    for k in range(3):
        ax.text(k + w / 2, yes[k] + 0.02, "worse" if yes[k] > no[k] else "better", ha="center", fontsize=11,
                color=ORANGE if yes[k] > no[k] else GREEN, weight="bold")
    ax.set_xticks(range(3), keys, fontsize=13); ax.set_ylim(0, max(no + yes) * 1.35); ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("Same forecasts, different verdicts"); ax.legend(frameon=False, fontsize=11, loc="upper right", ncol=1)
    barh(axes[1], RESULTS["lab03_lift"], [GREEN, INK, INK, INK], fmt="{:.2f}")
    axes[1].set_title("Median promo lift"); axes[1].set_xlim(0, max(RESULTS["lab03_lift"].values()) * 1.15)
    barh(axes[2], RESULTS["lab03_mono"], [INK, INK, BLUE], fmt="{:.1%}")
    axes[2].set_title("Price curves sloping down"); axes[2].set_xlim(0, 1.2)
    save(fig, "res_lab03")


def res_lab04():
    l4 = RESULTS["lab04"]
    if not l4:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    names = ["Chronos2_zs", "TimesFM_zs", "Chronos2_zs_cov", "Chronos2_ft", "NHITS"]
    labels = ["Chronos-2\nzero-shot", "TimesFM 2.5\nzero-shot", "Chronos-2\n+ covariates", "Chronos-2\nfine-tuned", "NHITS\ndefault"]
    w, x = 0.38, np.arange(len(names))
    val = [l4["valid"][n][0] for n in names]; tst = [l4["test"][n][0] for n in names]
    ax = axes[0]
    ax.bar(x - w / 2, val, w, color=GREY, label="validation window")
    ax.bar(x + w / 2, tst, w, color=[BLUE if "cov" in n or "ft" in n else INK for n in names], label="test window")
    for k in range(len(names)):
        ax.text(x[k] + w / 2, tst[k] + 0.01, f"{tst[k]:.2f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(RESULTS["lab04_refs"]["NHITSx"], color=ORANGE, ls="--", lw=1.5, label=f"NHITSx, test ({RESULTS['lab04_refs']['NHITSx']:.2f})")
    ax.set_xticks(x, labels, fontsize=11); ax.set_ylim(1.0, max(val) * 1.06); ax.set_ylabel("mean RMSSE")
    ax.set_title("Covariates are the lever; fine-tuning is not"); ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=1)
    ax = axes[1]
    for n, lab in zip(names, labels):
        c, r = l4["valid"][n][3], l4["test"][n][0]
        ax.scatter(c, r, s=160, color=BLUE if "cov" in n or "ft" in n else GREY if n == "NHITS" else INK, zorder=3)
        ax.text(c * 1.15, r, lab.replace("\n", " "), va="center", fontsize=11)
    for k, col in [("NHITSx", ORANGE), ("LightGBM", GREEN)]:
        ax.axhline(RESULTS["lab04_refs"][k], color=col, ls="--", lw=1.2)
        ax.text(1.1, RESULTS["lab04_refs"][k] + 0.004, k, color=col, fontsize=11, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(1, 1500); ax.set_xlabel("CPU seconds per 1,000 series, validation run (log scale)"); ax.set_ylabel("test RMSSE")
    ax.set_title("What each forecast costs on a laptop CPU")
    save(fig, "res_lab04")


def res_lab05():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    lv = RESULTS["lab05_levels"]
    base = np.array(RESULTS["lab05"]["Base (incoherent)"])
    styles = {"BottomUp": (BLUE, "-"), "TopDown": (ORANGE, "-"), "MinT-wls_var": (GREEN, "-")}
    for name, (c, ls) in styles.items():
        ax.plot(lv, np.array(RESULTS["lab05"][name]) / base, marker="o", color=c, ls=ls, lw=2.5, label=name)
    ax.axhline(1, color=INK, lw=1, ls="--"); ax.text(4.05, 1.0, "base", va="center", fontsize=12)
    rel = [v / b for n in styles for v, b in zip(RESULTS["lab05"][n], base)]
    ax.set_ylim(min(rel) - 0.05, max(rel) + 0.1)
    ax.set_ylabel("RMSSE ÷ base RMSSE (lower is better)")
    ax.set_title("Each reconciler trusts a different level")
    ax.legend(frameon=False, ncol=3, fontsize=12, loc="upper right")
    save(fig, "res_lab05")


def res_lab06():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    names = list(RESULTS["lab06_cov"])
    y = np.arange(len(names))
    for k, (n, (c80, c95)) in enumerate(RESULTS["lab06_cov"].items()):
        ax.plot([c80], [k - 0.12], "o", ms=12, color=BLUE, label="80 % interval" if k == 0 else None)
        ax.plot([c95], [k + 0.12], "s", ms=12, color=ORANGE, label="95 % interval" if k == 0 else None)
    ax.axvline(0.8, color=BLUE, ls="--", lw=1.2); ax.axvline(0.95, color=ORANGE, ls="--", lw=1.2)
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("share of test weeks inside the interval (dashed = promised)")
    ax.set_title("3 scores per series cannot promise 95 %; pooling across series can")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    save(fig, "res_lab06")


def res_lab07():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"wspace": 0.8})
    t = RESULTS["lab07_types"]
    n_tot = sum(n for n, _ in t.values()); e_tot = sum(n * r for n, r in t.values())
    ax = axes[0]
    cols = {"smooth": BLUE, "erratic": GREEN, "lumpy": GREY, "intermittent": ORANGE}
    bottom = [0.0, 0.0]
    for k, (n, r) in t.items():
        vals = [n / n_tot, n * r / e_tot]
        ax.bar([0, 1], vals, bottom=bottom, color=cols[k], label=k)
        for x in range(2):
            if vals[x] > 0.08:
                ax.text(x, bottom[x] + vals[x] / 2, f"{vals[x]:.0%}", ha="center", va="center", color="white", weight="bold")
        bottom = [bottom[0] + vals[0], bottom[1] + vals[1]]
    ax.set_xticks([0, 1], ["series", "summed RMSSE\n(LightGBM)"])
    ax.set_yticks([]); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=11)
    ax.set_title(f"1 in {round(n_tot / t['intermittent'][0])} series makes {t['intermittent'][0] * t['intermittent'][1] / e_tot:.0%} of the mean")
    barh(axes[1], RESULTS["lab07_holdout"], [GREEN, INK, GREY, GREY])
    axes[1].set_xlim(min(RESULTS["lab07_holdout"].values()) - 0.1, max(RESULTS["lab07_holdout"].values()) + 0.06); axes[1].set_title("Held-out half: averaging beats picking")
    save(fig, "res_lab07")


if __name__ == "__main__":
    for f in [rmsse_ruler, global_vs_local, nbeats_relay, window_scaling, loss_shapes, response_sketch,
              hierarchy_tree, pinball, conformal, elo_curve, res_day1, res_lab01, res_lab02, res_lab03,
              res_lab04, res_lab05, res_lab06, res_lab07, rolling_origin, fm_patches, crps, demand_types]:
        f()
    print("wrote", len(list(OUT.glob("*.png"))), "figures to", OUT)
