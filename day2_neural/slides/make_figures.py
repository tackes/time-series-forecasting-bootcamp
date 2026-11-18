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
# Reference run numbers, copied from the executed notebooks.
RESULTS = dict(
    day1={"SeasonalNaive": 1.525, "Naive": 1.332, "AutoETS": 1.285, "LightGBM": 1.169},
    day1_median={"SeasonalNaive": 0.929, "Naive": 0.771, "AutoETS": 0.708, "LightGBM": 0.601},
    lab01={"DLinear": 1.414, "N-BEATS interp.": 1.229, "N-BEATS generic": 1.225, "NHITS": 1.193, "LightGBM (Day 1)": 1.169},
    lab01_fit={"DLinear": 3.0, "N-BEATS interp.": 24.2, "N-BEATS generic": 32.2, "NHITS": 36.3, "LightGBM (Day 1)": 1.8},
    lab01_cv={"fold 1\n(to Sep 2024)": 2.346, "fold 2\n(to Jan 2025)": 1.269},
    lab02_spread={"loss": 0.248, "input_size": 0.058, "mlp_units": 0.037, "learning_rate": 0.026},
    lab02_seed=0.0085,
    lab02_loss={"MAE": 1.261, "Huber": 1.260, "MSE": 1.508},
    lab02_cost={"NHITS default": (36, 1.193), "NHITS tuned (8 Optuna trials)": (539, 1.174)},
    lab03_nhits={"mean\nRMSSE": (1.131, 1.153), "median\nRMSSE": (0.623, 0.563), "WAPE": (0.334, 0.269)},
    lab03_lift={"Data, raw\n(upper ref.)": 0.82, "LightGBMx": 0.55, "LightGBMx_mono": 0.48, "NHITSx": 0.39},
    lab03_mono={"NHITSx": 0.434, "LightGBMx": 0.734, "LightGBMx_mono": 0.978},
    lab05_levels=["total", "department", "class", "subclass", "item"],
    lab05={
        "Base (incoherent)": [0.885, 1.086, 1.197, 1.429, 1.898],
        "BottomUp": [1.143, 1.073, 1.176, 1.411, 1.898],
        "TopDown": [0.885, 1.071, 1.206, 1.460, 2.089],
        "MinT-ols": [0.887, 2.933, 4.478, 5.857, 10.892],
        "MinT-shrink": [1.103, 1.054, 1.169, 1.398, 1.897],
    },
    lab06_cov={"NHITS_MQ (quantile loss)\ninterval score 15": (0.830, 0.966),
               "LightGBM, per-series conformal\ninterval score 26": (0.674, 0.744),
               "LightGBM, pooled conformal\ninterval score 27": (0.799, 0.932),
               "AutoETS, per-series conformal\ninterval score 26": (0.683, 0.745)},
    lab07_types={"smooth": (1185, 0.674), "erratic": (287, 0.783), "lumpy": (187, 1.512), "intermittent": (337, 2.823)},
    lab07_mean={"NHITS_noexog": 1.101, "NHITSx": 1.119, "NHITS_MQ": 1.120, "LightGBM": 1.131, "NHITS_tuned": 1.142, "NHITS": 1.159},
    lab07_bt={"NHITSx": 1131, "LightGBM": 1090, "NHITS_noexog": 1076, "LightGBMx": 1065, "LightGBMx_mono": 1061, "NHITS_MQ": 1056},
    lab07_holdout={"ENS mean top-3": 1.246, "NHITS_noexog (best single)": 1.310, "ENS mean of all": 1.342, "Best model per class": 1.400},
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
    """Syntetos–Boylan quadrants with the workshop sample (training weeks)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import bootcamp as bc
    info = bc.series_info(bc.load_panel()).to_pandas()
    colors = {"smooth": GREEN, "erratic": BLUE, "intermittent": ORANGE, "lumpy": INK}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for t, g in info.groupby("demand_type"):
        ax.scatter(g["adi"].clip(upper=12), g["cv2_nonzero"].clip(upper=4), s=9, alpha=0.5, color=colors[t],
                   label=f"{t} ({len(g)} items)")
    ax.axvline(1.32, color=GREY, ls="--"); ax.axhline(0.49, color=GREY, ls="--")
    ax.set_xscale("log"); ax.set_xticks([1, 1.32, 2, 4, 8], ["1", "1.32", "2", "4", "8+"])
    ax.set_xlabel("ADI: average weeks between sales (log scale)"); ax.set_ylabel("CV² of non-zero weeks")
    ax.set_title("How often it sells × how steady it is (2,000-item sample, training weeks)")
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
    barh(axes[0], RESULTS["lab01"], cols); axes[0].set_xlim(1.0, 1.5)
    axes[0].set_title("Test RMSSE: the deep models land on top of each other")
    cv = RESULTS["lab01_cv"]
    axes[1].bar(list(cv), list(cv.values()), color=[ORANGE, ORANGE])
    for k, v in enumerate(cv.values()):
        axes[1].text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    axes[1].set_title("Same NHITS, two CV folds"); axes[1].set_ylabel("RMSSE")
    save(fig, "res_lab01")


def res_lab02():
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    barh(axes[0], RESULTS["lab02_spread"], [ORANGE, INK, INK, INK])
    axes[0].set_title("Validation RMSSE swing per knob"); axes[0].set_xlim(0, 0.32)
    axes[0].axvline(RESULTS["lab02_seed"], color=ORANGE, ls="--", lw=1.5, label="seed noise (same settings, 3 seeds)")
    axes[0].legend(frameon=False, fontsize=11, loc="lower right")
    for k, (name, (sec, r)) in enumerate(RESULTS["lab02_cost"].items()):
        axes[1].scatter(sec, r, s=180, color=[INK, ORANGE][k], zorder=3)
        axes[1].text(sec, r + 0.003, name, ha="center" if k == 0 else "right", va="bottom", fontsize=12)
    axes[1].scatter(1.8, 1.169, s=180, color=BLUE, zorder=3)
    axes[1].text(1.8, 1.169 - 0.004, "LightGBM", ha="left", va="top", fontsize=12)
    axes[1].set_xscale("log"); axes[1].set_xlim(0.8, 2000); axes[1].set_ylim(1.16, 1.202)
    axes[1].set_xlabel("fit time (s, log scale)"); axes[1].set_ylabel("test RMSSE")
    axes[1].set_title("15× the compute for 1.6 % better")
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
    ax.set_xticks(range(3), keys, fontsize=13); ax.set_ylim(0, 1.75); ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("Same forecasts, different verdicts"); ax.legend(frameon=False, fontsize=11, loc="upper right", ncol=1)
    barh(axes[1], RESULTS["lab03_lift"], [GREEN, INK, INK, INK], fmt="{:.2f}")
    axes[1].set_title("Median promo lift"); axes[1].set_xlim(0, 0.9)
    barh(axes[2], RESULTS["lab03_mono"], [INK, INK, BLUE], fmt="{:.0%}")
    axes[2].set_title("Price curves sloping down"); axes[2].set_xlim(0, 1.2)
    save(fig, "res_lab03")


def res_lab05():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    lv = RESULTS["lab05_levels"]
    base = np.array(RESULTS["lab05"]["Base (incoherent)"])
    styles = {"BottomUp": (BLUE, "-"), "TopDown": (ORANGE, "-"), "MinT-shrink": (GREEN, "-")}
    for name, (c, ls) in styles.items():
        ax.plot(lv, np.array(RESULTS["lab05"][name]) / base, marker="o", color=c, ls=ls, lw=2.5, label=name)
    ax.axhline(1, color=INK, lw=1, ls="--"); ax.text(4.05, 1.0, "base", va="center", fontsize=12)
    ax.set_ylim(0.85, 1.4)
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
    ax.set_title("1 in 6 series makes 40 % of the mean")
    barh(axes[1], RESULTS["lab07_holdout"], [GREEN, INK, GREY, GREY])
    axes[1].set_xlim(1.1, 1.48); axes[1].set_title("Held-out half: averaging beats picking")
    save(fig, "res_lab07")


if __name__ == "__main__":
    for f in [rmsse_ruler, global_vs_local, nbeats_relay, window_scaling, loss_shapes, response_sketch,
              hierarchy_tree, pinball, conformal, elo_curve, res_day1, res_lab01, res_lab02, res_lab03,
              res_lab05, res_lab06, res_lab07, rolling_origin, fm_patches, crps, demand_types]:
        f()
    print("wrote", len(list(OUT.glob("*.png"))), "figures to", OUT)
