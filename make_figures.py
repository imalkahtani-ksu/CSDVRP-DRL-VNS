"""
make_figures.py — figures of the revised study, built from results_v2/.

Run analyze.py first, then:  python make_figures.py
Outputs go to figures_v2/.
"""
import glob, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_v2"
FIG = ROOT / "figures_v2"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})

ORDER = ["VNS", "ALNS+", "DRL-ALNS", "VNS-9", "Fixed-small", "Random-9",
         "Roulette-9", "Marginal-9", "DRL-VNS"]
COL = {"VNS": "#1f77b4", "ALNS+": "#2ca02c", "DRL-ALNS": "#ff7f0e",
       "VNS-9": "#9467bd", "Fixed-small": "#8c564b", "Random-9": "#7f7f7f",
       "Roulette-9": "#17becf", "Marginal-9": "#bcbd22", "DRL-VNS": "#d62728"}
CLASSES = ["S", "M", "L", "XL"]
STRENGTHS = ["small", "medium", "large"]
REMOVALS = ["random", "related", "worst"]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def fig_method_comparison(per):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(CLASSES))
    w = 0.09
    for i, m in enumerate(ORDER):
        vals = [per[(per.size_class == c) & (per.method == m)].impr_pct.mean()
                for c in CLASSES]
        ax.bar(x + (i - len(ORDER) / 2 + 0.5) * w, vals, w, label=m, color=COL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in CLASSES])
    ax.set_ylabel("Mean improvement over Clarke–Wright (%)")
    ax.set_title("Solution quality by method and instance class (equal CPU budget, 10 runs)")
    ax.legend(ncol=5, fontsize=8)
    _save(fig, "fig_method_comparison.png")


def fig_boxplots(per):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, c in zip(axes, CLASSES):
        sub = per[per.size_class == c]
        data = [sub[sub.method == m].impr_pct.values for m in ORDER]
        bp = ax.boxplot(data, patch_artist=True, labels=ORDER)
        for box, m in zip(bp["boxes"], ORDER):
            box.set_facecolor(COL[m])
            box.set_alpha(0.75)
        ax.set_title(f"Class {c}")
        ax.set_ylabel("Improvement over CW (%)")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.suptitle("Distribution over instances of the mean improvement per method")
    _save(fig, "fig_boxplots.png")


def fig_vs_controls(per):
    """Per-instance difference between DRL-VNS and each control (negative = DRL-VNS better)."""
    piv = per.pivot_table(index=["instance", "size_class"], columns="method",
                          values="mean_Z")
    controls = [m for m in ["VNS", "VNS-9", "Fixed-small", "Random-9",
                            "Roulette-9", "Marginal-9"] if m in piv]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, m in enumerate(controls):
        d = (piv["DRL-VNS"] - piv[m]) / piv[m] * 100
        ax.scatter(np.full(len(d), i) + np.random.uniform(-.12, .12, len(d)),
                   d, s=14, alpha=0.6, color=COL[m])
        ax.hlines(d.mean(), i - .25, i + .25, color="black", lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(controls)))
    ax.set_xticklabels(controls, rotation=15)
    ax.set_ylabel("DRL-VNS minus control (% of control cost)")
    ax.set_title("Per-instance difference to each control (below 0 = DRL-VNS better);"
                 " bars are means")
    _save(fig, "fig_vs_controls.png")


def fig_action_usage(usage):
    u = usage[usage.method == "DRL-VNS"]
    if u.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # by search phase
    t = (u.groupby(["phase", "action"]).share.sum().unstack(fill_value=0))
    t = t.div(t.sum(axis=1), axis=0).reindex(["early", "middle", "late"])
    im = axes[0].imshow(t.values, cmap="YlOrRd", vmin=0, aspect="auto")
    axes[0].set_xticks(range(len(t.columns)))
    axes[0].set_xticklabels(t.columns, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(t.index)))
    axes[0].set_yticklabels(t.index)
    axes[0].set_title("Action use by search phase")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], label="share of steps")
    # by stagnation
    t2 = (u.groupby(["stagnation", "action"]).share.sum().unstack(fill_value=0))
    t2 = t2.div(t2.sum(axis=1), axis=0).reindex(["0-4", "5-19", "20+"])
    im2 = axes[1].imshow(t2.values, cmap="YlOrRd", vmin=0, aspect="auto")
    axes[1].set_xticks(range(len(t2.columns)))
    axes[1].set_xticklabels(t2.columns, rotation=90, fontsize=7)
    axes[1].set_yticks(range(len(t2.index)))
    axes[1].set_yticklabels(t2.index)
    axes[1].set_title("Action use by steps since last improvement")
    axes[1].grid(False)
    fig.colorbar(im2, ax=axes[1], label="share of steps")
    _save(fig, "fig_action_usage.png")


def fig_strength_by_size(usage):
    u = usage[usage.method == "DRL-VNS"].copy()
    if u.empty:
        return
    u["strength"] = u.action.str.split("-").str[1]
    t = (u.groupby(["size_class", "strength"]).share.sum().unstack(fill_value=0))
    t = t.div(t.sum(axis=1), axis=0).reindex(CLASSES)[STRENGTHS]
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(t))
    for s, c in zip(STRENGTHS, ["#9ecae1", "#4292c6", "#08519c"]):
        ax.bar(t.index, t[s], bottom=bottom, label=s, color=c)
        bottom += t[s].values
    ax.set_ylabel("Share of shakes")
    ax.set_title("Learned shake strength by instance class (DRL-VNS)")
    ax.legend()
    _save(fig, "fig_strength_by_size.png")


def fig_action_success(succ):
    s = succ[succ.method == "DRL-VNS"]
    if s.empty:
        return
    t = s.groupby("action")[["uses", "improvements"]].sum()
    t["rate"] = t.improvements / t.uses * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t.index, t.rate, color="#d62728")
    ax.set_ylabel("Steps that improved the incumbent (%)")
    ax.set_title("Success rate of each shake action (DRL-VNS)")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    _save(fig, "fig_action_success.png")


def fig_convergence():
    """Mean best-so-far against CPU time on the largest class."""
    curves = {}
    for f in glob.glob(str(RES / "runs_main_exact_case_shard*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if "n150" not in d["instance"] or d["config"] != "base":
                continue
            h = np.asarray(d["history"], float)
            if len(h) < 2:
                continue
            grid = np.linspace(0, 50, 100)
            vals = np.interp(grid, h[:, 0], h[:, 1] / h[0, 1])
            curves.setdefault(d["method"], []).append(vals)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    grid = np.linspace(0, 50, 100)
    for m in ORDER:
        if m in curves:
            ax.plot(grid, np.mean(curves[m], axis=0), label=m, color=COL[m])
    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel("Best objective / initial objective")
    ax.set_title("Convergence on class XL ($n=150$), mean over instances and runs")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig_convergence.png")


def fig_optimality_gap():
    f = RES / "exact_gaps.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    g = g[g.proven == True]
    if g.empty:
        return
    sizes = sorted(g.n.unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sizes))
    w = 0.09
    for i, m in enumerate(ORDER):
        vals = [g[(g.n == n) & (g.method == m)].mean_gap_pct.mean() for n in sizes]
        ax.bar(x + (i - len(ORDER) / 2 + 0.5) * w, vals, w, label=m, color=COL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in sizes])
    ax.set_ylabel("Mean gap to proven optimum (%)")
    ax.set_title("Optimality gap on instances solved exactly")
    ax.legend(ncol=5, fontsize=8)
    _save(fig, "fig_optimality_gap.png")


def fig_price_of_control():
    fs = glob.glob(str(RES / "runs_poc_shard*.csv"))
    if not fs:
        return
    p = pd.concat([pd.read_csv(f) for f in fs])
    g = p.groupby(["n", "config"]).agg(Z=("Z_best", "mean"),
                                       splits=("split_customers", "mean")).reset_index()
    base = g[g.config == "SDVRP"].set_index("n")
    rows = []
    for cfg in ["FULL", "no-N1", "no-N2", "no-N3"]:
        s = g[g.config == cfg].set_index("n")
        for n in s.index:
            rows.append({"n": n, "config": cfg,
                         "cost_pct": (s.Z[n] - base.Z[n]) / base.Z[n] * 100,
                         "split_pct": (1 - s.splits[n] / max(base.splits[n], 1e-9)) * 100})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cfg, c in zip(["FULL", "no-N1", "no-N2", "no-N3"],
                      ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]):
        s = d[d.config == cfg]
        axes[0].plot(s.n, s.cost_pct, "o-", label=cfg, color=c)
        axes[1].plot(s.n, s.split_pct, "o-", label=cfg, color=c)
    axes[0].set_xlabel("customers"); axes[0].set_ylabel("Cost relative to plain SDVRP (%)")
    axes[0].set_title("Price of control")
    axes[1].set_xlabel("customers"); axes[1].set_ylabel("Split customers removed (%)")
    axes[1].set_title("Effect on splitting")
    for a in axes:
        a.legend(fontsize=8)
    _save(fig, "fig_price_of_control.png")


def fig_sensitivity():
    fs = glob.glob(str(RES / "runs_sens_csens_shard*.csv"))
    if not fs:
        return
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d[d.suite == "sens"]
    if d.empty:
        return
    d["param"] = d.config.str.split("=").str[0]
    d["value"] = pd.to_numeric(d.config.str.split("=").str[1], errors="coerce")
    params = [p for p in ["beta", "mu", "delta", "umax"] if p in set(d.param)]
    fig, axes = plt.subplots(1, len(params), figsize=(3.4 * len(params), 3.6))
    if len(params) == 1:
        axes = [axes]
    titles = {"beta": r"$\beta_Q$", "mu": r"$\mu$", "delta": r"$\delta$",
              "umax": r"$U_{\max}$"}
    for ax, p in zip(axes, params):
        s = d[d.param == p]
        for m in ["VNS", "DRL-VNS"]:
            t = (s[s.method == m].groupby("value")
                 .agg(Z=("Z_best", "mean"), sp=("split_customers", "mean")))
            if len(t):
                ax.plot(t.index, t.Z, "o-", label=m, color=COL[m])
        ax.set_xlabel(titles[p]); ax.set_ylabel("Objective")
        ax.set_title(f"Sensitivity to {titles[p]}")
        ax.legend(fontsize=8)
    _save(fig, "fig_sensitivity.png")


def fig_case():
    f = RES / "case_study.csv"
    if not f.exists():
        return
    c = pd.read_csv(f).sort_values("mean_Z")
    fs = glob.glob(str(RES / "runs_sens_csens_shard*.csv"))
    fig, axes = plt.subplots(1, 3 if fs else 1, figsize=(13 if fs else 5, 4))
    axes = np.atleast_1d(axes)
    axes[0].bar(c.method, c.mean_Z, yerr=c.sd_Z, capsize=3,
                color=[COL.get(m, "#888") for m in c.method])
    axes[0].set_ylabel("Objective (mean of 10 runs)")
    axes[0].set_title("Riyadh case study")
    axes[0].tick_params(axis="x", rotation=90, labelsize=8)
    axes[0].set_ylim(bottom=min(c.mean_Z) * 0.97)
    if fs:
        d = pd.concat([pd.read_csv(f) for f in fs])
        d = d[d.suite == "csens"].copy()
        d["param"] = d.config.str.split("=").str[0]
        d["value"] = pd.to_numeric(d.config.str.split("=").str[1], errors="coerce")
        for ax, p, lab in [(axes[1], "delta", r"$\delta$"), (axes[2], "umax", r"$U_{\max}$")]:
            s = d[d.param == p]
            for m in ["VNS", "DRL-VNS"]:
                t = s[s.method == m].groupby("value").Z_best.mean()
                if len(t):
                    ax.plot(t.index, t.values, "o-", label=m, color=COL[m])
            ax.set_xlabel(lab); ax.set_ylabel("Objective")
            ax.set_title(f"Case study: sensitivity to {lab}")
            ax.legend(fontsize=8)
    _save(fig, "fig_casestudy.png")


def fig_training():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for f in sorted(glob.glob(str(ROOT / "models_v2" / "vns_seed*_curve.json"))):
        c = pd.DataFrame(json.load(open(f)))
        c = c.dropna(subset=["valid_gain"])
        if len(c):
            ax.plot(c.episode, c.valid_gain, "o-", alpha=0.8,
                    label=Path(f).stem.split("_")[1])
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Validation improvement over CW (%)")
    ax.set_title("DRL-VNS training: five independently trained policies")
    ax.legend(fontsize=8, title="seed")
    _save(fig, "fig_training.png")


def main():
    per = pd.read_csv(RES / "per_instance.csv")
    fig_method_comparison(per)
    fig_boxplots(per)
    fig_vs_controls(per)
    if (RES / "action_usage.csv").exists():
        usage = pd.read_csv(RES / "action_usage.csv")
        fig_action_usage(usage)
        fig_strength_by_size(usage)
    if (RES / "action_success.csv").exists():
        fig_action_success(pd.read_csv(RES / "action_success.csv"))
    fig_convergence()
    fig_optimality_gap()
    fig_price_of_control()
    fig_sensitivity()
    fig_case()
    fig_training()


if __name__ == "__main__":
    main()
