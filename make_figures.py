# Generate the figures used in the paper from the result CSV files.
# Run after the experiments; outputs go to figures/.
import os, re, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(BASE, "results")
FIG  = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
COL = {"CW": "#9aa0a6", "VNS": "#1f77b4", "ALNS": "#7f7f7f",
       "ALNS+": "#2ca02c", "DRL+": "#ff7f0e", "DRL-VNS": "#d62728"}


def recover_exact_gaps():
    # Use the saved CSV if present; otherwise parse it from the run log.
    csv_path = os.path.join(RES, "exact_gaps.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    log = os.path.join(BASE, "logs", "exact_gaps_run.log")
    rows = []
    pat = re.compile(r"n=\s*(\d+) s=(\d+) elig=\s*(\d+) \| MILP (\w+)\s+Z=([\d.]+) "
                     r"LB=([\d.]+) \| CW ([+\-][\d.]+)% VNS ([+\-][\d.]+)% "
                     r"ALNS ([+\-][\d.]+)% DRL ([+\-][\d.]+)%")
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                n, s, e, st, Z, LB, cw, vns, al, dr = m.groups()
                rows.append(dict(n=int(n), seed=int(s), status=st,
                                 Z=float(Z), LB=float(LB),
                                 CW_gap=float(cw), VNS_gap=float(vns),
                                 ALNSp_gap=float(al), DRLp_gap=float(dr)))
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


def fig_optimality_gap(df):
    # Mean optimality gap of each heuristic by instance size (proven-optimal n).
    sizes = sorted(df["n"].unique())
    methods = [("VNS", "VNS_gap"), ("ALNS+", "ALNSp_gap"), ("DRL+", "DRLp_gap")]
    x = np.arange(len(sizes)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (name, col) in enumerate(methods):
        means = [max(0.0, df[df.n == n][col].mean()) for n in sizes]
        ax.bar(x + (i - 1) * w, means, w, label=name, color=COL[name])
    ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in sizes])
    ax.set_ylabel("Mean gap to optimum (%)")
    ax.set_title("Optimality gap of the metaheuristics vs. exact MILP")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_optimality_gap.png")); plt.close(fig)


def fig_method_comparison(agg):
    classes = ["S", "M", "L", "XL"]
    methods = ["VNS", "ALNS+", "DRL+", "DRL-VNS"]
    x = np.arange(len(classes)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(methods):
        vals = []
        for sc in classes:
            sub = agg[agg.size_class == sc]
            cw = sub["CW"].mean()
            vals.append((cw - sub[m].mean()) / cw * 100 if m in sub else 0)
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=COL[m])
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_xlabel("Instance class"); ax.set_ylabel("Mean improvement over CW (%)")
    ax.set_title("Solution quality by method and instance class (equal time)")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_method_comparison.png")); plt.close(fig)


def fig_drlvns_vs_vns(agg):
    # Per-instance percentage gap of DRL-VNS relative to VNS (negative = DRL-VNS better).
    classes = ["S", "M", "L", "XL"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for sc in classes:
        sub = agg[agg.size_class == sc]
        rel = (sub["DRL-VNS"].values - sub["VNS"].values) / sub["VNS"].values * 100
        ax.scatter([sc] * len(rel), rel, color=COL["DRL-VNS"], alpha=0.7, s=40)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("DRL-VNS minus VNS (% of VNS cost)")
    ax.set_title("DRL-VNS vs. standard VNS per instance (below 0 = DRL-VNS better)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_drlvns_vs_vns.png")); plt.close(fig)


def fig_price_of_control(poc):
    classes = ["S", "M", "L"]
    cost_pen, split_cut = [], []
    for sc in classes:
        sub = poc[poc.size_class == sc]
        full = sub[sub.config == "FULL"]; sdv = sub[sub.config == "SDVRP"]
        cost_pen.append((full.total.mean() - sdv.total.mean()) / sdv.total.mean() * 100)
        split_cut.append((1 - full.n_split_cust.mean() / sdv.n_split_cust.mean()) * 100)
    x = np.arange(len(classes))
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx(); ax2.grid(False)
    ax1.bar(x - 0.2, cost_pen, 0.4, color="#d62728", label="Cost penalty")
    ax2.bar(x + 0.2, split_cut, 0.4, color="#1f77b4", label="Splits eliminated")
    ax1.set_xticks(x); ax1.set_xticklabels(classes)
    ax1.set_xlabel("Instance class")
    ax1.set_ylabel("Cost penalty of control (%)", color="#d62728")
    ax2.set_ylabel("Split operations eliminated (%)", color="#1f77b4")
    ax1.set_title("Price of control: cost vs. split reduction (C-SDVRP vs. plain SDVRP)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_price_of_control.png")); plt.close(fig)


def fig_casestudy():
    d = pd.read_csv(os.path.join(RES, "case_study", "main.csv"))
    sd = pd.read_csv(os.path.join(RES, "case_study", "sensitivity_delta.csv"))
    su = pd.read_csv(os.path.join(RES, "case_study", "sensitivity_umax.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(d["method"], d["Z"], color=[COL.get(m, "#888") for m in d["method"]])
    ax[0].set_title("Riyadh case study: objective by method"); ax[0].set_ylabel("Z")
    for t in ax[0].get_xticklabels(): t.set_rotation(20)
    ax[1].plot(sd["delta"], sd["Z_DRLVNS"], "o-", color=COL["DRL-VNS"])
    ax[1].set_xlabel("Emission penalty $\\delta$"); ax[1].set_ylabel("Z"); ax[1].set_title("Sensitivity to $\\delta$")
    ax[2].bar(su["U_max"].astype(str), su["Z_DRLVNS"], color="#2ca02c")
    ax[2].set_xlabel("Tier-1 visit cap $U_{max}$"); ax[2].set_ylabel("Z"); ax[2].set_title("Sensitivity to $U_{max}$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_casestudy.png")); plt.close(fig)


def fig_training():
    log = os.path.join(BASE, "train_drl_vns.log")
    eps, val = [], []
    pat = re.compile(r"Ep\s+(\d+)/\d+.*valid=\s*([\d.]+)%")
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m: eps.append(int(m.group(1))); val.append(float(m.group(2)))
    if eps:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(eps, val, "o-", color=COL["DRL-VNS"])
        ax.set_xlabel("Training episode"); ax.set_ylabel("Validation improvement over CW (%)")
        ax.set_title("DRL-VNS training convergence")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_training.png")); plt.close(fig)


def main():
    ex = recover_exact_gaps()
    agg = pd.read_csv(os.path.join(RES, "benchmark_aggregated.csv"))
    poc = pd.read_csv(os.path.join(RES, "price_of_control.csv"))
    fig_optimality_gap(ex)
    fig_method_comparison(agg)
    fig_drlvns_vs_vns(agg)
    fig_price_of_control(poc)
    fig_casestudy()
    fig_training()
    print("Figures written to", FIG)
    print("Files:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
