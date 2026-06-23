"""
run_statistics.py — Wilcoxon analysis for the equal-time v4 benchmark.
Reads results/benchmark.csv; writes results/statistics.csv, results/benchmark_aggregated.csv.
"""
import sys, os, csv
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(BASE, "results", "benchmark.csv")

def eff_r(stat, d):
    nz=d[d!=0]; n=len(nz)
    if n==0: return 0.0
    z=abs(stat-n*(n+1)/4)/np.sqrt(n*(n+1)*(2*n+1)/24); return z/np.sqrt(n)

def test(a,b):
    d=np.array(a)-np.array(b)
    if np.all(d==0) or len(d[d!=0])<3: return dict(p=1.0,r=0.0,n=len(d),wins=int((d>0).sum()))
    s,p=wilcoxon(d,alternative="greater"); return dict(p=float(p),r=float(eff_r(s,d)),n=len(d),wins=int((d>0).sum()))

def main():
    if not os.path.exists(RAW):
        print("no benchmark.csv yet"); return
    raw=pd.read_csv(RAW)
    piv=(raw.groupby(["instance","size_class","n","method"])["Z_best"].mean().reset_index()
            .pivot(index=["instance","size_class","n"],columns="method",values="Z_best").reset_index())
    piv.columns.name=None
    piv.to_csv(os.path.join(BASE,"results","benchmark_aggregated.csv"),index=False)
    cols=[c for c in ["CW","VNS","ALNS","ALNS+","DRL+","DRL-VNS"] if c in piv.columns]
    print("Methods:",cols,"| instances:",len(piv))
    rows=[]
    defs=[("DRL-VNS<VNS","VNS","DRL-VNS"),("DRL-VNS<ALNS+","ALNS+","DRL-VNS"),
          ("DRL-VNS<DRL+","DRL+","DRL-VNS"),
          ("DRL+<VNS","VNS","DRL+"),("DRL+<ALNS+","ALNS+","DRL+"),
          ("ALNS+<VNS","VNS","ALNS+"),("ALNS+<ALNS","ALNS","ALNS+"),
          ("VNS<ALNS","ALNS","VNS")]
    def block(name,df):
        for label,hi,lo in defs:
            if hi in df and lo in df:
                t=test(df[hi].values,df[lo].values)
                sig="***" if t["p"]<0.001 else "**" if t["p"]<0.01 else "*" if t["p"]<0.05 else "n.s."
                rows.append({"scope":name,"test":label,"N":t["n"],"p":t["p"],"r":round(t["r"],3),"wins":t["wins"],"sig":sig})
                print(f"  {name:8s} {label:12s} N={t['n']:3d} p={t['p']:.4g} r={t['r']:.3f} {sig}",flush=True)
    block("Overall",piv)
    for sc in ["S","M","L","XL"]:
        sub=piv[piv["size_class"]==sc]
        if len(sub): block(sc,sub)
    print("\nMean improvement over CW (%):")
    for sc in ["S","M","L","XL","Overall"]:
        sub = piv if sc=="Overall" else piv[piv["size_class"]==sc]
        if not len(sub): continue
        cw=sub["CW"].mean(); line=f"  {sc:8s}"
        for m in ["VNS","ALNS","ALNS+","DRL+","DRL-VNS"]:
            if m in sub: line+=f" | {m} {(cw-sub[m].mean())/cw*100:5.2f}%"
        print(line)
    with open(os.path.join(BASE,"results","statistics.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["scope","test","N","p","r","wins","sig"]); w.writeheader(); w.writerows(rows)
    print("\nSaved results/statistics.csv and results/benchmark_aggregated.csv")

if __name__=="__main__": main()
