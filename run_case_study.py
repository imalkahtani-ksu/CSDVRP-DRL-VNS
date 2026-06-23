"""
run_case_study.py — NUPCO-Riyadh case study under the v4 framework.
Equal-time comparison of CW / VNS / ALNS+ / DRL-VNS (proposed) on the
30-hospital instance, plus sensitivity analysis on delta and U_max using
the proposed DRL-VNS solver. Writes results/case_study/*.csv
"""
import sys, os, csv, json, time
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.instance      import CSDVRPInstance
from src.vns           import VNS
from src.alns_plus     import ClassicalALNSPlus
from src.drl_vns       import DRLVNS, N_ACTIONS_VNS, STATE_DIM_VNS
from src.ppo           import PPOAgent
from src.clarke_wright import clarke_wright

BASE = os.path.dirname(os.path.abspath(__file__))
CS   = os.path.join(BASE, "C_SDVRP_Benchmark", "case_study_riyadh")
OUT  = os.path.join(BASE, "results", "case_study"); os.makedirs(OUT, exist_ok=True)
MODEL_VNS = os.path.join(BASE, "models", "ppo_drl_vns.pt")
BUDGET = 30; N_RUNS = 5; BIG = 10_000_000

def load_instance(delta=None, umax_cap=None):
    nodes = pd.read_csv(os.path.join(CS, "nodes.csv"))
    veh   = pd.read_csv(os.path.join(CS, "vehicles.csv"))
    with open(os.path.join(CS, "params.json")) as f: params = json.load(f)
    params["n"] = int((nodes["node_type"]=="customer").sum())
    if delta is not None: params["delta"] = delta
    if umax_cap is not None:
        nodes.loc[(nodes.node_type=="customer") & (nodes.U_max>1), "U_max"] = umax_cap
    return CSDVRPInstance(nodes, veh, params)

def main():
    agent = PPOAgent(n_actions=N_ACTIONS_VNS, state_dim=STATE_DIM_VNS)
    if os.path.exists(MODEL_VNS): agent.load(MODEL_VNS, for_inference=True)
    inst = load_instance()
    print(f"CSDVRP-CS-Riyadh n={inst.n} Q={inst.params.get('Q')} "
          f"elig={int((inst.nodes.U_max>1).sum())}", flush=True)

    # main comparison (equal time)
    rows=[]
    cw = clarke_wright(inst, seed=0).objective()
    rows.append({"method":"CW","Z":round(cw,2),"impr_pct":0.0})
    for name, fn in [("VNS", lambda r: VNS().solve(inst,I_max=BIG,seed=r,time_limit=BUDGET)),
                     ("ALNS+", lambda r: ClassicalALNSPlus().solve(inst,I_max=BIG,seed=r,time_limit=BUDGET)),
                     ("DRL-VNS", lambda r: DRLVNS().solve(inst,agent,I_max=BIG,seed=r,train=False,time_limit=BUDGET))]:
        zs=[fn(r)["Z_best"] for r in range(N_RUNS)]
        zb=min(zs)
        rows.append({"method":name,"Z":round(zb,2),"impr_pct":round((cw-zb)/cw*100,2)})
        print(f"  {name}: Z={zb:.1f} ({(cw-zb)/cw*100:.1f}%)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT,"main.csv"), index=False)

    # sensitivity: delta
    sd=[]
    for delta in [0,5,10,15,20,25,30]:
        i2=load_instance(delta=delta)
        z=min(DRLVNS().solve(i2,agent,I_max=BIG,seed=r,train=False,time_limit=BUDGET)["Z_best"] for r in range(3))
        sd.append({"delta":delta,"Z_DRLVNS":round(z,2)}); print(f"  delta={delta}: {z:.1f}",flush=True)
    pd.DataFrame(sd).to_csv(os.path.join(OUT,"sensitivity_delta.csv"), index=False)

    # sensitivity: U_max (Tier-1 cap)
    su=[]
    for um in [1,2,3]:
        i2=load_instance(umax_cap=um)
        z=min(DRLVNS().solve(i2,agent,I_max=BIG,seed=r,train=False,time_limit=BUDGET)["Z_best"] for r in range(3))
        su.append({"U_max":um,"Z_DRLVNS":round(z,2)}); print(f"  U_max={um}: {z:.1f}",flush=True)
    pd.DataFrame(su).to_csv(os.path.join(OUT,"sensitivity_umax.csv"), index=False)
    print("Case study v2 complete ->", OUT, flush=True)

if __name__=="__main__": main()
