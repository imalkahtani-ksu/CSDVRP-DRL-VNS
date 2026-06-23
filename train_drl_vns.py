"""
train_drl_vns.py — Train the DRL-VNS agent (learned adaptive shaking).
Saves models/ppo_drl_vns.pt.
"""
import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

_LOG = Path(__file__).parent / "train_drl_vns.log"
_fh = open(_LOG, "w", encoding="utf-8", buffering=1)
def log(m): print(m, flush=True); _fh.write(m+"\n"); _fh.flush()

from src.gen_benchmark import make_instance
from src.ppo import PPOAgent
from src.drl_vns import DRLVNS, N_ACTIONS_VNS, STATE_DIM_VNS

EPISODES = 240
VALID_EVERY = 30
IMAX_CAP = 600
MODEL = Path(__file__).parent / "models" / "ppo_drl_vns.pt"
MODEL.parent.mkdir(exist_ok=True)
TRAIN_SIZES = [("S",15),("S",20),("M",30),("M",50),("L",75)]
VALID_SIZES = [("S",20),("M",30),("M",50)]
MASTER = 42

def imax(n): return min(2500 if n<=50 else 1500, IMAX_CAP)

def main():
    log("="*60); log("  DRL-VNS Training (learned adaptive shaking)"); log("="*60)
    agent = PPOAgent(n_actions=N_ACTIONS_VNS, state_dim=STATE_DIM_VNS)
    solver = DRLVNS()
    valid = [make_instance(n, sc, seed=9000+i*13) for i,(sc,n) in enumerate(VALID_SIZES)]
    best = -1e9; t0=time.perf_counter()
    for ep in range(EPISODES):
        sc,n = TRAIN_SIZES[ep % len(TRAIN_SIZES)]
        try:
            inst = make_instance(n, size_class=sc, seed=MASTER+ep*97+3)
            agent.net.train()
            te=time.perf_counter()
            r = solver.solve(inst, agent, I_max=imax(n), seed=MASTER+ep, train=True)
            el=time.perf_counter()-te
            gap=(r["Z_best"]-r["Z_init"])/max(r["Z_init"],1e-6)*100
        except Exception as e:
            log(f"  Ep {ep+1} ERROR ({n}): {e}"); continue
        if (ep+1)%VALID_EVERY==0:
            agent.net.eval(); vg=[]
            for vi in valid:
                vr=solver.solve(vi, agent, I_max=imax(vi.n), seed=0, train=False)
                vg.append((vr["Z_init"]-vr["Z_best"])/max(vr["Z_init"],1e-6)*100)
            av=float(np.mean(vg)); mark=""
            if av>best: best=av; agent.save(str(MODEL)); mark=" *** SAVED"
            log(f"  Ep {ep+1:4d}/{EPISODES} | n={n:3d} | gap={gap:6.2f}% | valid={av:6.2f}% "
                f"| {el:.1f}s | total {(time.perf_counter()-t0)/60:.1f}min{mark}")
        elif (ep+1)%5==0:
            log(f"  Ep {ep+1:4d}/{EPISODES} | n={n:3d} | gap={gap:6.2f}% | {el:.1f}s")
    if not MODEL.exists(): agent.save(str(MODEL))
    log(f"\n  Done in {(time.perf_counter()-t0)/60:.1f} min. Best valid: {best:.2f}%")
    _fh.close()

if __name__=="__main__": main()
