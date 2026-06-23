# Controlled Split Delivery Vehicle Routing Problem (C-SDVRP)

Code, data, and results for the Controlled Split Delivery Vehicle Routing
Problem and its solution methods. The C-SDVRP extends the split delivery VRP
with three operational constraints:

- **N1 – controlled split activation**: a customer may be served by more than
  one vehicle only if its demand reaches a capacity-based threshold
  (`d_i >= beta_Q * Q`); smaller orders must be served in a single visit.
- **N2 – receiving capacity**: the number of vehicle visits per customer is
  capped at `U_max`.
- **N3 – split-linked emissions**: each extra visit to a customer carries an
  emission penalty in the objective.

The repository contains an exact MILP solver (for small instances), several
metaheuristics, the full benchmark and case-study results, the trained
policies, and scripts to reproduce every figure.

## Methods

| Name | Description |
|------|-------------|
| CW | Clarke–Wright savings (construction heuristic, used as the start solution) |
| VNS | Variable Neighborhood Search with RVND local search |
| ALNS+ | Adaptive Large Neighborhood Search with RVND local search and an 18-action operator/removal-size space |
| DRL-ALNS | ALNS+ with a PPO policy selecting the operator/removal-size action |
| DRL-VNS | VNS whose shake neighborhood and strength are chosen by a PPO policy |

## Repository layout

```
src/                 problem model, solvers, instance generator, exact MILP
  instance.py        instance container and distance matrix
  solution.py        solution representation and objective
  operators.py       destroy / repair / local-search operators
  clarke_wright.py   savings construction
  vns.py             Variable Neighborhood Search
  alns_plus.py       ALNS+ and DRL-ALNS
  drl_vns.py         DRL-VNS
  ppo.py             PPO actor-critic agent
  exact_milp.py      exact MILP (PuLP/CBC)
  gen_benchmark.py   benchmark instance generator
train_drl_vns.py     train the DRL-VNS policy
train_drl_alns.py    train the DRL-ALNS policy
run_benchmark.py     equal-time comparison of all methods
run_exact_gaps.py    optimality gaps vs the exact MILP
run_price_of_control.py   effect of N1/N2/N3 on cost and number of splits
run_case_study.py    NUPCO–Riyadh hospital case study
run_statistics.py    Wilcoxon signed-rank tests
make_figures.py      figures from the result files
models/              trained PPO checkpoints
results/             result CSV files
logs/                training and experiment logs
figures/             figures used in the paper
data/case_study_riyadh/   the case-study instance
```

## Requirements

```
pip install -r requirements.txt
```

## Reproducing the results

```
python train_drl_vns.py        # trains models/ppo_drl_vns.pt
python run_benchmark.py        # results/benchmark.csv  (equal wall-clock time)
python run_statistics.py       # results/statistics.csv
python run_exact_gaps.py       # results/exact_gaps.csv
python run_price_of_control.py # results/price_of_control.csv
python run_case_study.py       # results/case_study/
python make_figures.py         # figures/
```

Benchmark instances are generated on the fly from `(n, size_class, seed)` in
`src/gen_benchmark.py`, so every instance is reproducible without storing it.

## Summary of results

Equal wall-clock comparison over 42 instances (mean improvement over the
Clarke–Wright start solution):

| Class | VNS | ALNS+ | DRL-ALNS | DRL-VNS |
|-------|-----|-------|----------|---------|
| S | 8.98% | 7.20% | 6.67% | 8.50% |
| M | 11.19% | 9.04% | 8.82% | 11.41% |
| L | 7.16% | 6.98% | 6.98% | 8.55% |
| XL | 7.92% | 7.50% | 7.50% | 7.95% |
| Overall | 8.41% | 7.59% | 7.49% | **8.94%** |

DRL-VNS gives the lowest mean cost overall. The improvement over standard VNS
is significant on the paired Wilcoxon test (overall p = 0.008; Class-L
p = 0.0002, winning all 12 instances).

On the small instances solved to proven optimality by the MILP, all
metaheuristics are within 0–6% of the optimum. Comparing the full C-SDVRP
with the unconstrained SDVRP, the N1/N2/N3 constraints add roughly 3–6% to
routing cost while removing 74–85% of the split operations.
