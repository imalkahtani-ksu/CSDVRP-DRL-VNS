# Controlled Split Delivery Vehicle Routing Problem (C-SDVRP)

Code, data and results for the Controlled Split Delivery Vehicle Routing
Problem and for a controlled study of learned neighborhood selection on it.
The C-SDVRP extends the split delivery VRP with three operational controls:

- **N1 – controlled split activation**: a customer may be served by more than
  one vehicle only if its demand reaches a capacity-based threshold
  (`d_i >= beta_Q * Q`); smaller orders must be served in a single visit.
- **N2 – receiving capacity**: the number of vehicle visits per customer is
  capped at `U_max`.
- **N3 – extra-contact penalty**: every visit beyond the first is priced in
  the objective. N3 is an objective term, not a constraint.

The repository contains an exact MILP solver, the metaheuristics, the full
benchmark, the trained policies, every per-run result including a feasibility
audit, and the scripts that regenerate every table and figure of the paper.

## Correction relative to the first version of this study

An earlier version of this code enforced N1 and N2 in the exact model but
**not inside the heuristic insertion operators**, so heuristic solutions could
split customers that were not split-eligible. Those solutions were feasible
only for a relaxation of the problem and were therefore not comparable with
the certified optima. Three further defects affected the learned method: the
reward charged a rejected shake its full deterioration, which taught the
policy to avoid large moves; the search-progress feature was computed from an
iteration cap that differed between training and evaluation; and the
benchmark averaged two runs per instance while the paper described it
differently.

All of this is fixed here. The operators enforce capacity, N1, N2 and
one visit per vehicle during insertion; budgets are measured in process CPU
time; every run is audited; and the conclusions changed as a result. Results
produced with the earlier code are in the git history under the tag of the
first release and should not be used.

## Methods

All methods minimise the same objective and start from the same
Clarke–Wright solution.

| Name | Description |
|------|-------------|
| CW | Clarke–Wright savings construction (reference solution) |
| VNS | Variable neighborhood search with RVND local search |
| ALNS+ | Adaptive large neighborhood search, 18 destroy/repair/size actions, RVND, simulated-annealing acceptance |
| DRL-ALNS | ALNS+ with a PPO policy selecting the action |
| DRL-VNS | VNS whose shake action is selected by a PPO policy from a 12-feature search state |

`DRL-VNS` is compared against controls that share its action set, repair
operator, local search and acceptance rule, and differ **only** in how the
action is chosen:

| Control | Selection rule |
|---------|----------------|
| VNS-9 / VNS-18 | the standard VNS schedule over the same actions |
| Fixed-small / Fixed-dominant | always one action |
| Random-9 / Random-18 | uniform random |
| Roulette-9 / Roulette-18 | adaptive weights, no learning |
| Marginal-9 / Marginal-18 | the policy's own action frequencies, ignoring the state |

Two action sets are studied: nine actions (3 removal operators × 3 strengths,
greedy insertion) and eighteen actions, which add regret-2 insertion so that
the set **contains the moves of the VNS baseline**.

## Summary of results

Equal CPU budget, 42 instances, 10 runs per instance and method, five
independently trained policies per learned method, all solutions audited.

Mean improvement over Clarke–Wright (%):

| Method | S | M | L | XL | All |
|--------|---|---|---|----|-----|
| VNS | 10.52 | 14.63 | 12.20 | **12.22** | **12.42** |
| DRL-VNS | 10.47 | **14.89** | **12.21** | 11.56 | 12.38 |
| Roulette-9 | 10.65 | 14.80 | 11.99 | 11.68 | 12.36 |
| Marginal-9 | **10.70** | 14.72 | 11.96 | 11.16 | 12.27 |
| VNS-9 | 10.34 | 14.66 | 11.83 | 11.04 | 12.10 |
| DRL-ALNS | 9.75 | 13.13 | 8.67 | 8.55 | 10.24 |
| ALNS+ | 9.39 | 13.02 | 7.89 | 8.03 | 9.81 |

- **DRL-VNS does not beat standard VNS** (17 wins / 23 losses, Holm-corrected
  *p* = 0.87), and is worse on the largest instances. It clearly beats both
  ALNS variants (*p* < 0.001).
- Within the nine-action set it does beat the cyclic, uniform, fixed and
  state-blind controls, including Marginal-9 (*p* = 0.032; at class L,
  11 wins against 1, *p* = 0.005), so the policy does use the search state.
- With the **eighteen-action set**, which contains the baseline's moves,
  DRL-VNS18 is statistically indistinguishable from *every* control,
  including the state-blind one. The action set matters more than the rule
  that picks among it.
- Exact solver: optimality proven for 17 of 20 small instances; all methods
  except one control are within 0.14% of the proven optima on average.
- Price of control: the three controls remove 93–97% of split customers for
  about 1% additional routing cost; the contact penalty does most of the work
  and the visit cap is rarely binding.
- Feasibility audit: **0 violations in 10,892 solver runs**.

## Repository layout

```
src/                 problem model, solvers, instance generator, exact MILP
  instance.py        instance container and distance matrix
  solution.py        solution representation, objective, feasibility audit
  operators.py       removal and insertion operators (enforce capacity, N1, N2)
  search_common.py   CPU-time budget, shared RVND, destroy-repair step
  clarke_wright.py   savings construction
  vns.py             standard VNS baseline
  alns_plus.py       ALNS+ and DRL-ALNS
  drl_vns.py         the nine- and eighteen-action VNS family and DRL-VNS
  ppo.py             PPO actor-critic agent
  exact_milp.py      exact MILP (PuLP/CBC), single-commodity flow
  gen_benchmark.py   benchmark instance generator
  budgets.py         CPU-time budgets per instance size
train_policy.py      train one policy (--algo vns|alns, --actions 9|18)
run_experiments.py   all heuristic suites (benchmark, exact, case, poc, sens)
run_exact_milp.py    exact solutions of the small instances
run_sdvrp_benchmark.py  the same solvers on the classical SDVRP SET-1
analyze.py           aggregation, Wilcoxon with Holm correction, Friedman
make_tables.py       writes the LaTeX tables of the paper from the raw results
make_figures.py      writes the figures of the paper
tools/               job pool, SET-1 fetcher, highlighted-PDF builder
models_v2/           trained policies, training logs and curves
results_v2/          one row per solver run, with audit columns
figures_v2/          figures used in the paper
tables_v2/           LaTeX tables used in the paper
```

## Reproducing

```
pip install -r requirements.txt

python train_policy.py --algo vns --actions 9  --seed 1     # and seeds 2..5
python train_policy.py --algo vns --actions 18 --seed 1     # and seeds 2..5
python train_policy.py --algo alns --seed 1                 # and seeds 2..5

python run_experiments.py --suites main,exact,case --shard 0 --nshards 6
python run_experiments.py --suites family18 --shard 0 --nshards 5
python run_experiments.py --suites poc,sens,csens --shard 0 --nshards 4
python run_exact_milp.py

python tools/fetch_sdvrp_set1.py      # downloads the public SET-1 instances
python run_sdvrp_benchmark.py

python analyze.py && python make_tables.py && python make_figures.py
```

`tools/run_pool.py` runs the shards in parallel, one per CPU core, pinned and
at below-normal priority. Budgets are process CPU seconds, so other activity
on the machine does not change the results.

Benchmark instances are generated on the fly from `(n, size_class, seed)` in
`src/gen_benchmark.py`, so every instance is reproducible without storing it.

## Third-party data

The classical SDVRP instances (SET-1) and the reference values of the 12th
DIMACS Implementation Challenge are **not redistributed here**.
`tools/fetch_sdvrp_set1.py` downloads them from the
[Alkaid-SDVRP repository](https://github.com/HUST-Smart/Alkaid-SDVRP) (MIT
licence), the software and data archive of Lin, He, Jiang, Ma, Su and Lü,
*AlkaidSD: An Efficient Open-Source Solver for the Split Delivery Vehicle
Routing Problem*, INFORMS Journal on Computing, 2024. The instances are those
of Chen, Golden and Wasil, *Networks* 49(4):318–329, 2007. Distances are
Euclidean rounded to the nearest integer, as in the challenge.

For the case study, hospital locations come from the Ministry of Health Saudi
Arabia interactive map and the NUPCO central warehouse location is publicly
documented; the demands are proxies derived from approximate bed counts and
are not measured data.
