"""CPU-time budgets (seconds) per instance size, shared by all experiments."""

BUDGET = {15: 5, 20: 6, 30: 10, 50: 15, 75: 25, 100: 35, 150: 50}

# small instances of the exact-solver study
EXACT_BUDGET = 5

# 30-hospital case study
CASE_BUDGET = 30


def budget_for(n: int) -> float:
    if n in BUDGET:
        return BUDGET[n]
    if n <= 12:
        return EXACT_BUDGET
    keys = sorted(BUDGET)
    for a, b in zip(keys, keys[1:]):
        if a < n < b:
            return BUDGET[a] + (BUDGET[b] - BUDGET[a]) * (n - a) / (b - a)
    return BUDGET[keys[-1]]
