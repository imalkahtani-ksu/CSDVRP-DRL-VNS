"""
gen_exact_instance.py — small N1-active instances for the exact-MILP
optimality-gap study. Uses the SAME demand/constraint model as the main
benchmark (src.gen_benchmark) so exact and large-scale results are
directly comparable; only the size class (capacity Q) is set small to keep
the MILP tractable.
"""
from .gen_benchmark import make_instance


def make_exact_instance(n: int, seed: int, size_class: str = "S",
                        **kwargs) -> "CSDVRPInstance":
    """Small homogeneous, N1-active instance (defaults to S class, Q=15)."""
    inst = make_instance(n, size_class=size_class, seed=seed, **kwargs)
    inst.params["instance"] = f"exact-n{n}-s{seed}"
    return inst
