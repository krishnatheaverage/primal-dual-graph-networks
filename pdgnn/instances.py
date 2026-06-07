"""Instance generators for weighted minimum vertex cover.

Two random graph families, matching Section 6.1 of the paper:
  * Erdos-Renyi  G(n, p) with p = c/n  (constant average degree c)
  * Barabasi-Albert preferential attachment with attachment parameter m

Node weights are drawn i.i.d. from Uniform[0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class Instance:
    """A single weighted graph instance.

    Attributes
    ----------
    n        : number of nodes
    edges    : int array of shape [m, 2], each row (u, v) with u < v, deduplicated
    weights  : float array of shape [n], node weights in [0, 1]
    family   : 'ER' or 'BA'
    param    : c (target average degree) for ER, m (attachment) for BA
    seed     : RNG seed used to build the instance
    """

    n: int
    edges: np.ndarray
    weights: np.ndarray
    family: str
    param: float
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    @property
    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.n, dtype=np.int64)
        if self.m:
            np.add.at(deg, self.edges[:, 0], 1)
            np.add.at(deg, self.edges[:, 1], 1)
        return deg

    @property
    def avg_degree(self) -> float:
        return 2.0 * self.m / self.n if self.n else 0.0


def _edges_from_nx(g: nx.Graph) -> np.ndarray:
    edges = [tuple(sorted(e)) for e in g.edges() if e[0] != e[1]]
    edges = sorted(set(edges))
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def make_er(n: int, c: float, seed: int) -> Instance:
    """Erdos-Renyi G(n, p) with p = c / n and Uniform[0,1] node weights."""
    rng = np.random.default_rng(seed)
    p = min(1.0, c / max(1, n))  # p = c/n, as in Section 6.1
    g = nx.gnp_random_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
    weights = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    return Instance(n=n, edges=_edges_from_nx(g), weights=weights,
                    family="ER", param=float(c), seed=seed)


def make_ba(n: int, m: int, seed: int) -> Instance:
    """Barabasi-Albert preferential attachment with Uniform[0,1] node weights."""
    rng = np.random.default_rng(seed)
    m = max(1, int(m))
    g = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31 - 1)))
    weights = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    return Instance(n=n, edges=_edges_from_nx(g), weights=weights,
                    family="BA", param=float(m), seed=seed)


def make_instance(family: str, n: int, param: float, seed: int) -> Instance:
    if family.upper() == "ER":
        return make_er(n, param, seed)
    if family.upper() == "BA":
        return make_ba(n, int(param), seed)
    raise ValueError(f"unknown family {family!r} (expected 'ER' or 'BA')")


def make_dataset(family: str, param: float, n_lo: int, n_hi: int,
                 count: int, base_seed: int) -> list[Instance]:
    """A list of `count` instances with node counts sampled uniformly in [n_lo, n_hi].

    Distinct seeds per instance keep graphs and weights independent and
    reproducible from `base_seed`.
    """
    rng = np.random.default_rng(base_seed)
    out: list[Instance] = []
    for i in range(count):
        n = int(rng.integers(n_lo, n_hi + 1))
        seed = int(rng.integers(0, 2**31 - 1))
        out.append(make_instance(family, n, param, seed))
    return out
