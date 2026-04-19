"""Disjoint LinUCB (Li et al. 2010, Algorithm 1).

Each arm maintains its own ridge-regression statistics :math:`A_a`, :math:`b_a`
with :math:`\\hat{\\theta}_a = A_a^{-1} b_a`. The selection score is
:math:`\\hat{\\theta}_a^\\top x + \\alpha \\sqrt{x^\\top A_a^{-1} x}`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class LinUCB:
    def __init__(
        self,
        num_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        seed: int | None = None,
    ):
        if num_arms < 1:
            raise ValueError("num_arms must be >= 1")
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = float(alpha)
        self.seed = seed
        self._tie_rng = np.random.default_rng(seed)

        d = context_dim
        self._A: list[np.ndarray] = [np.eye(d, dtype=np.float64) for _ in range(num_arms)]
        self._b: list[np.ndarray] = [np.zeros(d, dtype=np.float64) for _ in range(num_arms)]

    def theta(self, arm: int) -> np.ndarray:
        """Least-squares estimate :math:`A_a^{-1} b_a` (for inspection / tests)."""
        return np.linalg.solve(self._A[arm], self._b[arm])

    def choose(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context must have length {self.context_dim}, got {x.shape[0]}"
            )

        scores: list[float] = []
        for a in range(self.num_arms):
            A_inv_x = np.linalg.solve(self._A[a], x)
            exploit = float(self._b[a] @ A_inv_x)
            explore = float(np.sqrt(max(0.0, x @ A_inv_x)))
            scores.append(exploit + self.alpha * explore)
        best = max(scores)
        tol = 1e-9
        best_arms = [a for a, s in enumerate(scores) if abs(s - best) <= tol]
        if len(best_arms) == 1:
            return best_arms[0]
        return int(self._tie_rng.choice(best_arms))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context must have length {self.context_dim}, got {x.shape[0]}"
            )
        if arm < 0 or arm >= self.num_arms:
            raise IndexError(f"arm must be in [0, {self.num_arms - 1}], got {arm}")

        self._A[arm] += np.outer(x, x)
        self._b[arm] += float(reward) * x

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Persist ``A_a``, ``b_a``, and a JSON metadata blob (``np.savez_compressed``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "num_arms": self.num_arms,
            "context_dim": self.context_dim,
            "alpha": self.alpha,
            "seed": self.seed,
        }
        if metadata:
            meta.update(metadata)
        meta_bytes = json.dumps(meta).encode("utf-8")
        np.savez_compressed(
            path,
            A_stack=np.stack(self._A, axis=0),
            b_stack=np.stack(self._b, axis=0),
            meta_json=np.frombuffer(meta_bytes, dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: str | Path) -> LinUCB:
        """Load checkpoint written by :meth:`save`."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        meta = json.loads(data["meta_json"].tobytes().decode("utf-8"))
        inst = cls(
            int(meta["num_arms"]),
            int(meta["context_dim"]),
            float(meta.get("alpha", 1.0)),
            seed=meta.get("seed"),
        )
        A_stack = np.asarray(data["A_stack"])
        b_stack = np.asarray(data["b_stack"])
        inst._A = [A_stack[i].copy() for i in range(inst.num_arms)]
        inst._b = [b_stack[i].copy() for i in range(inst.num_arms)]
        inst._checkpoint_metadata = meta
        return inst


__all__ = ["LinUCB"]
