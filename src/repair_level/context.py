"""Repair-level context builder.

Produces a 7-dim feature vector describing the *current* schedule state at a
given repair round. Designed for disjoint LinUCB over repair strategies.

Features (all in roughly [0, 1]):
  0. s1_share        — optimal coverage penalty share of total soft penalty
  1. s2_share        — consecutive-shift/work share
  2. s3_share        — consecutive days-off share
  3. s4_share        — preference share
  4. s5_s7_share     — weekend penalties (complete + working weekends) combined
  5. s6_share        — total-assignments share
  6. progress_ratio  — round_idx / max(total_rounds - 1, 1)

A final constant-1 bias term is *not* appended — LinUCB already fits an
intercept implicitly via the ridge prior A = I.
"""
from __future__ import annotations

import numpy as np

from schedule.penalty import PenaltyResult


FEATURE_LABELS: tuple[str, ...] = (
    "s1_share",
    "s2_share",
    "s3_share",
    "s4_share",
    "s5_s7_share",
    "s6_share",
    "progress_ratio",
)


def build_repair_context(
    penalty: PenaltyResult,
    round_idx: int,
    total_rounds: int,
) -> np.ndarray:
    """Build a 7-dim context from a pre-computed ``PenaltyResult``.

    Using the already-computed ``PenaltyResult`` avoids a redundant pass over
    the schedule — the runner already has it in scope.
    """
    total = max(penalty.total, 1)  # avoid div-by-zero at zero-penalty state
    s57 = penalty.s5_complete_weekends + penalty.s7_working_weekends
    progress = round_idx / max(total_rounds - 1, 1)
    return np.array(
        [
            penalty.s1_optimal_coverage / total,
            penalty.s2_consecutive / total,
            penalty.s3_days_off / total,
            penalty.s4_preferences / total,
            s57 / total,
            penalty.s6_total_assignments / total,
            float(progress),
        ],
        dtype=np.float64,
    )


__all__ = ["FEATURE_LABELS", "build_repair_context"]
