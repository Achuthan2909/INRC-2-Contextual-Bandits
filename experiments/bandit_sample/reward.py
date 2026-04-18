"""
Proxy reward function for the bandit sample run.

The official penalty comes from tools/validator.jar (Java). This module
computes a Python proxy from the assignments and uncovered list returned
by each strategy, so the bandit can learn without needing a Java subprocess.

Proxy formula (based on ref_docs/INRC2_Project_Instructions.md):
  r = -(lambda_cov  * p_cov
        + lambda_pref * p_pref)

  p_cov  — total nurse-shifts short across all uncovered slots
  p_pref — total shift-off request violations in the assignments

Consecutive-work and forbidden-succession penalties (p_consec, p_succ from
the ref doc) are not computed here because they require full week-state
tracking; add them when integrating the Java validator.

Higher reward = better schedule (less penalty).
"""
from __future__ import annotations

from typing import Any


def compute_proxy_reward(
    assignments: list[dict[str, Any]],
    uncovered: list[dict[str, Any]],
    week_data: dict[str, Any],
    lambda_cov: float  = 30.0,
    lambda_pref: float = 15.0,
) -> float:
    """
    Parameters
    ----------
    assignments : list of {nurseId, day, shiftType, skill} from a strategy
    uncovered   : list of {day, shiftType, skill, required, assigned, shortage}
    week_data   : weekly demand JSON (used to read shiftOffRequests)
    lambda_cov  : penalty weight per uncovered shift
    lambda_pref : penalty weight per violated shift-off request

    Returns
    -------
    Negative penalty (higher is better).
    """
    # Coverage penalty: sum of shortages
    p_cov = sum(row["shortage"] for row in uncovered)

    # Preference penalty: count assignments that violate a shift-off request
    # Request types:
    #   shiftType == "Any"        → nurse wanted any day off (+10 per violation)
    #   shiftType == <specific>   → nurse wanted that specific shift off (+15)
    requests_any      = set()   # (nurseId, day)
    requests_specific = set()   # (nurseId, day, shiftType)

    for req in week_data.get("shiftOffRequests", []):
        nurse = req["nurse"]
        day   = req["day"]
        st    = req["shiftType"]
        if st == "Any":
            requests_any.add((nurse, day))
        else:
            requests_specific.add((nurse, day, st))

    p_pref = 0
    for a in assignments:
        nid, day, st = a["nurseId"], a["day"], a["shiftType"]
        if (nid, day, st) in requests_specific:
            p_pref += 15
        elif (nid, day) in requests_any:
            p_pref += 10

    return -(lambda_cov * p_cov + lambda_pref * p_pref)
