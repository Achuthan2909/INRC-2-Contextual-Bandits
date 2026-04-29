"""
Simulator wrapper — note and Python stand-in.

What tools/Simulator_withTimeout.jar does
------------------------------------------
The official INRC-2 simulator is a Java program that orchestrates the
full multi-week scheduling process:

  1. Reads the scenario (Sc-*.json) and initial history (H0-*.json)
  2. Calls your scheduling program for Week 1, passing scenario + history
     + week demand (WD-*.json). Your program writes a solution file.
  3. Updates nurse history from the Week 1 solution (consecutive counters,
     last shift worked, weekends worked, etc.)
  4. Calls your scheduling program again for Week 2 with the updated history
  5. Repeats for all weeks
  6. At the end, calls tools/validator.jar to compute the official P_total

In production, your scheduling program would be called by the simulator as
a subprocess and would read/write JSON solution files in the INRC-2 format.

How to run it (command line)
-----------------------------
  java -jar tools/Simulator_withTimeout.jar
       --scenario  Dataset/datasets_json/n030w4/Sc-n030w4.json
       --history   Dataset/datasets_json/n030w4/H0-n030w4-0.json
       --weeks     Dataset/datasets_json/n030w4/WD-n030w4-0.json \
                   Dataset/datasets_json/n030w4/WD-n030w4-1.json \
                   Dataset/datasets_json/n030w4/WD-n030w4-2.json \
                   Dataset/datasets_json/n030w4/WD-n030w4-3.json
       --program   "python src/main.py"

Python stand-in used in this sample run
-----------------------------------------
run_sample.py uses a pure-Python week loop instead of the Java simulator.
After each week, it manually updates the history dict from the assignments
produced by the chosen strategy. This is a simplified approximation —
it updates `lastAssignedShiftType` and `numberOfAssignments` per nurse
but does not recompute all consecutive-day counters exactly as the
official simulator would.

For official results, replace the Python loop in run_sample.py with
a subprocess call to the Java simulator.
"""
from __future__ import annotations

from typing import Any


def update_history_from_assignments(
    history: dict[str, Any],
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Lightweight history update after one week.

    Updates per nurse:
      - lastAssignedShiftType : last shift worked this week (or None if no shift)
      - numberOfAssignments   : incremented by shifts worked this week

    Does NOT recompute numberOfConsecutiveWorkingDays / numberOfConsecutiveDaysOff
    exactly — the Java simulator handles that precisely.  This approximation
    is sufficient for the bandit's context vector in the sample run.

    Returns a new history dict (does not mutate the original).
    """
    import copy
    new_history = copy.deepcopy(history)

    # Index nurse history records by nurse id
    hist_by_nurse: dict[str, dict[str, Any]] = {
        rec["nurse"]: rec for rec in new_history["nurseHistory"]
    }

    # Count assignments per nurse this week and track last shift
    DAY_ORDER = ["Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday", "Sunday"]
    day_rank = {d: i for i, d in enumerate(DAY_ORDER)}

    shifts_this_week: dict[str, list[tuple[int, str]]] = {}
    for a in assignments:
        nid = a["nurseId"]
        if nid not in shifts_this_week:
            shifts_this_week[nid] = []
        shifts_this_week[nid].append((day_rank[a["day"]], a["shiftType"]))

    for nid, shifts in shifts_this_week.items():
        if nid not in hist_by_nurse:
            continue
        rec = hist_by_nurse[nid]
        # Last shift = shift on the latest day worked
        last_shift = max(shifts, key=lambda x: x[0])[1]
        rec["lastAssignedShiftType"]  = last_shift
        rec["numberOfAssignments"]   += len(shifts)

    return new_history
