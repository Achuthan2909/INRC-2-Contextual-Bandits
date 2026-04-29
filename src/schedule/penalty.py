"""INRC-II penalty calculator — replicates validator.jar scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .representation import Schedule, REQ_DAY_KEYS, DAY_NAMES_FULL

# Constraint weights
W_S1 = 30   # Optimal coverage
W_S2_SHIFT = 15   # Consecutive same-shift assignments
W_S2_WORK = 30    # Consecutive working days
W_S3 = 30   # Consecutive days off
W_S4 = 10   # Preferences
W_S5 = 30   # Complete weekends
W_S6 = 20   # Total assignments
W_S7 = 30   # Total working weekends


@dataclass
class PenaltyResult:
    s1_optimal_coverage: int = 0
    s2_consecutive: int = 0
    s3_days_off: int = 0
    s4_preferences: int = 0
    s5_complete_weekends: int = 0
    s6_total_assignments: int = 0
    s7_working_weekends: int = 0
    hard: dict[str, int] = field(
        default_factory=lambda: {"H1": 0, "H2": 0, "H3": 0, "H4": 0}
    )

    @property
    def total(self) -> int:
        return (
            self.s1_optimal_coverage
            + self.s2_consecutive
            + self.s3_days_off
            + self.s4_preferences
            + self.s5_complete_weekends
            + self.s6_total_assignments
            + self.s7_working_weekends
        )

    @property
    def total_hard(self) -> int:
        return sum(self.hard.values())


def compute_penalty(
    schedule: Schedule,
    scenario: dict[str, Any],
    week_data_list: list[dict[str, Any]],
    initial_history: dict[str, Any],
) -> PenaltyResult:
    result = PenaltyResult()
    # Hard constraints first (H1–H4), then soft constraints (S1–S7).
    result.hard = _compute_hard(schedule, scenario, week_data_list, initial_history)
    result.s1_optimal_coverage = _compute_s1(schedule, week_data_list)
    result.s2_consecutive = _compute_s2(schedule, scenario, initial_history)
    result.s3_days_off = _compute_s3(schedule, scenario, initial_history)
    result.s4_preferences = _compute_s4(schedule, week_data_list)
    result.s5_complete_weekends = _compute_s5(schedule, scenario)
    result.s6_total_assignments = _compute_s6(schedule, scenario, initial_history)
    result.s7_working_weekends = _compute_s7(schedule, scenario, initial_history)
    return result


# ---------------------------------------------------------------------------
# Hard constraints H1–H4
# ---------------------------------------------------------------------------

def _compute_hard(
    schedule: Schedule,
    scenario: dict[str, Any],
    week_data_list: list[dict[str, Any]],
    initial_history: dict[str, Any],
) -> dict[str, int]:
    return {
        "H1": _compute_h1(schedule, scenario),
        "H2": _compute_h2(schedule, week_data_list),
        "H3": _compute_h3(schedule, scenario, initial_history),
        "H4": _compute_h4(schedule, scenario),
    }


def _compute_h1(schedule: Schedule, scenario: dict) -> int:
    """H1: Single assignment per day — each nurse works at most one shift."""
    # Schedule representation already enforces single assignment per
    # (nurse, day) key, so H1 violations are always 0 in our model.
    # Kept for completeness / if the representation ever changes.
    return 0


def _compute_h2(schedule: Schedule, week_data_list: list[dict]) -> int:
    """H2: Minimum coverage — each (day, shift, skill) must meet minimum."""
    total = 0
    for week_idx, wd in enumerate(week_data_list):
        for req in wd["requirements"]:
            shift_type = req["shiftType"]
            skill = req["skill"]
            for day_in_week, day_key in enumerate(REQ_DAY_KEYS):
                minimum = req[day_key]["minimum"]
                if minimum == 0:
                    continue
                count = schedule.coverage(
                    week_idx * 7 + day_in_week, shift_type, skill
                )
                total += max(minimum - count, 0)
    return total


def _compute_h3(
    schedule: Schedule, scenario: dict, initial_history: dict,
) -> int:
    """H3: Forbidden shift type successions.

    If nurse works shift s1 on day d, they cannot work a shift in
    s1's forbidden successor list on day d+1.  Also applies across
    the history boundary (last shift of previous horizon -> day 0).
    """
    # Build lookup: preceding_shift -> set of forbidden successors
    forbidden: dict[str, set[str]] = {}
    for entry in scenario["forbiddenShiftTypeSuccessions"]:
        succ = entry["succeedingShiftTypes"]
        if succ:
            forbidden[entry["precedingShiftType"]] = set(succ)

    if not forbidden:
        return 0

    hist_map = {h["nurse"]: h for h in initial_history["nurseHistory"]}
    total = 0

    for nurse in scenario["nurses"]:
        nid = nurse["id"]
        # Check history -> day 0 boundary
        prev_shift = hist_map[nid]["lastAssignedShiftType"]
        if prev_shift not in (None, "None") and prev_shift in forbidden:
            day0_shift = schedule.shift(nid, 0)
            if day0_shift in forbidden[prev_shift]:
                total += 1

        # Check consecutive days within the schedule
        for d in range(schedule.num_days - 1):
            s1 = schedule.shift(nid, d)
            if s1 is not None and s1 in forbidden:
                s2 = schedule.shift(nid, d + 1)
                if s2 in forbidden[s1]:
                    total += 1

    return total


def _compute_h4(schedule: Schedule, scenario: dict) -> int:
    """H4: Required skill — nurse must possess the skill they're assigned."""
    nurse_skills: dict[str, set[str]] = {
        n["id"]: set(n["skills"]) for n in scenario["nurses"]
    }
    total = 0
    for nurse in scenario["nurses"]:
        nid = nurse["id"]
        skills = nurse_skills[nid]
        for d in range(schedule.num_days):
            assignment = schedule.get(nid, d)
            if assignment is not None:
                _, skill = assignment
                if skill not in skills:
                    total += 1
    return total


# ---------------------------------------------------------------------------
# S1: Optimal coverage
# ---------------------------------------------------------------------------

def _compute_s1(schedule: Schedule, week_data_list: list[dict]) -> int:
    """Each missing nurse below *optimal* coverage costs W_S1 = 30."""
    total = 0
    for week_idx, wd in enumerate(week_data_list):
        for req in wd["requirements"]:
            shift_type = req["shiftType"]
            skill = req["skill"]
            for day_in_week, day_key in enumerate(REQ_DAY_KEYS):
                optimal = req[day_key]["optimal"]
                if optimal == 0:
                    continue
                count = schedule.coverage(
                    week_idx * 7 + day_in_week, shift_type, skill
                )
                total += max(optimal - count, 0) * W_S1
    return total


# ---------------------------------------------------------------------------
# S4: Preferences (shift-off / day-off requests)
# ---------------------------------------------------------------------------

def _compute_s4(
    schedule: Schedule, week_data_list: list[dict],
) -> int:
    """Each violated shift-off or day-off request costs W_S4 = 10."""
    total = 0
    for week_idx, wd in enumerate(week_data_list):
        for req in wd.get("shiftOffRequests", []):
            nurse_id = req["nurse"]
            day_name = req["day"]            # "Monday", "Tuesday", ...
            shift_type = req["shiftType"]    # specific shift or "Any"
            day_in_week = DAY_NAMES_FULL.index(day_name)
            gd = week_idx * 7 + day_in_week

            if shift_type == "Any":
                # day-off request: violated if nurse works any shift
                if schedule.is_working(nurse_id, gd):
                    total += W_S4
            else:
                # shift-off request: violated if nurse works that specific shift
                if schedule.shift(nurse_id, gd) == shift_type:
                    total += W_S4
    return total


# ---------------------------------------------------------------------------
# S5: Complete weekends
# ---------------------------------------------------------------------------

def _compute_s5(schedule: Schedule, scenario: dict) -> int:
    """If contract requires complete weekends, working exactly one of
    Sat/Sun costs W_S5 = 30."""
    contract_map = {c["id"]: c for c in scenario["contracts"]}
    total = 0
    for nurse in scenario["nurses"]:
        contract = contract_map[nurse["contract"]]
        if not contract.get("completeWeekends", 0):
            continue
        nid = nurse["id"]
        for w in range(schedule.num_weeks):
            sat = w * 7 + 5
            sun = w * 7 + 6
            works_sat = schedule.is_working(nid, sat)
            works_sun = schedule.is_working(nid, sun)
            if works_sat != works_sun:
                total += W_S5
    return total


# ---------------------------------------------------------------------------
# S6: Total assignments (horizon-end only)
# ---------------------------------------------------------------------------

def _compute_s6(
    schedule: Schedule, scenario: dict, initial_history: dict,
) -> int:
    """Penalty for total assignments outside [min, max] over full horizon.
    Weight = 20 per deviation."""
    contract_map = {c["id"]: c for c in scenario["contracts"]}
    hist_by_nurse = {h["nurse"]: h for h in initial_history["nurseHistory"]}
    total = 0
    for nurse in scenario["nurses"]:
        nid = nurse["id"]
        contract = contract_map[nurse["contract"]]
        lo = contract["minimumNumberOfAssignments"]
        hi = contract["maximumNumberOfAssignments"]
        prior = hist_by_nurse[nid]["numberOfAssignments"]
        assigned = sum(
            1 for d in range(schedule.num_days) if schedule.is_working(nid, d)
        )
        total_assigned = prior + assigned
        total += max(total_assigned - hi, 0) * W_S6
        total += max(lo - total_assigned, 0) * W_S6
    return total


# ---------------------------------------------------------------------------
# S7: Total working weekends (horizon-end only)
# ---------------------------------------------------------------------------

def _compute_s7(
    schedule: Schedule, scenario: dict, initial_history: dict,
) -> int:
    """Penalty for exceeding maximum working weekends over full horizon.
    Weight = 30 per excess weekend."""
    contract_map = {c["id"]: c for c in scenario["contracts"]}
    hist_by_nurse = {h["nurse"]: h for h in initial_history["nurseHistory"]}
    total = 0
    for nurse in scenario["nurses"]:
        nid = nurse["id"]
        contract = contract_map[nurse["contract"]]
        max_we = contract["maximumNumberOfWorkingWeekends"]
        prior = hist_by_nurse[nid]["numberOfWorkingWeekends"]
        weekends_worked = sum(
            1
            for w in range(schedule.num_weeks)
            if schedule.is_working(nid, w * 7 + 5)
            or schedule.is_working(nid, w * 7 + 6)
        )
        total_we = prior + weekends_worked
        total += max(total_we - max_we, 0) * W_S7
    return total


# ===================================================================
# Helper: generic consecutive-run violation counting
# ===================================================================

def _max_consec_violations(
    active_seq: list[bool], c_hist: int, max_limit: int,
) -> int:
    """Count violation-days for exceeding *max_limit* consecutive active days.

    Border handling (Appendix B, Tables 4-6):
      - Leading run from history: only count NEW extra violations.
      - Trailing run at end of week: count full violations.
    """
    violations = 0
    running = c_hist
    already = max(c_hist - max_limit, 0)

    for active in active_seq:
        if active:
            running += 1
        else:
            if running > 0:
                violations += max(running - max_limit, 0) - already
                already = 0
            running = 0

    # trailing run at end of week
    if running > 0:
        violations += max(running - max_limit, 0) - already

    return violations


def _min_consec_violations(
    active_seq: list[bool], c_hist: int, min_limit: int,
) -> int:
    """Count violation-days for falling below *min_limit* consecutive active days.

    Border handling (Appendix B, Tables 7-8):
      - Leading run from history: penalised when it ends within the week.
      - Trailing run at end of week: NOT penalised (deferred to next stage).
    """
    violations = 0
    running = c_hist

    for active in active_seq:
        if active:
            running += 1
        else:
            if running > 0 and running < min_limit:
                violations += min_limit - running
            running = 0

    # trailing run — deferred, no penalty
    return violations


# ===================================================================
# History update between weeks
# ===================================================================

def _compute_week_history(
    schedule: Schedule, week_idx: int, prev_history: dict, scenario: dict,
) -> dict:
    """Derive the history state after *week_idx* completes."""
    prev_map = {h["nurse"]: h for h in prev_history["nurseHistory"]}
    nurse_histories = []

    for nurse in scenario["nurses"]:
        nid = nurse["id"]
        ph = prev_map[nid]
        base = week_idx * 7

        # accumulate counters
        week_assignments = sum(
            1 for d in range(7) if schedule.is_working(nid, base + d)
        )
        worked_we = (
            schedule.is_working(nid, base + 5)
            or schedule.is_working(nid, base + 6)
        )

        # border data: walk backwards from Sunday
        sun_shift = schedule.shift(nid, base + 6)

        if sun_shift is not None:
            # nurse worked Sunday
            consec_shift = 0
            for d in range(6, -1, -1):
                if schedule.shift(nid, base + d) == sun_shift:
                    consec_shift += 1
                else:
                    break
            consec_work = 0
            for d in range(6, -1, -1):
                if schedule.is_working(nid, base + d):
                    consec_work += 1
                else:
                    break
            # extend through history if entire week was same
            if consec_work == 7:
                consec_work += ph["numberOfConsecutiveWorkingDays"]
            if consec_shift == 7 and ph["lastAssignedShiftType"] == sun_shift:
                consec_shift += ph["numberOfConsecutiveAssignments"]
            consec_off = 0
        else:
            # nurse rested Sunday
            consec_off = 0
            for d in range(6, -1, -1):
                if not schedule.is_working(nid, base + d):
                    consec_off += 1
                else:
                    break
            if consec_off == 7:
                consec_off += ph["numberOfConsecutiveDaysOff"]
            consec_shift = 0
            consec_work = 0
            sun_shift = "None"

        nurse_histories.append({
            "nurse": nid,
            "numberOfAssignments": ph["numberOfAssignments"] + week_assignments,
            "numberOfWorkingWeekends": (
                ph["numberOfWorkingWeekends"] + (1 if worked_we else 0)
            ),
            "lastAssignedShiftType": sun_shift,
            "numberOfConsecutiveAssignments": consec_shift,
            "numberOfConsecutiveWorkingDays": consec_work,
            "numberOfConsecutiveDaysOff": consec_off,
        })

    return {
        "week": week_idx + 1,
        "scenario": prev_history.get("scenario", ""),
        "nurseHistory": nurse_histories,
    }


# ===================================================================
# S2: Consecutive assignments  (weight 15 per-shift, 30 working days)
# ===================================================================

def _compute_s2(
    schedule: Schedule, scenario: dict, initial_history: dict,
) -> int:
    contract_map = {c["id"]: c for c in scenario["contracts"]}
    shift_type_map = {s["id"]: s for s in scenario["shiftTypes"]}

    total = 0
    history = initial_history

    for week_idx in range(schedule.num_weeks):
        hist_map = {h["nurse"]: h for h in history["nurseHistory"]}
        base = week_idx * 7

        for nurse in scenario["nurses"]:
            nid = nurse["id"]
            contract = contract_map[nurse["contract"]]
            nh = hist_map[nid]

            work_seq = [schedule.is_working(nid, base + d) for d in range(7)]
            shift_seq = [schedule.shift(nid, base + d) for d in range(7)]

            # --- consecutive working days ---
            c_work = nh["numberOfConsecutiveWorkingDays"]
            total += _max_consec_violations(
                work_seq, c_work,
                contract["maximumNumberOfConsecutiveWorkingDays"],
            ) * W_S2_WORK
            total += _min_consec_violations(
                work_seq, c_work,
                contract["minimumNumberOfConsecutiveWorkingDays"],
            ) * W_S2_WORK

            # --- consecutive same-shift assignments ---
            total += _shift_consec_violations(
                shift_seq, nh["lastAssignedShiftType"],
                nh["numberOfConsecutiveAssignments"], shift_type_map,
            ) * W_S2_SHIFT

        # advance history
        if week_idx < schedule.num_weeks - 1:
            history = _compute_week_history(
                schedule, week_idx, history, scenario,
            )

    return total


def _shift_consec_violations(
    shift_seq: list[str | None],
    hist_shift: str,
    hist_c: int,
    shift_type_map: dict[str, dict],
) -> int:
    """Return total violation-days for consecutive same-shift constraints.

    Walks the 7-day shift sequence, tracking each maximal run of one shift
    type and applying the same border-data logic as the working-day helpers.
    """
    violations = 0

    if hist_shift not in (None, "None") and hist_shift in shift_type_map:
        cur = hist_shift
        running = hist_c
        already = max(
            hist_c
            - shift_type_map[cur]["maximumNumberOfConsecutiveAssignments"],
            0,
        )
    else:
        cur = None
        running = 0
        already = 0

    for d in range(7):
        s = shift_seq[d]
        if s is not None and s == cur:
            running += 1
        else:
            # previous shift-run ended
            if cur is not None and running > 0:
                st = shift_type_map[cur]
                mx = st["maximumNumberOfConsecutiveAssignments"]
                mn = st["minimumNumberOfConsecutiveAssignments"]
                violations += max(running - mx, 0) - already
                violations += max(mn - running, 0)

            # start new run
            if s is not None:
                cur = s
                running = 1
                already = 0
            else:
                cur = None
                running = 0
                already = 0

    # trailing run at end of week: max penalised, min deferred
    if cur is not None and running > 0:
        st = shift_type_map[cur]
        mx = st["maximumNumberOfConsecutiveAssignments"]
        violations += max(running - mx, 0) - already

    return violations


# ===================================================================
# S3: Consecutive days off  (weight 30)
# ===================================================================

def _compute_s3(
    schedule: Schedule, scenario: dict, initial_history: dict,
) -> int:
    contract_map = {c["id"]: c for c in scenario["contracts"]}

    total = 0
    history = initial_history

    for week_idx in range(schedule.num_weeks):
        hist_map = {h["nurse"]: h for h in history["nurseHistory"]}
        base = week_idx * 7

        for nurse in scenario["nurses"]:
            nid = nurse["id"]
            contract = contract_map[nurse["contract"]]
            nh = hist_map[nid]

            off_seq = [not schedule.is_working(nid, base + d) for d in range(7)]
            c_off = nh["numberOfConsecutiveDaysOff"]

            total += _max_consec_violations(
                off_seq, c_off,
                contract["maximumNumberOfConsecutiveDaysOff"],
            ) * W_S3
            total += _min_consec_violations(
                off_seq, c_off,
                contract["minimumNumberOfConsecutiveDaysOff"],
            ) * W_S3

        if week_idx < schedule.num_weeks - 1:
            history = _compute_week_history(
                schedule, week_idx, history, scenario,
            )

    return total
