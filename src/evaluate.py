from __future__ import annotations

from typing import Any

from schedule.penalty import PenaltyResult, compute_penalty
from schedule.representation import DAY_NAMES_FULL, Schedule

# Single-letter roster cells (matches common INRC-II shift names).
_SHIFT_LETTER = {
    "Early": "E",
    "Late": "L",
    "Night": "N",
    "Day": "D",
}


def _shift_cell(shift_type: str, _skill: str) -> str:
    ch = _SHIFT_LETTER.get(shift_type)
    if ch is None and shift_type:
        ch = shift_type.strip()[:1].upper()
    if ch is None:
        ch = "?"
    # Keep one display column; disambiguate Day vs other "D*" types if needed.
    if len(ch) != 1:
        ch = ch[0]
    return ch


def evaluate_schedule(
    schedule: Any,
    scenario: dict[str, Any],
    week_data_list: list[dict[str, Any]],
    history: dict[str, Any],
) -> dict[str, Any]:
    """Small helper to standardize evaluation outputs."""
    result = compute_penalty(schedule, scenario, week_data_list, history)
    # `PenaltyResult.total` is the sum of soft-constraint penalties (S1–S7).
    # Hard constraints are returned separately as a per-constraint dict.
    return {
        "total": result.total,
        "hard": result.hard,
        "hard_total": result.total_hard,
        "soft": result.total,
        "soft_breakdown": {
            "S1_optimal_coverage": result.s1_optimal_coverage,
            "S2_consecutive": result.s2_consecutive,
            "S3_days_off": result.s3_days_off,
            "S4_preferences": result.s4_preferences,
            "S5_complete_weekends": result.s5_complete_weekends,
            "S6_total_assignments": result.s6_total_assignments,
            "S7_working_weekends": result.s7_working_weekends,
        },
    }


def format_schedule_detailed(schedule: Schedule) -> str:
    """Week-by-week text: each cell is "-" or ``ShiftType/skill``."""
    lines: list[str] = []
    lines.append("=== Final Schedule (detailed) ===")
    lines.append("")

    day_hdr = " ".join([d[:3] for d in DAY_NAMES_FULL])  # Mon Tue ...
    for w in range(schedule.num_weeks):
        lines.append(f"Week {w}: {day_hdr}")
        base = w * 7
        for nid in schedule.nurse_ids:
            cells: list[str] = []
            for d in range(7):
                a = schedule.get(nid, base + d)
                if a is None:
                    cells.append("-")
                else:
                    shift_type, skill = a
                    cells.append(f"{shift_type}/{skill}")
            lines.append(f"{nid}: " + " ".join(cells))
        if w != schedule.num_weeks - 1:
            lines.append("")

    return "\n".join(lines)


def format_schedule(schedule: Schedule) -> str:
    """Render a compact ASCII grid: all nurses in ``schedule.nurse_ids``, every week.

    Header row repeats ``|M|T|W|T|F|S|S|`` per week. Each body row is one nurse;
    cells are a single shift letter (E/L/N/D/…) or ``-`` off. Weeks are spaced
    for readability; the name column is padded so the grid lines up.
    """
    lines: list[str] = []
    lines.append("=== Final Schedule ===")
    lines.append("")

    nurse_ids = list(schedule.nurse_ids)
    if not nurse_ids:
        lines.append("(no nurses)")
        return "\n".join(lines)

    name_w = max(len(nid) for nid in nurse_ids)
    prefix_w = name_w + 1  # name column + one space before the grid
    week_hdr = "|" + "|".join(day[0] for day in DAY_NAMES_FULL) + "|"
    grid_hdr = " ".join([week_hdr] * schedule.num_weeks)
    lines.append(" " * prefix_w + grid_hdr)
    lines.append(" " * prefix_w + "-" * len(grid_hdr))

    for nid in nurse_ids:
        row_parts: list[str] = []
        for w in range(schedule.num_weeks):
            base = w * 7
            cells: list[str] = []
            for d in range(7):
                a = schedule.get(nid, base + d)
                if a is None:
                    cells.append("-")
                else:
                    shift_type, skill = a
                    cells.append(_shift_cell(shift_type, skill))
            row_parts.append("|" + "|".join(cells) + "|")
        grid = " ".join(row_parts)
        lines.append(f"{nid:<{name_w}} {grid}")

    return "\n".join(lines)


def format_validator_report(result: PenaltyResult) -> str:
    """Render a validator.jar-like penalty breakdown."""
    hard = result.hard or {}
    lines: list[str] = []
    lines.append("=== INRC-II Penalty Report ===")
    lines.append("")
    lines.append("Hard constraints (violation counts):")
    lines.append(f"  H1 Single assignment/day        : {int(hard.get('H1', 0))}")
    lines.append(f"  H2 Minimum coverage             : {int(hard.get('H2', 0))}")
    lines.append(f"  H3 Forbidden shift successions  : {int(hard.get('H3', 0))}")
    lines.append(f"  H4 Required skills              : {int(hard.get('H4', 0))}")
    lines.append(f"  Hard total                      : {int(result.total_hard)}")
    lines.append("")
    lines.append("Soft constraints (weighted penalties):")
    lines.append(f"  S1 Optimal coverage             : {int(result.s1_optimal_coverage)}")
    lines.append(f"  S2 Consecutive                  : {int(result.s2_consecutive)}")
    lines.append(f"  S3 Consecutive days off         : {int(result.s3_days_off)}")
    lines.append(f"  S4 Preferences                  : {int(result.s4_preferences)}")
    lines.append(f"  S5 Complete weekends            : {int(result.s5_complete_weekends)}")
    lines.append(f"  S6 Total assignments            : {int(result.s6_total_assignments)}")
    lines.append(f"  S7 Working weekends             : {int(result.s7_working_weekends)}")
    lines.append(f"  Soft total                      : {int(result.total)}")
    lines.append("")
    lines.append(f"TOTAL (soft total)                : {int(result.total)}")
    return "\n".join(lines)