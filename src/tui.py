"""Curses-based terminal UI for repair-level experiments.

Run:
    python src/tui.py

Keys:
    - Up/Down: move
    - Enter: edit/select
    - Space: toggle (for validator report)
    - r: run
    - q: quit
"""

from __future__ import annotations

import curses
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow running directly from repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bandit import available as available_bandits, get_bandit  # noqa: E402
from evaluate import format_schedule, format_validator_report  # noqa: E402
from repair_level.init import generate_initial_schedule  # noqa: E402
from repair_level.repairs import build_all_strategies  # noqa: E402
from repair_level.runner import RunResult, run_repairs  # noqa: E402
from instance_loader import load_instance  # noqa: E402
from schedule.representation import Schedule  # noqa: E402


@dataclass
class Config:
    dataset_root: str = "Dataset/datasets_json"
    dataset: str = "n030w4"
    history_idx: int = 0
    weeks: str = "0 1 2 3"
    bandit: str = "ucb1"
    rounds: int = 500
    seed: int = 0
    output_dir: str = "runs"
    init_source: str = "greedy"  # greedy | week_level
    run_repairs: bool = True  # False: init + score only (0 repair rounds)
    # Week-level mode: how arms are chosen when init_source == "week_level".
    # coverage_only: single CoverageFirstArm (original behaviour).
    # all_first:     all 4 arms, first_arm_selector picks arms[0] every week.
    # all_random:    all 4 arms, uniformly random pick each week.
    # all_linucb:    all 4 arms, fresh LinUCB learns online over this horizon.
    week_mode: str = "coverage_only"
    linucb_alpha: float = 1.0
    final_validator: bool = False
    profile_timings: bool = False
    lazy_find_violations: bool = False


def _list_datasets(dataset_root: str) -> list[str]:
    root = Path(dataset_root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _draw(stdscr: curses.window, cfg: Config, field_idx: int, msg: str) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    title = "INRC-II Repair Runner (TUI)  |  r=run  Enter=edit  q=quit"
    stdscr.addnstr(0, 0, title, w - 1, curses.A_BOLD)

    fields = [
        ("Dataset root", cfg.dataset_root),
        ("Dataset", cfg.dataset),
        ("History idx", str(cfg.history_idx)),
        ("Weeks", cfg.weeks),
        ("Init source", cfg.init_source),
        ("Week mode", cfg.week_mode),
        ("LinUCB alpha", f"{cfg.linucb_alpha:g}"),
        ("Run repairs", "ON" if cfg.run_repairs else "OFF"),
        ("Bandit", cfg.bandit),
        ("Rounds", str(cfg.rounds)),
        ("Seed", str(cfg.seed)),
        ("Output dir", cfg.output_dir),
        ("Final validator report", "ON" if cfg.final_validator else "OFF"),
        ("Profile timings", "ON" if cfg.profile_timings else "OFF"),
        ("Lazy find_violations", "ON" if cfg.lazy_find_violations else "OFF"),
    ]

    y = 2
    for i, (k, v) in enumerate(fields):
        attr = curses.A_REVERSE if i == field_idx else curses.A_NORMAL
        line = f"{k:22s} : {v}"
        stdscr.addnstr(y + i, 2, line, w - 4, attr)

    help_line = "Tip: Dataset uses Enter for a list; Space toggles ON/OFF fields; r runs."
    stdscr.addnstr(h - 3, 0, help_line, w - 1, curses.A_DIM)
    stdscr.addnstr(h - 2, 0, msg, w - 1, curses.A_DIM)
    stdscr.addnstr(h - 1, 0, "Status: ready", w - 1, curses.A_DIM)
    stdscr.refresh()


def _prompt_line(stdscr: curses.window, prompt: str, default: str) -> str:
    h, w = stdscr.getmaxyx()
    curses.echo()
    try:
        stdscr.move(h - 1, 0)
        stdscr.clrtoeol()
        stdscr.addnstr(h - 1, 0, f"{prompt} [{default}]: ", w - 1)
        stdscr.refresh()
        raw = stdscr.getstr(h - 1, min(len(prompt) + len(default) + 5, w - 2)).decode("utf-8")
        raw = raw.strip()
        return raw or default
    finally:
        curses.noecho()


def _choose_from_list(stdscr: curses.window, title: str, options: list[str], current: str) -> str:
    if not options:
        return current
    idx = options.index(current) if current in options else 0
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addnstr(0, 0, title, w - 1, curses.A_BOLD)
        stdscr.addnstr(1, 0, "Enter=select  q=cancel", w - 1, curses.A_DIM)

        top = max(0, idx - (h - 4) // 2)
        view = options[top : top + (h - 3)]
        for i, opt in enumerate(view):
            y = 2 + i
            a = curses.A_REVERSE if (top + i) == idx else curses.A_NORMAL
            stdscr.addnstr(y, 2, opt, w - 4, a)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord("q"), 27):  # ESC
            return current
        if ch in (curses.KEY_UP, ord("k")):
            idx = max(0, idx - 1)
        elif ch in (curses.KEY_DOWN, ord("j")):
            idx = min(len(options) - 1, idx + 1)
        elif ch in (curses.KEY_ENTER, 10, 13):
            return options[idx]


def _parse_weeks(weeks_str: str) -> list[int]:
    parts = [p for p in weeks_str.split() if p]
    return [int(p) for p in parts]


def _format_timings(out: RunResult) -> str | None:
    t = out.timings_s
    if not isinstance(t, dict):
        return None
    scan = float(t.get("scan_candidates_total", 0.0))
    pen = float(t.get("compute_penalty_total", 0.0)) + float(t.get("compute_penalty_initial", 0.0))
    apply = float(t.get("apply_total", 0.0))
    sel = float(t.get("selector_total", 0.0))
    lines: list[str] = []
    lines.append("=== Timing breakdown (seconds) ===")
    lines.append(f"scan candidates (find_violations): {scan:.3f}")
    lines.append(f"compute_penalty (incl initial):   {pen:.3f}")
    lines.append(f"apply repairs:                    {apply:.3f}")
    lines.append(f"selector overhead:                {sel:.3f}")
    per = t.get("find_violations_by_strategy")
    if isinstance(per, dict):
        lines.append("")
        lines.append("Top find_violations strategies (seconds):")
        top = sorted(per.items(), key=lambda kv: -float(kv[1]))[:10]
        for name, secs in top:
            lines.append(f"  {name:36s} {float(secs):.3f}")
    return "\n".join(lines)


def _run(cfg: Config) -> tuple[str, str | None, str | None]:
    # returns (summary_message, timings_report_or_none, validator_report_or_none)
    instance = load_instance(
        dataset_root=cfg.dataset_root,
        dataset_name=cfg.dataset,
        history_idx=cfg.history_idx,
        week_indices=_parse_weeks(cfg.weeks),
    )
    scenario, history, week_data_list = instance.scenario, instance.initial_history, instance.weeks

    week_summary: str | None = None
    if cfg.init_source == "week_level":
        import random as _py_random
        from bandit.linucb import LinUCB
        from week_level.context_builder import build_context
        from week_level.runner import first_arm_selector, run_week_level
        from week_level.arms import (
            CoverageFirstArm,
            FatigueAwareArm,
            WeekendBalancingArm,
            PreferenceRespectingArm,
        )

        if cfg.week_mode == "coverage_only":
            arms = [CoverageFirstArm()]
        else:
            arms = [
                CoverageFirstArm(),
                FatigueAwareArm(),
                WeekendBalancingArm(),
                PreferenceRespectingArm(),
            ]

        run_kwargs: dict[str, Any] = {
            "scenario": scenario,
            "initial_history": history,
            "week_data_list": week_data_list,
            "arms": arms,
        }

        if cfg.week_mode == "all_random":
            _rng = _py_random.Random(cfg.seed)
            run_kwargs["selector"] = lambda arms, week_idx: _rng.choice(arms)
        elif cfg.week_mode == "all_linucb":
            probe, _ = build_context(
                scenario, history, week_data_list[0], 0, len(week_data_list),
            )
            run_kwargs["bandit"] = LinUCB(
                num_arms=len(arms),
                context_dim=probe.shape[0],
                alpha=cfg.linucb_alpha,
                seed=cfg.seed,
            )
            # Raw soft-penalty deltas are O(1000s); scale down so theta stays
            # O(1) and the UCB exploration bonus can compete with exploited
            # arms across the short 4-8 week horizon.
            run_kwargs["reward_scale"] = 1000.0
        else:  # coverage_only or all_first
            run_kwargs["selector"] = first_arm_selector

        wk = run_week_level(**run_kwargs)
        init_schedule: Schedule = wk["schedule"]
        picks = wk.get("arms_picked", [])
        from collections import Counter
        pick_counts = Counter(picks)
        week_summary = (
            f"week_mode={cfg.week_mode} picks={dict(pick_counts)} "
            f"per_week={picks}"
        )
    else:
        init_schedule = generate_initial_schedule(scenario, history, week_data_list)

    strategies = build_all_strategies(scenario, history, week_data_list, seed=cfg.seed)
    selector = get_bandit(cfg.bandit, strategy_names=[s.name for s in strategies], seed=cfg.seed)

    repair_rounds = 0 if not cfg.run_repairs else cfg.rounds

    t0 = time.perf_counter()
    out: RunResult = run_repairs(
        scenario=scenario,
        history=history,
        week_data_list=week_data_list,
        strategies=strategies,
        schedule=init_schedule,
        selector=selector,
        num_rounds=repair_rounds,
        seed=cfg.seed,
        collect_timings=cfg.profile_timings,
        lazy_find_violations=cfg.lazy_find_violations,
    )
    runtime_s = time.perf_counter() - t0

    # write artifact (same style)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    artifact_path = out_dir / f"{instance.dataset_name}_{cfg.bandit}_seed{cfg.seed}_{ts}.json"
    import json

    artifact = {
        "meta": {
            "dataset": instance.dataset_name,
            "dataset_root": cfg.dataset_root,
            "history_idx": cfg.history_idx,
            "weeks": _parse_weeks(cfg.weeks),
            "init_source": cfg.init_source,
            "run_repairs": cfg.run_repairs,
            "bandit": cfg.bandit,
            "rounds_requested": repair_rounds,
            "seed": cfg.seed,
            "runtime_s": runtime_s,
        },
        "result": out.to_dict(),
    }
    stats_fn = getattr(selector, "stats", None)
    if callable(stats_fn):
        artifact["bandit_stats"] = stats_fn()
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))

    delta = out.initial_penalty - out.final_penalty
    summary = f"Done: final={out.final_penalty} (Δ={delta:+d}) in {runtime_s:.1f}s | wrote {artifact_path}"
    if week_summary:
        summary = f"{summary} | {week_summary}"

    timings_report: str | None = _format_timings(out) if cfg.profile_timings else None

    validator_report: str | None = None
    if cfg.final_validator:
        from schedule.penalty import compute_penalty

        final = compute_penalty(out.schedule, scenario, week_data_list, history)
        validator_report = format_schedule(out.schedule) + "\n\n" + format_validator_report(final)
    return summary, timings_report, validator_report


def _app(stdscr: curses.window) -> int:
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    cfg = Config()
    msg = ""
    field_idx = 0
    num_fields = 15
    week_modes = ["coverage_only", "all_first", "all_random", "all_linucb"]

    def _cycle_week_mode() -> None:
        idx = week_modes.index(cfg.week_mode) if cfg.week_mode in week_modes else 0
        cfg.week_mode = week_modes[(idx + 1) % len(week_modes)]

    while True:
        _draw(stdscr, cfg, field_idx, msg)
        ch = stdscr.getch()

        if ch in (ord("q"), 27):  # q or ESC
            return 0
        if ch in (curses.KEY_UP, ord("k")):
            field_idx = (field_idx - 1) % num_fields
            continue
        if ch in (curses.KEY_DOWN, ord("j")):
            field_idx = (field_idx + 1) % num_fields
            continue
        if ch == ord(" "):
            if field_idx == 4:
                cfg.init_source = "week_level" if cfg.init_source == "greedy" else "greedy"
            elif field_idx == 5:
                _cycle_week_mode()
            elif field_idx == 7:
                cfg.run_repairs = not cfg.run_repairs
            elif field_idx == 12:
                cfg.final_validator = not cfg.final_validator
            elif field_idx == 13:
                cfg.profile_timings = not cfg.profile_timings
            elif field_idx == 14:
                cfg.lazy_find_violations = not cfg.lazy_find_violations
            continue
        if ch in (ord("r"),):
            msg = "Running..."
            _draw(stdscr, cfg, field_idx, msg)
            try:
                summary, timings_report, validator_report = _run(cfg)
                msg = summary
                if timings_report:
                    _show_report(stdscr, timings_report, title="Timing report (q to close, ↑/↓ to scroll)")
                if validator_report:
                    _show_report(stdscr, validator_report, title="Final validator report (q to close, ↑/↓ to scroll)")
            except Exception as e:  # keep UI alive
                msg = f"Error: {e}"
            continue
        if ch in (curses.KEY_ENTER, 10, 13):
            try:
                if field_idx == 0:
                    cfg.dataset_root = _prompt_line(stdscr, "Dataset root", cfg.dataset_root)
                    # refresh dataset if root changed
                elif field_idx == 1:
                    opts = _list_datasets(cfg.dataset_root)
                    cfg.dataset = _choose_from_list(stdscr, "Choose dataset", opts, cfg.dataset)
                elif field_idx == 2:
                    cfg.history_idx = int(_prompt_line(stdscr, "History idx", str(cfg.history_idx)))
                elif field_idx == 3:
                    cfg.weeks = _prompt_line(stdscr, "Weeks", cfg.weeks)
                elif field_idx == 4:
                    cfg.init_source = "week_level" if cfg.init_source == "greedy" else "greedy"
                elif field_idx == 5:
                    _cycle_week_mode()
                elif field_idx == 6:
                    cfg.linucb_alpha = float(_prompt_line(stdscr, "LinUCB alpha", str(cfg.linucb_alpha)))
                elif field_idx == 7:
                    cfg.run_repairs = not cfg.run_repairs
                elif field_idx == 8:
                    cfg.bandit = _choose_from_list(stdscr, "Choose bandit", available_bandits(), cfg.bandit)
                elif field_idx == 9:
                    cfg.rounds = int(_prompt_line(stdscr, "Rounds", str(cfg.rounds)))
                elif field_idx == 10:
                    cfg.seed = int(_prompt_line(stdscr, "Seed", str(cfg.seed)))
                elif field_idx == 11:
                    cfg.output_dir = _prompt_line(stdscr, "Output dir", cfg.output_dir)
                elif field_idx == 12:
                    cfg.final_validator = not cfg.final_validator
                elif field_idx == 13:
                    cfg.profile_timings = not cfg.profile_timings
                elif field_idx == 14:
                    cfg.lazy_find_violations = not cfg.lazy_find_violations
            except ValueError:
                msg = "Invalid number."
            continue


def _show_report(stdscr: curses.window, report: str, title: str) -> None:
    lines = report.splitlines()
    idx = 0
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addnstr(0, 0, title, w - 1, curses.A_BOLD)
        view = lines[idx : idx + (h - 2)]
        for i, line in enumerate(view):
            stdscr.addnstr(1 + i, 0, line, w - 1)
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (ord("q"), 27, curses.KEY_ENTER, 10, 13):
            return
        if ch in (curses.KEY_UP, ord("k")):
            idx = max(0, idx - 1)
        elif ch in (curses.KEY_DOWN, ord("j")):
            idx = min(max(0, len(lines) - 1), idx + 1)
        elif ch == curses.KEY_NPAGE:
            idx = min(max(0, len(lines) - 1), idx + (h - 2))
        elif ch == curses.KEY_PPAGE:
            idx = max(0, idx - (h - 2))


def main() -> int:
    # Curses needs a real TTY.
    if not sys.stdin.isatty():
        print("This UI requires a TTY. Run `python src/tui.py` in a terminal.")
        return 2
    return curses.wrapper(_app)


if __name__ == "__main__":
    raise SystemExit(main())

