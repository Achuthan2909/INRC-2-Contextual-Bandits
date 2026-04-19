from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

@dataclass
class INRCInstance:
    dataset_name: str
    scenario: dict[str, Any]
    initial_history: dict[str, Any]
    weeks: list[dict[str, Any]]
    history_file: str
    week_files: list[str]
    scenario_file: str

def load_json (path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def load_dataset_files (dataset_root: str | Path, dataset_name: str) -> dict[str, Any]:
    dataset_dir = Path(dataset_root) / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    scenario_files = sorted(dataset_dir.glob("Sc-*.json"))
    history_files = sorted(dataset_dir.glob("H0-*.json"))
    week_files = sorted(dataset_dir.glob("WD-*.json"))

    if len(scenario_files) != 1:
        raise ValueError(f"Expected exactly one scenario file in {dataset_dir}, found {len(scenario_files)}")
    if not history_files:
        raise ValueError(f"No history files found in {dataset_dir}")
    if not week_files:
        raise ValueError(f"No week files found in {dataset_dir}")
    
    return {
        "dataset_dir": dataset_dir,
        "scenario_file": scenario_files[0],
        "history_files": history_files,
        "week_files": week_files
    }

def load_instance_from_bundle(
    dataset_name: str,
    scenario_file: Path,
    history_files: list[Path],
    week_files: list[Path],
    history_idx: int,
    week_indices: list[int],
) -> INRCInstance:
    """Load an instance given explicit paths (folder layout agnostic)."""
    if history_idx < 0 or history_idx >= len(history_files):
        raise IndexError(
            f"History index {history_idx} out of range (0 to {len(history_files) - 1})"
        )
    for idx in week_indices:
        if idx < 0 or idx >= len(week_files):
            raise IndexError(
                f"Week index {idx} out of range (0 to {len(week_files) - 1})"
            )

    history_file = history_files[history_idx]
    chosen_week_files = [week_files[i] for i in week_indices]

    scenario = load_json(scenario_file)
    initial_history = load_json(history_file)
    weeks = [load_json(p) for p in chosen_week_files]

    return INRCInstance(
        dataset_name=dataset_name,
        scenario=scenario,
        initial_history=initial_history,
        weeks=weeks,
        history_file=history_file.name,
        week_files=[p.name for p in chosen_week_files],
        scenario_file=scenario_file.name,
    )


def load_instance(
    dataset_root: str | Path,
    dataset_name: str,
    history_idx: int,
    week_indices: list[int],
) -> INRCInstance:
    files = load_dataset_files(dataset_root, dataset_name)
    return load_instance_from_bundle(
        dataset_name=dataset_name,
        scenario_file=files["scenario_file"],
        history_files=files["history_files"],
        week_files=files["week_files"],
        history_idx=history_idx,
        week_indices=week_indices,
    )

def summarize_instance(instance: INRCInstance) -> None:
    print(f"Dataset: {instance.dataset_name}")
    print(f"Scenario file: {instance.scenario_file}")
    print(f"Initial history: {instance.history_file}")
    print(f"Week files: {instance.week_files}")
    print(f"Number of weeks loaded: {len(instance.weeks)}")
    print("=" * 60)

    print("\nScenario top-level keys:")
    print(list(instance.scenario.keys()))

    print("\nInitial history top-level keys:")
    print(list(instance.initial_history.keys()))

    if instance.weeks:
        print("\nFirst week top-level keys:")
        print(list(instance.weeks[0].keys()))
