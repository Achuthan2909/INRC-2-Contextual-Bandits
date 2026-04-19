"""Enumerate INRC-II instances from dataset folders for cross-instance training.

Folder layouts:

* **Standard** — ``dataset_root/<scenario>/Sc-*.json``, ``H0-*.json``, ``WD-*.json``
  (e.g. ``Dataset/datasets_json/n030w4/``).
* **Nested** — ``dataset_root/<N-weeks>/<scenario>/`` with the same file pattern
  (e.g. ``late-dataset-json/4-weeks/n030w4/``).
* **Flat** — all JSON in ``dataset_root`` with prefixed names
  (e.g. ``hidden-JSON/Sc-n035w4.json``).
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from instance_loader import (
    INRCInstance,
    load_dataset_files,
    load_instance,
    load_instance_from_bundle,
)

# Repo-relative default roots (resolved at call time).
_TRAIN_REL = Path("Dataset") / "datasets_json"
_DEV_REL = Path("Dataset") / "testdatasets_json"
_VAL_REL = Path("Dataset") / "hidden-JSON"
_TEST_REL = Path("Dataset") / "late-dataset-json"

_WEEKS_DIR_RE = re.compile(r"^\d+-weeks$")
_SCENARIO_TAIL_RE = re.compile(r"^n\d+w\d+$")
_H0_STEM_RE = re.compile(r"^H0-(n\d+w\d+)-\d+$")
_WD_STEM_RE = re.compile(r"^WD-(n\d+w\d+)-\d+$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_root(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve()


@dataclass(frozen=True)
class _ScenarioRef:
    """Where to load one scenario from (no JSON loaded yet, except flat file lists)."""

    dataset_root: Path
    dataset_name: str
    layout: str  # "standard" | "nested" | "flat"
    scenario_file: Path | None = None
    history_files: tuple[Path, ...] | None = None
    week_files: tuple[Path, ...] | None = None


def infer_horizon_length(scenario_name: str) -> int | None:
    """Parse trailing ``wN`` from the last path segment (e.g. ``n030w4`` -> 4)."""
    tail = scenario_name.split("/")[-1]
    m = re.search(r"w(\d+)$", tail)
    return int(m.group(1)) if m else None


def _discover_standard(root: Path) -> list[_ScenarioRef]:
    out: list[_ScenarioRef] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not list(child.glob("Sc-*.json")):
            continue
        out.append(
            _ScenarioRef(
                dataset_root=root,
                dataset_name=child.name,
                layout="standard",
            )
        )
    return out


def _discover_nested(root: Path) -> list[_ScenarioRef]:
    out: list[_ScenarioRef] = []
    for hdir in sorted(root.iterdir()):
        if not hdir.is_dir() or not _WEEKS_DIR_RE.match(hdir.name):
            continue
        for sdir in sorted(hdir.iterdir()):
            if not sdir.is_dir():
                continue
            sc = list(sdir.glob("Sc-*.json"))
            if len(sc) != 1:
                continue
            if not _SCENARIO_TAIL_RE.match(sdir.name):
                continue
            rel = f"{hdir.name}/{sdir.name}"
            out.append(
                _ScenarioRef(
                    dataset_root=root,
                    dataset_name=rel,
                    layout="nested",
                )
            )
    return out


def _discover_flat(root: Path) -> list[_ScenarioRef]:
    """Multiple scenarios in one directory (filename prefixes)."""
    sc_files = sorted(root.glob("Sc-*.json"))
    if not sc_files:
        return []

    by_scen: dict[str, dict[str, list[Path]]] = defaultdict(
        lambda: {"sc": [], "h0": [], "wd": []}
    )
    for p in sc_files:
        name = p.stem[3:] if p.stem.startswith("Sc-") else p.stem
        by_scen[name]["sc"].append(p)

    for p in sorted(root.glob("H0-*.json")):
        m = _H0_STEM_RE.match(p.stem)
        if m:
            by_scen[m.group(1)]["h0"].append(p)
    for p in sorted(root.glob("WD-*.json")):
        m = _WD_STEM_RE.match(p.stem)
        if m:
            by_scen[m.group(1)]["wd"].append(p)

    out: list[_ScenarioRef] = []
    for name, bundle in sorted(by_scen.items()):
        if len(bundle["sc"]) != 1:
            continue
        h0 = sorted(bundle["h0"])
        wd = sorted(bundle["wd"])
        if not h0 or not wd:
            continue
        out.append(
            _ScenarioRef(
                dataset_root=root,
                dataset_name=name,
                layout="flat",
                scenario_file=bundle["sc"][0],
                history_files=tuple(h0),
                week_files=tuple(wd),
            )
        )
    return out


def _discover_scenarios(dataset_root: Path) -> list[_ScenarioRef]:
    """Pick discovery strategy: flat root, nested *-weeks, or standard subfolders."""
    if list(dataset_root.glob("Sc-*.json")):
        flat = _discover_flat(dataset_root)
        if flat:
            return flat

    nested = _discover_nested(dataset_root)
    if nested:
        return nested

    return _discover_standard(dataset_root)


def _metadata(ref: _ScenarioRef) -> tuple[int, int] | None:
    """Return (n_history, n_week_files) without loading JSON; None if unusable."""
    try:
        if ref.layout == "flat":
            assert ref.history_files is not None and ref.week_files is not None
            return len(ref.history_files), len(ref.week_files)
        files = load_dataset_files(ref.dataset_root, ref.dataset_name)
        return len(files["history_files"]), len(files["week_files"])
    except (FileNotFoundError, ValueError, OSError):
        return None


def _load(ref: _ScenarioRef, history_idx: int, week_indices: list[int]) -> INRCInstance:
    if ref.layout == "flat":
        assert ref.scenario_file is not None
        assert ref.history_files is not None
        assert ref.week_files is not None
        return load_instance_from_bundle(
            dataset_name=ref.dataset_name,
            scenario_file=ref.scenario_file,
            history_files=list(ref.history_files),
            week_files=list(ref.week_files),
            history_idx=history_idx,
            week_indices=week_indices,
        )
    return load_instance(
        ref.dataset_root,
        ref.dataset_name,
        history_idx,
        week_indices,
    )


def _week_combos(
    n_week_files: int,
    horizon: int,
    k: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    """Up to ``k`` distinct ordered tuples of ``horizon`` distinct WD indices."""
    if horizon > n_week_files or horizon < 1 or k < 1:
        return []
    uniq: dict[tuple[int, ...], None] = {}
    max_attempts = max(k * 100, k + 50)
    attempts = 0
    while len(uniq) < k and attempts < max_attempts:
        attempts += 1
        t = tuple(rng.sample(range(n_week_files), horizon))
        uniq.setdefault(t, None)
    return list(uniq.keys())


def enumerate_instances(
    dataset_root: str | Path,
    scenarios: list[str] | None = None,
    horizon_length: int | None = None,
    week_combos_per_scenario: int = 20,
    histories_per_scenario: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[INRCInstance]:
    """Yield :class:`INRCInstance` objects lazily (JSON loaded per yield).

    Parameters
    ----------
    dataset_root
        Folder to scan (relative paths are resolved from the repo root).
    scenarios
        If set, only scenario names whose last path segment or full relative
        name matches (e.g. ``n030w4`` or ``4-weeks/n030w4``).
    horizon_length
        If set, only scenarios whose inferred horizon (``wN`` in name) equals
        this value. If ``None``, inferred from each scenario name.
    week_combos_per_scenario
        Cap on random distinct week-index tuples sampled per
        (scenario, history) pair.
    histories_per_scenario
        If set, only the first *H* history files (by sorted order) are used.
    seed
        Seeded RNG for shuffles and week-tuple sampling.
    shuffle
        If True, shuffle scenario order and combo order (within caps).
    """
    root = _resolve_root(dataset_root)
    if not root.is_dir():
        return

    refs_all = _discover_scenarios(root)
    if scenarios is not None:
        want = set(scenarios)
        refs_all = [
            r
            for r in refs_all
            if r.dataset_name in want or r.dataset_name.split("/")[-1] in want
        ]

    # Filter by horizon and drop broken metadata
    refs: list[_ScenarioRef] = []
    for ref in refs_all:
        inferred = infer_horizon_length(ref.dataset_name)
        if inferred is None:
            continue
        if horizon_length is not None and inferred != horizon_length:
            continue
        meta = _metadata(ref)
        if meta is None:
            continue
        n_hist, n_wd = meta
        W = inferred if horizon_length is None else horizon_length
        if W > n_wd or n_hist < 1:
            continue
        refs.append(ref)

    rng = random.Random(seed)

    # Build work list: (ref, history_idx, week_tuple) — indices only until yield
    work: list[tuple[_ScenarioRef, int, tuple[int, ...]]] = []
    for ref in refs:
        meta = _metadata(ref)
        if meta is None:
            continue
        n_hist, n_wd = meta
        W = infer_horizon_length(ref.dataset_name)
        assert W is not None
        if W > n_wd:
            continue
        h_max = n_hist if histories_per_scenario is None else min(
            n_hist, histories_per_scenario
        )
        for h_idx in range(h_max):
            combos = _week_combos(
                n_wd, W, week_combos_per_scenario, rng
            )
            for combo in combos:
                work.append((ref, h_idx, combo))

    if shuffle:
        rng.shuffle(work)

    for ref, h_idx, combo in work:
        try:
            yield _load(ref, h_idx, list(combo))
        except (IndexError, FileNotFoundError, ValueError, OSError, KeyError):
            continue


def train_instances(**kwargs) -> Iterator[INRCInstance]:
    """``Dataset/datasets_json`` (default train split)."""
    return enumerate_instances(_resolve_root(_TRAIN_REL), **kwargs)


def dev_instances(**kwargs) -> Iterator[INRCInstance]:
    """``Dataset/testdatasets_json``."""
    return enumerate_instances(_resolve_root(_DEV_REL), **kwargs)


def val_instances(**kwargs) -> Iterator[INRCInstance]:
    """``Dataset/hidden-JSON`` (flat layout)."""
    return enumerate_instances(_resolve_root(_VAL_REL), **kwargs)


def test_instances(**kwargs) -> Iterator[INRCInstance]:
    """``Dataset/late-dataset-json`` (nested layout)."""
    return enumerate_instances(_resolve_root(_TEST_REL), **kwargs)


def _cli() -> None:
    p = argparse.ArgumentParser(description="Enumerate INRC-II instances (smoke test).")
    p.add_argument(
        "--split",
        choices=("train", "dev", "val", "test"),
        default="train",
        help="Which default dataset root to use.",
    )
    p.add_argument("--limit", type=int, default=5, help="Max instances to print.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-shuffle", action="store_true", help="Deterministic scenario order.")
    args = p.parse_args()

    split_map = {
        "train": train_instances,
        "dev": dev_instances,
        "val": val_instances,
        "test": test_instances,
    }
    gen = split_map[args.split](
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    n = 0
    for inst in gen:
        if n >= args.limit:
            break
        parts = inst.history_file.rsplit("-", 1)
        try:
            h_idx = int(parts[-1].replace(".json", ""))
        except ValueError:
            h_idx = -1
        week_idx = [
            int(x.rsplit("-", 1)[-1].replace(".json", "")) for x in inst.week_files
        ]
        tail = inst.dataset_name.split("/")[-1]
        wlen = infer_horizon_length(tail)
        print(
            f"{inst.dataset_name}\thistory={h_idx}\tweeks={week_idx}\thorizon={wlen}"
        )
        n += 1


if __name__ == "__main__":
    _cli()
