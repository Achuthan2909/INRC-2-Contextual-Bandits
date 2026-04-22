"""Named dataset splits for cross-instance training and evaluation.

Thin convenience layer on top of :func:`data.instances.enumerate_instances`
so training scripts can pass ``--split {train,dev,val,test}`` and evaluation
scripts know which instances to use as held-out data.

Splits (by directory, all under ``Dataset/``):

* ``train`` → ``datasets_json``      (primary training set, standard layout)
* ``dev``   → ``testdatasets_json``  (standard layout; quick sanity checks)
* ``val``   → ``hidden-JSON``        (flat layout, larger scenarios)
* ``test``  → ``late-dataset-json``  (nested layout, INRC-II late round)

Use ``train`` to fit the bandit, ``val`` or ``test`` for held-out reporting.
"""
from __future__ import annotations

from typing import Iterator

from data.instances import (
    dev_instances,
    enumerate_instances,
    test_instances,
    train_instances,
    val_instances,
)
from instance_loader import INRCInstance


SPLITS = ("train", "dev", "val", "test")


def split_instances(
    split: str,
    **kwargs,
) -> Iterator[INRCInstance]:
    """Yield instances from the named split.

    Parameters are forwarded to :func:`data.instances.enumerate_instances`
    (e.g. ``seed``, ``week_combos_per_scenario``, ``scenarios``).
    """
    split = split.lower()
    if split == "train":
        return train_instances(**kwargs)
    if split == "dev":
        return dev_instances(**kwargs)
    if split == "val":
        return val_instances(**kwargs)
    if split == "test":
        return test_instances(**kwargs)
    raise ValueError(f"unknown split '{split}'; must be one of {SPLITS}")


__all__ = ["SPLITS", "split_instances"]
