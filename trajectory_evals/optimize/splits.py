"""Train, val and holdout splits of a case source, by row label or by hashed id."""

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from trajectory_evals.core.protocols import CaseSource
from trajectory_evals.core.types import DatasetExample
from trajectory_evals.optimize.settings import Split, settings
from trajectory_evals.simulation.cases import _MemorySource

logger = logging.getLogger(__name__)

TRAIN, VAL, HOLDOUT = "train", "val", "holdout"
SPLITS = (TRAIN, VAL, HOLDOUT)
_HASH_BYTES = 8  # 64 bits of the digest place a case in [0, 1) evenly enough
_HASH_SPAN = 2 ** (8 * _HASH_BYTES)


@dataclass(frozen=True)
class Splits:
    """A source's cases in three in-memory sources; labelled if rows chose them."""

    train: CaseSource
    val: CaseSource
    holdout: CaseSource
    labelled: bool

    def sizes(self) -> dict[str, int]:
        """Return how many cases each split holds."""
        return {name: sum(1 for _ in getattr(self, name).examples()) for name in SPLITS}


def split_cases(source: CaseSource, ratios: Split, *, seed: int) -> Splits:
    """Return the source's cases split into train, val and holdout.

    When every case's metadata names its split, that split is used; otherwise
    each case goes where sha256("<seed>:<case_id>") falls among the ratios, so
    the same seed and ids always give the same splits. Raises ValueError for a
    split label other than train, val or holdout, or for an empty split.
    Logs a warning when val holds fewer cases than [run].min_val_cases.
    """
    cases = list(source.examples())
    labels = [case["metadata"].get(settings.keys.split) for case in cases]
    labelled = all(label is not None for label in labels)
    if not labelled:
        if any(label is not None for label in labels):
            logger.warning(settings.texts.partial_split)
        labels = [_hashed_split(case["case_id"], ratios, seed) for case in cases]
    groups = _group(cases, labels)
    minimum = settings.run.min_val_cases
    if len(groups[VAL]) < minimum:
        logger.warning(settings.texts.small_val, len(groups[VAL]), minimum)
    named = {
        name: _MemorySource(f"{source.name}-{name}", groups[name]) for name in SPLITS
    }
    return Splits(**named, labelled=labelled)


def _group(
    cases: Sequence[DatasetExample], labels: Sequence[object]
) -> dict[str, tuple[DatasetExample, ...]]:
    """Return the cases by split label; ValueError for a bad label or empty split."""
    texts = settings.texts
    unknown = sorted({str(label) for label in labels} - set(SPLITS))
    if unknown:
        raise ValueError(texts.bad_split.format(allowed=list(SPLITS), labels=unknown))
    groups = {
        name: tuple(
            case for case, label in zip(cases, labels, strict=True) if label == name
        )
        for name in SPLITS
    }
    # aevaluate refuses an empty dataset, so a run could not start on one.
    empty = [name for name in SPLITS if not groups[name]]
    if empty:
        raise ValueError(texts.empty_split.format(splits=empty))
    return groups


def _hashed_split(case_id: str, ratios: Split, seed: int) -> str:
    """Return the split the case's hashed id falls in."""
    digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
    point = int.from_bytes(digest[:_HASH_BYTES]) / _HASH_SPAN
    if point < ratios.train:
        return TRAIN
    return VAL if point < ratios.train + ratios.val else HOLDOUT


__all__ = ["HOLDOUT", "SPLITS", "TRAIN", "VAL", "Splits", "split_cases"]
