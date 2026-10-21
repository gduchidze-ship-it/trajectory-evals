"""Tests for splitting cases into train, val and holdout."""

import logging
from collections.abc import Iterator

import pytest

from trajectory_evals.core.types import DatasetExample
from trajectory_evals.optimize.settings import Split, settings
from trajectory_evals.optimize.splits import SPLITS, Splits, split_cases

RATIOS = Split(train=0.5, val=0.25, holdout=0.25)


class _Source:
    """A case source over examples built in the test."""

    name = "golden"

    def __init__(self, cases: list[DatasetExample]) -> None:
        """Hold the examples to yield."""
        self._cases = cases

    def examples(self) -> Iterator[DatasetExample]:
        """Yield every example in order."""
        return iter(self._cases)


def _source(count: int, labels: list[str | None] | None = None) -> _Source:
    """Return count cases, each labelled with its entry in labels, if any."""
    labels = labels or [None] * count
    return _Source(
        [
            DatasetExample(
                case_id=f"case-{number}",
                inputs={},
                outputs={},
                metadata={} if label is None else {"split": label},
            )
            for number, label in enumerate(labels)
        ]
    )


def _ids(splits: Splits) -> dict[str, list[str]]:
    """Return each split's case ids in order."""
    return {
        name: [case["case_id"] for case in getattr(splits, name).examples()]
        for name in SPLITS
    }


def test_hashed_splits_are_deterministic_per_seed() -> None:
    """The same seed gives the same split; another seed moves some cases."""
    source = _source(80)

    first = _ids(split_cases(source, RATIOS, seed=0))
    again = _ids(split_cases(source, RATIOS, seed=0))
    other = _ids(split_cases(source, RATIOS, seed=1))

    assert first == again
    assert first != other
    assert sorted(sum(first.values(), [])) == sorted(f"case-{n}" for n in range(80))
    assert all(first[name] for name in SPLITS)


def test_every_case_labelled_uses_the_labels() -> None:
    """When every case names its split, those names decide, not the hash."""
    labels: list[str | None] = ["holdout", "train", "val", "train"]

    splits = split_cases(_source(4, labels), RATIOS, seed=0)

    assert splits.labelled
    assert _ids(splits) == {
        "train": ["case-1", "case-3"],
        "val": ["case-2"],
        "holdout": ["case-0"],
    }
    assert splits.sizes() == {"train": 2, "val": 1, "holdout": 1}
    assert splits.val.name == "golden-val"


def test_partly_labelled_cases_are_split_by_hash_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Labels on only some cases are ignored, and the fallback is logged."""
    labels: list[str | None] = ["train", *[None] * 79]

    with caplog.at_level(logging.WARNING):
        splits = split_cases(_source(80, labels), RATIOS, seed=0)

    assert not splits.labelled
    assert settings.texts.partial_split in caplog.text
    assert _ids(splits) == _ids(split_cases(_source(80), RATIOS, seed=0))


def test_a_small_val_split_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """Fewer val cases than [run].min_val_cases logs that acceptance is noisy."""
    labels: list[str | None] = ["train", "val", "holdout"]

    with caplog.at_level(logging.WARNING):
        split_cases(_source(3, labels), RATIOS, seed=0)

    assert "the val split has 1 cases, fewer than 20" in caplog.text


BAD = {
    "unknown_label": (["train", "val", "test"], "metadata.split must be one of"),
    "empty_split": (["train", "val", "val"], "the ['holdout'] split has no cases"),
}


@pytest.mark.parametrize(("labels", "reason"), BAD.values(), ids=BAD.keys())
def test_bad_labels_or_an_empty_split_are_refused(
    labels: list[str | None], reason: str
) -> None:
    """A label outside train/val/holdout, or a split with no case, raises."""
    with pytest.raises(ValueError, match=reason.replace("[", r"\[")):
        split_cases(_source(3, labels), RATIOS, seed=0)
