"""The simulation dataset: sampled real benign openers crossed with personas."""

import hashlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from itertools import zip_longest

from evals.jailbreak.protocols import CaseSource
from evals.jailbreak.settings import settings
from evals.jailbreak.simulation.personas import PERSONAS, Persona
from evals.jailbreak.types import (
    DatasetExample,
    PersonaKind,
    SimulationExpectation,
    SimulationInputs,
    SimulationReference,
)
from evals.jailbreak.utils import _has_text

_EXPECTED: Mapping[PersonaKind, SimulationExpectation] = {
    "benign": "allow",
    "attack": "block",
}


class SimulationCaseSource:
    """Crosses personas with seeded, per-channel ALLOWED openers lacking history."""

    def __init__(
        self, source: CaseSource, personas: Sequence[str], sample: int, seed: int
    ) -> None:
        """Hold the source and the named personas; reject unknown names or no sample."""
        texts = settings.texts
        unknown = sorted(set(personas) - PERSONAS.keys())
        if unknown:
            message = texts.unknown_personas
            raise ValueError(message.format(unknown=unknown, known=list(PERSONAS)))
        if sample < 1:
            raise ValueError(texts.at_least_one.format(name="sample", value=sample))
        self._source = source
        self._personas = [PERSONAS[name] for name in personas]
        self._sample = sample
        self._seed = seed

    @property
    def name(self) -> str:
        """Return the name of the source the openers come from."""
        return self._source.name

    def examples(self) -> Iterator[DatasetExample]:
        """Yield one example per sampled opener and persona, opener by opener."""
        for opener in self._openers():
            for persona in self._personas:
                yield _crossed(opener, persona)

    def _openers(self) -> list[DatasetExample]:
        """Return the seeded sample, drawing from each channel in turn."""
        by_channel: defaultdict[str, list[DatasetExample]] = defaultdict(list)
        for example in self._source.examples():
            if _is_benign_opener(example):
                by_channel[_channel(example)].append(example)
        strata = [
            sorted(cases, key=self._draw_order)
            for _, cases in sorted(by_channel.items())
        ]
        interleaved = [
            case for row in zip_longest(*strata) for case in row if case is not None
        ]
        return interleaved[: self._sample]

    def _draw_order(self, example: DatasetExample) -> str:
        """Return the case's seeded hash, which stays put when cases are appended."""
        key = f"{self._seed}:{example['case_id']}"
        return hashlib.sha256(key.encode()).hexdigest()


def _is_benign_opener(example: DatasetExample) -> bool:
    """Return whether the case is ALLOWED, has text and has no history before it."""
    text = example["inputs"].get("text")
    allowed = example["outputs"].get("expected_action") == "ALLOWED"
    fresh = not example["inputs"].get("history")
    return allowed and fresh and _has_text(text)


def _channel(example: DatasetExample) -> str:
    """Return the channel the case came in on, or the unknown channel if it has none."""
    return str(example["inputs"]["channel"] or settings.channels.unknown)


def _crossed(opener: DatasetExample, persona: Persona) -> DatasetExample:
    """Return the example in which persona continues the opener's conversation."""
    inputs = SimulationInputs(
        opening_message=opener["inputs"]["text"],
        persona=persona.name,
        channel=_channel(opener),
    )
    reference = SimulationReference(expected=_EXPECTED[persona.kind])
    return DatasetExample(
        case_id=f"{opener['case_id']}:{persona.name}",
        inputs=dict(inputs),
        outputs=dict(reference),
        metadata={
            settings.simulation.source_case_id_key: opener["case_id"],
            settings.simulation.persona_kind_key: persona.kind,
        },
    )
