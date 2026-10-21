"""Simulation cases: scenario rows from JSONL, and red-team cases built in memory."""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field, PositiveInt

from trajectory_evals.core.jsonl import read_cases
from trajectory_evals.core.protocols import CaseSource
from trajectory_evals.core.settings import RawText, Text, _Section
from trajectory_evals.core.settings import settings as core_settings
from trajectory_evals.core.types import DatasetExample, Expectations
from trajectory_evals.simulation.personas import Persona
from trajectory_evals.simulation.settings import settings
from trajectory_evals.simulation.types import ScenarioInputs

_DIGEST_HEX = 8  # hex chars of the persona-set digest in a red-team source's name


class Scenario(_Section):
    """One scenario row; the keys it does not name become its metadata."""

    id: Text
    persona: Text
    scenario: Text
    opening_message: RawText | None = None
    max_turns: PositiveInt | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    expected: Expectations = Field(default_factory=Expectations)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_example(self) -> DatasetExample:
        """Return the scenario as a dataset example whose outputs are expected."""
        inputs = ScenarioInputs(
            persona=self.persona,
            scenario=self.scenario,
            opening_message=self.opening_message,
            max_turns=self.max_turns,
            context=dict(self.context),
        )
        return DatasetExample(
            case_id=self.id,
            inputs=dict(inputs),
            outputs=dict(self.expected),
            metadata=dict(self.metadata),
        )


_ROW_KEYS = Scenario.model_fields.keys() - {"metadata"}


class ScenarioSource:
    """The scenarios of one JSONL file, read lazily in file order."""

    def __init__(self, path: Path) -> None:
        """Keep the path of the JSONL file that iteration will read."""
        self._path = path

    @property
    def name(self) -> str:
        """Return the stem of the file the scenarios come from."""
        return self._path.stem

    def examples(self) -> Iterator[DatasetExample]:
        """Yield every scenario as a dataset example; ids never repeat.

        Raises CaseFormatError, naming path:line, when iteration reaches a bad row.
        """
        scenarios = read_cases(self._path, _scenario, lambda scenario: scenario.id)
        return (scenario.to_example() for scenario in scenarios)


def _scenario(row: dict[str, Any]) -> Scenario:
    """Build a scenario from one decoded row, its unnamed keys as metadata."""
    named = {key: value for key, value in row.items() if key in _ROW_KEYS}
    metadata = {key: value for key, value in row.items() if key not in _ROW_KEYS}
    return Scenario.model_validate({**named, "metadata": metadata})


@dataclass(frozen=True)
class MemorySource:
    """A CaseSource over examples built in memory."""

    name: str
    cases: tuple[DatasetExample, ...]

    def examples(self) -> Iterator[DatasetExample]:
        """Yield the held examples in order."""
        return iter(self.cases)


def redteam_cases(
    personas: Mapping[str, Persona], *, names: Sequence[str] | None, attempts: int
) -> CaseSource:
    """Return one case per attack persona in names, every one if None, per attempt.

    Ids are persona and attempt, so the same arguments give the same examples.
    The source's name holds a digest of the sorted persona names and attempts,
    so an uploaded run of one set of attacks never mirrors over another's.
    Raises ValueError for attempts below 1 or a name that is not an attack persona.
    """
    if attempts < 1:
        message = core_settings.texts.at_least_one
        raise ValueError(message.format(name="attempts", value=attempts))
    attackers = [name for name, persona in personas.items() if persona.kind == "attack"]
    chosen = attackers if names is None else list(dict.fromkeys(names))
    rejected = [name for name in chosen if name not in attackers]
    if rejected:
        message = settings.texts.not_attack_personas
        raise ValueError(message.format(names=rejected, known=attackers))
    cases = tuple(
        _redteam_case(name, attempt)
        for name in chosen
        for attempt in range(1, attempts + 1)
    )
    return MemorySource(_redteam_source(chosen, attempts), cases)


def _redteam_source(personas: Sequence[str], attempts: int) -> str:
    """Return the red-team source's name, the same for any order of personas."""
    attacks = json.dumps([sorted(personas), attempts]).encode()
    digest = hashlib.sha256(attacks).hexdigest()[:_DIGEST_HEX]
    return settings.keys.redteam_source.format(digest=digest)


def _redteam_case(persona: str, attempt: int) -> DatasetExample:
    """Return the example in which persona attacks for the attempt-th time."""
    inputs = ScenarioInputs(
        persona=persona,
        scenario=settings.prompts.redteam_scenario,
        opening_message=None,
        max_turns=None,
        context={},
    )
    case_id = settings.keys.redteam_case_id.format(persona=persona, attempt=attempt)
    return DatasetExample(case_id=case_id, inputs=dict(inputs), outputs={}, metadata={})


__all__ = ["MemorySource", "Scenario", "ScenarioSource", "redteam_cases"]
