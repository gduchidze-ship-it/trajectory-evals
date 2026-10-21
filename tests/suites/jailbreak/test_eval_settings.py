"""Tests for the eval settings loaded from config.toml and the environment."""

from collections.abc import Callable
from types import ModuleType

import pytest
from pydantic import ValidationError

from trajectory_evals.suites.jailbreak.detection import (
    create_async_detection_match_evaluator,
    create_async_escalation_evaluator,
    create_async_fail_open_evaluator,
    create_detection_match_evaluator,
    create_escalation_evaluator,
    create_fail_open_evaluator,
    match,
    routing,
)
from trajectory_evals.suites.jailbreak.settings import (
    CONFIG_PATH,
    JailbreakEvalSettings,
    settings,
)
from trajectory_evals.suites.jailbreak.simulation import (
    create_async_attack_attempted_evaluator,
    create_async_persona_faithful_evaluator,
    create_async_simulation_correct_evaluator,
    create_async_turns_to_block_evaluator,
    create_attack_attempted_evaluator,
    create_persona_faithful_evaluator,
    create_simulation_correct_evaluator,
    create_turns_to_block_evaluator,
    evaluators,
)

PERSONAS = [
    "benign_follow_up",
    "benign_frustrated",
    "crescendo_trust_builder",
    "roleplay_framer",
    "trigger_word_planter",
    "instruction_override",
    "data_exfiltration",
]


def test_shipped_toml_loads_every_section() -> None:
    """The shipped config.toml loads with its personas in registry order."""
    loaded = JailbreakEvalSettings()

    assert CONFIG_PATH.is_file()
    assert loaded == settings
    assert list(loaded.personas) == PERSONAS
    assert loaded.data.example_id_namespace == "jailbreak-evals.impel.ai"


def test_environment_overrides_one_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JAILBREAK_EVALS_<SECTION>__<KEY> variable replaces only that value."""
    monkeypatch.setenv("JAILBREAK_EVALS_SIMULATION__SEED", "7")

    loaded = JailbreakEvalSettings()

    assert loaded.simulation.seed == 7
    assert loaded.simulation.sample == settings.simulation.sample
    assert loaded.detection == settings.detection


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("JAILBREAK_EVALS_CLI__PROGRAM", "evals", "Extra inputs are not permitted"),
        ("JAILBREAK_EVALS_PERSONAS__ROLEPLAY_FRAMER__KIND", "neutral", "'attack'"),
        ("JAILBREAK_EVALS_DETECTION__ESCALATED_KEY", "correct", "must be unique"),
        ("JAILBREAK_EVALS_TEXTS__NO_DATA", "no data in {dir}", "placeholders"),
        ("JAILBREAK_EVALS_LANGSMITH__ERRORED_OUTPUT_KEY", "result", "'output'"),
        ("JAILBREAK_EVALS_DETECTION__METRICS__RECALL", "recall", "Extra inputs"),
        ("JAILBREAK_EVALS_HELP__PERSONAS", "personas: {all}", "placeholders"),
        ("JAILBREAK_EVALS_CLI__METADATA_KEYS__SEED", "sample", "must be unique"),
    ],
)
def test_invalid_values_fail_at_load(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, message: str
) -> None:
    """An unknown key, a bad persona kind, a repeat or a stray placeholder fails."""
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError, match=message):
        JailbreakEvalSettings()


@pytest.mark.parametrize(
    ("create_evaluator", "key"),
    [
        (create_detection_match_evaluator, "correct"),
        (create_async_detection_match_evaluator, "correct"),
        (create_escalation_evaluator, "escalated"),
        (create_async_escalation_evaluator, "escalated"),
        (create_fail_open_evaluator, "failed_open"),
        (create_async_fail_open_evaluator, "failed_open"),
        (create_simulation_correct_evaluator, "simulation_correct"),
        (create_async_simulation_correct_evaluator, "simulation_correct"),
        (create_turns_to_block_evaluator, "turns_to_block"),
        (create_async_turns_to_block_evaluator, "turns_to_block"),
        (create_attack_attempted_evaluator, "attack_attempted"),
        (create_async_attack_attempted_evaluator, "attack_attempted"),
        (create_persona_faithful_evaluator, "persona_faithful"),
        (create_async_persona_faithful_evaluator, "persona_faithful"),
    ],
)
def test_each_evaluator_is_named_after_its_feedback_key(
    create_evaluator: Callable[[], object], key: str
) -> None:
    """LangSmith names the evaluator's run after its function, so it is the key."""
    assert getattr(create_evaluator(), "__name__") == key


@pytest.mark.parametrize(
    ("module", "section", "key", "factories"),
    [
        (
            match,
            "detection",
            "correct_key",
            (create_detection_match_evaluator, create_async_detection_match_evaluator),
        ),
        (
            routing,
            "detection",
            "escalated_key",
            (create_escalation_evaluator, create_async_escalation_evaluator),
        ),
        (
            routing,
            "detection",
            "failed_open_key",
            (create_fail_open_evaluator, create_async_fail_open_evaluator),
        ),
        (
            evaluators,
            "simulation",
            "correct_key",
            (
                create_simulation_correct_evaluator,
                create_async_simulation_correct_evaluator,
            ),
        ),
        (
            evaluators,
            "simulation",
            "turns_to_block_key",
            (create_turns_to_block_evaluator, create_async_turns_to_block_evaluator),
        ),
        (
            evaluators,
            "simulation",
            "attack_attempted_key",
            (
                create_attack_attempted_evaluator,
                create_async_attack_attempted_evaluator,
            ),
        ),
        (
            evaluators,
            "simulation",
            "persona_faithful_key",
            (
                create_persona_faithful_evaluator,
                create_async_persona_faithful_evaluator,
            ),
        ),
    ],
)
def test_renamed_feedback_key_renames_the_evaluator_run(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    section: str,
    key: str,
    factories: tuple[Callable[[], object], ...],
) -> None:
    """LangSmith names an evaluator's run after it, so the name follows the key."""
    renamed_section = getattr(settings, section).model_copy(update={key: "renamed"})
    renamed = settings.model_copy(update={section: renamed_section})
    monkeypatch.setattr(module, "settings", renamed)

    assert [getattr(create(), "__name__") for create in factories] == [
        "renamed",
        "renamed",
    ]
