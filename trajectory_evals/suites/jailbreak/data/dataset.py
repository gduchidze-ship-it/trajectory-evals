"""Core's LangSmith dataset mirror under the jailbreak suite's example-id namespace."""

from trajectory_evals.core import dataset
from trajectory_evals.suites.jailbreak.settings import settings


class LangSmithDatasetSync(dataset.LangSmithDatasetSync):
    """Mirrors examples under the suite's namespace, so stored example ids hold."""

    def __init__(self, client: dataset._LangSmithClient) -> None:
        """Mirror datasets through client; nothing is called until sync."""
        super().__init__(client, namespace=settings.data.example_id_namespace)


__all__ = ["LangSmithDatasetSync"]
