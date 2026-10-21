"""Skip the jailbreak suite's tests when its detector package is not importable."""

import importlib.util

_DETECTOR = "src.toolsets.jailbreak_detection"

try:
    _HAS_DETECTOR = importlib.util.find_spec(_DETECTOR) is not None
except ModuleNotFoundError:
    # find_spec imports the parent packages, so a missing src raises instead.
    _HAS_DETECTOR = False

if not _HAS_DETECTOR:
    collect_ignore_glob = ["suites/jailbreak/*"]
