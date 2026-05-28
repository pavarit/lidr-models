"""Artifact loader. Read + validate. Used by tooling / tests; lidr has its
own TS reader generated from the same schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from lidr_core.contract.writer import validate_artifact


def load_artifact(path: Path) -> dict:
    """Read ``path`` as JSON, validate against the schema, and return the dict."""
    payload = json.loads(Path(path).read_text())
    validate_artifact(payload)
    return payload
