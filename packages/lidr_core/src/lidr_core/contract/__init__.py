"""Artifact contract — JSON Schema + writer + loader. The single source of
truth for the model→lidr file format.
"""

from lidr_core.contract.loader import load_artifact
from lidr_core.contract.writer import build_artifact, validate_artifact, write_artifact

__all__ = ["build_artifact", "write_artifact", "validate_artifact", "load_artifact"]
