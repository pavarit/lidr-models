"""Console encoding helpers.

On Windows the default console codec is cp1252, not UTF-8, so any ``print`` of
a non-ASCII character (the pipelines emit ``→`` in their progress lines) raises
``UnicodeEncodeError`` before the backtest does any work. Calling
``ensure_utf8_stdout`` at CLI entry reconfigures stdout/stderr to UTF-8 so the
pipelines run on a stock Windows shell without ``PYTHONIOENCODING=utf-8``.
"""

from __future__ import annotations

import contextlib
import sys


def ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 if the current encoding can't handle it.

    ``io.TextIOWrapper.reconfigure`` exists on Python 3.7+, but the streams may be
    redirected to objects that don't support it (e.g. a captured buffer under
    pytest); the ``getattr`` guard makes this a no-op in that case.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8":
            continue
        # Stream may not support re-encoding (e.g. a captured buffer); leave as-is.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")
