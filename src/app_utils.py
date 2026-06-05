"""Helpers for the Streamlit UI (timeouts, logging)."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError

_logger = logging.getLogger("chatpdf.streamlit")
if not _logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)


def log_line(msg: str) -> None:
    _logger.info(msg)


def run_with_timeout(fn, timeout_sec: float, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a worker thread; raise on timeout."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout_sec)
