"""Size-capped reads for the registry files ``crucible fuelcheck`` parses.

Input-boundary rule: a registry over its cap is rejected with its size and the cap, never
parsed — a 33 MB fleet registry took 46 s to parse and a multi-GB one would exhaust memory.
"""

from __future__ import annotations

import os
from pathlib import Path

_MAX_REGISTRY_BYTES = 1024 * 1024


class RegistryTooLargeError(OSError):
    """A registry file over its size cap (``EFBIG`` in spirit): rejected, never parsed."""


def read_capped(path: Path, cap: int) -> bytes:
    """Read ``path`` as bytes, reading at most ``cap + 1`` bytes even if it grows meanwhile.

    Raises:
        RegistryTooLargeError: the file holds more than ``cap`` bytes.
        OSError: the file cannot be opened or read.
    """
    with path.open("rb") as fh:
        data = fh.read(cap + 1)
        if len(data) > cap:
            size = os.fstat(fh.fileno()).st_size
            raise RegistryTooLargeError(f"{size} bytes, over the {cap}-byte cap")
    return data


def read_failure(exc: Exception, otherwise: str) -> str:
    """``is too large: ...`` for an oversized registry, else ``<otherwise>: <exc>``."""
    if isinstance(exc, RegistryTooLargeError):
        return f"is too large: {exc}"
    return f"{otherwise}: {exc}"
