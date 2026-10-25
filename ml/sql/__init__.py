"""SQL shared by the Python pipeline and the Go API.

Each statement lives once, in ``ml/sql/<name>.sql``, with ``%(name)s``
parameters. Python runs it as-is through psycopg; the Go package in this
directory embeds the same files and rewrites the parameters to pgx's
``@name`` form. Training-time candidates and online candidates therefore
come from the same text, and the parity checks guard the rest.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent


@cache
def load(name: str) -> str:
    """The statement in ``<name>.sql``."""
    return (_DIR / f"{name}.sql").read_text()
