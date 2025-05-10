"""
Connect Five – package bootstrap.

Usage:
    python -m connect_five      # launch the game GUI
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("connect_five")
except PackageNotFoundError:      # editable/dev install
    __version__ = "0.0.0.dev0"

# public helpers -----------------------------------------------------------
from .game import main as run     # so callers can:  from connect_five import run
__all__ = ["run"]
