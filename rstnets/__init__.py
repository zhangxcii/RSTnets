import os

os.environ.setdefault("DDE_BACKEND", "tensorflow")

from ._version import __version__
from .config import CaseConfig, RunConfig

__all__ = ["__version__", "RSTnet", "FCNN", "CaseConfig", "RunConfig"]


def __getattr__(name):
    if name == "RSTnet":
        from .model import RSTnet
        return RSTnet
    if name == "FCNN":
        from .networks import FCNN
        return FCNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
