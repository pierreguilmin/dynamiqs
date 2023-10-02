from importlib.metadata import version

from .mesolve import mesolve
from .plots import *
from .sesolve import sesolve
from .smesolve import smesolve
from .utils import *

# get version from pyproject.toml
__version__ = version(__package__)
