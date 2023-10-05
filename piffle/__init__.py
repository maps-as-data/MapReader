# use version from in pyproject.toml
# NOTE: this will pull the version for the currently installed version
import importlib.metadata

__version__ = importlib.metadata.version("piffle")
