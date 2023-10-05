# use version from in pyproject.toml
import pkg_resources

__version__ = pkg_resources.get_distribution("piffle").version
