from mapreader.loader.images import mapImages
from mapreader.loader.loader import loader
from mapreader.loader.loader import load_patches

from mapreader.download.sheet_downloader import SheetDownloader
from mapreader.download.downloader import Downloader
from mapreader.download.downloader_utils import create_polygon_from_latlons, create_line_from_latlons

from mapreader.annotate.load_annotate import loadAnnotations

from mapreader.train.datasets import patchTorchDataset
from mapreader.train.datasets import patchContextDataset
from mapreader.train.classifier import classifier
from mapreader.train.classifier_context import classifierContext
from mapreader.train import custom_models

from mapreader.process import process

from . import _version
__version__ = _version.get_versions()['version']

from mapreader.utils import geo_utils
