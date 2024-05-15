from mapreader.load.images import MapImages
from mapreader.load.loader import loader
from mapreader.load.loader import load_patches

from mapreader.download.sheet_downloader import SheetDownloader
from mapreader.download.downloader import Downloader
from mapreader.download.downloader_utils import create_polygon_from_latlons, create_line_from_latlons

from mapreader.classify.load_annotations import AnnotationsLoader
from mapreader.classify.datasets import PatchDataset
from mapreader.classify.datasets import PatchContextDataset
from mapreader.classify.classifier import ClassifierContainer
from mapreader.classify import custom_models

try:
    from mapreader.spot_text.deepsolo_runner import DeepSoloRunner
except ImportError:
    pass

try:
    from mapreader.spot_text.dptext_detr_runner import DPTextDETRRunner
except ImportError:
    pass

from mapreader.process import process

from mapreader.annotate.annotator import Annotator

from . import _version

__version__ = _version.get_versions()["version"]

from mapreader.load import geo_utils

import mapreader

def print_version():
    print(mapreader.__version__)
