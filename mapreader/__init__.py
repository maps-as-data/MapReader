# download
from mapreader.download.sheet_downloader import SheetDownloader
try:
    from mapreader.download.iiif_downloader import IIIFDownloader
except ImportError:
    print("[WARNING] Piffle not installed, please install it from the `iiif_dataclasses` branch of https://github.com/rwood-97/piffle/tree/iiif_dataclasses")
    pass
from mapreader.download.downloader import Downloader
from mapreader.download.downloader_utils import create_polygon_from_latlons, create_line_from_latlons


# load
from mapreader.load.images import MapImages
from mapreader.load.loader import loader
from mapreader.load.loader import load_patches

# annotate
from mapreader.annotate.annotator import Annotator

# classify
from mapreader.classify.load_annotations import AnnotationsLoader
from mapreader.classify.datasets import PatchDataset
from mapreader.classify.datasets import PatchContextDataset
from mapreader.classify.classifier import ClassifierContainer
from mapreader.classify import custom_models

# spot_text
try:
    from mapreader.spot_text.deepsolo_runner import DeepSoloRunner
except ImportError:
    pass

try:
    from mapreader.spot_text.dptext_detr_runner import DPTextDETRRunner
except ImportError:
    pass

try:
    from mapreader.spot_text.maptext_runner import MapTextRunner
except ImportError:
    pass


# post process
from mapreader.process.context_post_process import ContextPostProcessor
from mapreader.process.occlusion_analysis import OcclusionAnalyzer

# utils
from mapreader.load import geo_utils


# version
from . import _version
__version__ = _version.get_versions()["version"]

import mapreader

def print_version():
    """Print the current version of mapreader."""
    print(mapreader.__version__)
