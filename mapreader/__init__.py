from mapreader.loader.images import mapImages
from mapreader.loader.loader import loader
from mapreader.loader.loader import load_patches

from mapreader.download import azure_access
from mapreader.download.tileserver_access import TileServer

from mapreader.annotate.load_annotate import loadAnnotations

from mapreader.train.datasets import patchTorchDataset
from mapreader.train.datasets import patchContextDataset
from mapreader.train.classifier import classifier
from mapreader.train.classifier_context import classifierContext
from mapreader.train import custom_models

from mapreader.process import process

from mapreader.utils import utils
