import pytest


def test_import():
    # This is based on all the imports found in the various tutorial notebooks
    from mapreader import (
        TileServer,
        classifier,
        load_patches,
        loadAnnotations,
        loader,
        mapImages,
        patchTorchDataset,
        read,
        read_patches,
    )
    from mapreader.annotate.utils import prepare_annotation, save_annotation
