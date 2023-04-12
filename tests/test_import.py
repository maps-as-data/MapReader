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
        Downloader,
        SheetDownloader,
    )
    import mapreader.download.azure_access as azure_access
    from mapreader.annotate.utils import prepare_annotation, save_annotation

    # These imports are the various geo packages that previously where a separate subpackage
    import geopy
    import rasterio
    import keplergl
    import simplekml
