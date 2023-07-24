

def test_import():
    # This is based on all the imports found in the various tutorial notebooks
    from mapreader import (
        ClassifierContainer,
        load_patches,
        AnnotationsLoader,
        loader,
        MapImages,
        PatchDataset,
        Downloader,
        SheetDownloader,
    )
    from mapreader.annotate.utils import prepare_annotation, save_annotation

    # These imports are the various geo packages that previously where a separate subpackage
    import geopy
    import rasterio
    import keplergl
    import simplekml
