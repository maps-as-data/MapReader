from __future__ import annotations

import pathlib

import adet
import pytest
from detectron2.engine import DefaultPredictor

print(adet.__version__)

from mapreader import MapTextRunner
from mapreader.load import MapImages


@pytest.fixture
def sample_dir():
    return pathlib.Path(__file__).resolve().parent.parent / "tests" / "sample_files"


@pytest.fixture
def init_dataframes(sample_dir, tmp_path):
    """Initializes MapImages object (with metadata from csv and patches) and creates parent and patch dataframes.
    Returns
    -------
    tuple
        path to parent and patch dataframes
    """
    maps = MapImages(f"{sample_dir}/cropped_74488689.png")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=3, path_save=tmp_path)  # gives 9 patches
    maps.add_center_coord(tree_level="parent")
    maps.add_patch_polygons()
    parent_df, patch_df = maps.convert_images()
    return parent_df, patch_df


def test_dptext_init(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = MapTextRunner(
        parent_df,
        patch_df,
    )
    assert isinstance(runner, MapTextRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
