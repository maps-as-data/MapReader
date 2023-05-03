import pytest
from pytest import approx
from mapreader import loader
from mapreader.load.images import MapImages
import os
from pathlib import Path
import pathlib

# functions


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"

@pytest.fixture
def metadata_patchify(sample_dir, tmp_path):
    """Initialises MapImages object (with metadata from csv and patches).

    Returns
    -------
    list
        image_ID (of parent png image), ts_map (MapImages object), parent_list (== image_ID) and patch_list (list of patches).
    """
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    ts_map.patchify_all(patch_size=3, path_save=tmp_path) # gives 3 patches 
    parent_list = ts_map.list_parents()
    patch_list = ts_map.list_patches()

    return image_ID, ts_map, parent_list, patch_list


# tests


# png tests (separate geo info)
def test_loader_png(sample_dir):
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    assert len(ts_map) == 1
    assert isinstance(ts_map, MapImages)


def test_loader_add_metadata(sample_dir):
    #metadata csv
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    assert "coordinates" in ts_map.images["parent"][image_ID].keys()
    assert ts_map.images["parent"][image_ID]["coordinates"] == approx(
        (-4.83,55.80, -4.21, 56.059), rel=1e-2
    )
    #metadata xlsx
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.add_metadata(f"{sample_dir}/ts_downloaded_maps.xlsx")
    assert "coordinates" in ts_map.images["parent"][image_ID].keys()
    assert ts_map.images["parent"][image_ID]["coordinates"] == approx(
        (-4.83,55.80, -4.21, 56.059), rel=1e-2
    )


# tiff tests (no geo info)
def test_loader_tiff(sample_dir):
    image_ID = "cropped_non_geo.tif"
    tiff = loader(f"{sample_dir}/{image_ID}")
    assert len(tiff) == 1
    assert isinstance(tiff, MapImages)

# geotiff tests (contains geo info)
def test_loader_geotiff(sample_dir):
    image_ID = "cropped_geo.tif"
    geotiff = loader(f"{sample_dir}/{image_ID}")
    assert len(geotiff) == 1
    assert isinstance(geotiff, MapImages)

def test_loader_add_geo_info(sample_dir):
    # check it works for geotiff
    image_ID = "cropped_geo.tif"
    geotiff = loader(f"{sample_dir}/{image_ID}")
    geotiff.add_geo_info()
    assert "shape" in geotiff.images["parent"][image_ID].keys()
    assert "coordinates" in geotiff.images["parent"][image_ID].keys()
    assert geotiff.images["parent"][image_ID]["coordinates"] == approx((-0.061, 51.6142, -0.0610, 51.614), rel=1e-2)

    # check nothing happens for png/tiff (no metadata)
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    keys = list(ts_map.images["parent"][image_ID].keys())
    ts_map.add_geo_info()
    assert list(ts_map.images["parent"][image_ID].keys()) == keys

    image_ID = "cropped_non_geo.tif"
    tiff = loader(f"{sample_dir}/{image_ID}")
    keys = list(tiff.images["parent"][image_ID].keys())
    tiff.add_geo_info()
    assert list(tiff.images["parent"][image_ID].keys()) == keys

# could add jpeg, IIIF, etc. here too
def test_loader_tiff_32bit(sample_dir):
    image_ID = "cropped_32bit.tif"
    with pytest.raises(NotImplementedError, match = "Image mode"): 
        loader(f"{sample_dir}/{image_ID}")

# test other functions

def test_loader_patchify_all(sample_dir, tmp_path):
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.patchify_all(patch_size=3, path_save=tmp_path)
    parent_list = ts_map.list_parents()
    patch_list = ts_map.list_patches()
    assert len(parent_list) == 1
    assert len(patch_list) == 9
    assert os.path.isfile(f"{tmp_path}/patch-0-0-3-3-#{image_ID}#.png")


def test_loader_coord_functions(metadata_patchify, sample_dir):
    # test for png with added metadata
    image_ID, ts_map, _, patch_list = metadata_patchify
    ts_map.add_center_coord()
    assert "dlon" in ts_map.images["parent"][image_ID].keys()
    assert "center_lon" in ts_map.images["patch"][patch_list[0]].keys()

    # test for geotiff with added geoinfo
    image_ID = "cropped_geo.tif"
    geotiff = loader(f"{sample_dir}/{image_ID}")
    geotiff.add_geo_info()
    geotiff.add_coord_increments()
    geotiff.add_center_coord(tree_level="parent")
    assert "dlon" in geotiff.images["parent"][image_ID].keys()
    assert "center_lon" in geotiff.images["parent"][image_ID].keys()

    # test for tiff with no geo info (i.e. no coords so nothing should happen)
    image_ID = "cropped_non_geo.tif"
    tiff = loader(f"{sample_dir}/{image_ID}")
    keys = list(tiff.images["parent"][image_ID].keys())
    tiff.add_coord_increments()
    tiff.add_center_coord(tree_level="parent")
    assert list(tiff.images["parent"][image_ID].keys()) == keys

def test_loader_calc_pixel_stats(metadata_patchify, sample_dir, tmp_path):
    image_ID, ts_map, _, patch_list = metadata_patchify
    ts_map.calc_pixel_stats()
    expected_cols = ["mean_pixel_R", "mean_pixel_G", "mean_pixel_B", "mean_pixel_A", "std_pixel_R", "std_pixel_G", "std_pixel_B", "std_pixel_A",]
    for col in expected_cols:
        assert col in ts_map.images["patch"][patch_list[0]].keys()

    # geotiff/tiff will not have alpha channel, so only RGB returned
    image_ID = "cropped_geo.tif"
    geotiff = loader(f"{sample_dir}/{image_ID}")
    geotiff.patchify_all(patch_size=3, path_save=tmp_path)
    patch_list = geotiff.list_patches()
    geotiff.calc_pixel_stats()
    expected_cols = ["mean_pixel_R", "mean_pixel_G", "mean_pixel_B", "std_pixel_R", "std_pixel_G", "std_pixel_B"]
    for col in expected_cols:
        assert col in geotiff.images["patch"][patch_list[0]].keys()

def test_loader_convert_images(metadata_patchify):
    _, ts_map, _, _ = metadata_patchify
    parent_df, patch_df = ts_map.convert_images()
    assert parent_df.shape == (1, 12)
    assert patch_df.shape == (9, 5)