import pytest
from pytest import approx
from mapreader.utils import geo_utils
from pathlib import Path


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent / "sample_files"


def test_extractGeoInfo(sample_dir):
    image_ID = "cropped_101200740.27.tif"
    image_path = f"{sample_dir}/{image_ID}"
    shape, crs, coord = geo_utils.extractGeoInfo(image_path)
    assert shape == (9,9,3)
    assert crs == "EPSG:27700"
    assert coord == approx(
        (534348, 192378, 534349, 192379), rel=1e-0)


def test_reproject(sample_dir):
    image_ID = "cropped_101200740.27.tif"
    image_path = f"{sample_dir}/{image_ID}"
    _, _, new_crs, reprojected_coord, size_in_m = geo_utils.reprojectGeoInfo(
        image_path, calc_size_in_m="gc"
    )
    assert new_crs == "EPSG:4326"
    assert reprojected_coord == approx((-0.061, 51.6142, -0.0610, 51.614), rel=1e-2)
    # assert size_in_m == approx((1118.21355, 1118.02101, 869.14959, 869.14959), rel=1e-2) check this

