import pytest
from pytest import approx
from mapreader.utils import geo_utils
from pathlib import Path

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent/"sample_files"

def test_extractGeoInfo(sample_dir):
    image_ID="101200740.27_JPEG.tif"
    image_path=f"{sample_dir}/{image_ID}"
    shape, crs, coord = geo_utils.extractGeoInfo(image_path)
    assert shape == (12447, 16967, 3)
    assert crs == 'EPSG:27700'
    assert coord == approx((534087.58772, 191720.18852, 535231.81704, 192559.59573), rel=1e-3)

def test_reproject(sample_dir):
    image_ID="101200740.27_JPEG.tif"
    image_path=f"{sample_dir}/{image_ID}"
    _, _, new_crs, reprojected_coord, size_in_m = geo_utils.reprojectGeoInfo(image_path, calc_size_in_m="gc")
    assert new_crs == "EPSG:4326"
    assert reprojected_coord == approx((-0.06471, -0.04852, 51.60808, 51.61590), rel=1e-3)
    assert size_in_m == approx((1118.21355, 1118.02101, 869.14959, 869.14959), rel=1e-3)
