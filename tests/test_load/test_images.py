import pytest
from pytest import approx
from mapreader import loader
from mapreader.load.images import MapImages
import os
from pathlib import Path
import pathlib
import pandas as pd
import geopandas as geopd
from PIL import Image
import PIL
from random import randint
from shapely.geometry import Polygon

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"

@pytest.fixture
def init_ts_maps(sample_dir, tmp_path):
    """Initialises MapImages object (with metadata from csv and patches).

    Returns
    -------
    list
        image_ID (of parent png image), ts_map (MapImages object), parent_list (== image_ID) and patch_list (list of patches).
    """
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    ts_map.patchify_all(patch_size=3, path_save=tmp_path) # gives 9 patches 
    parent_list = ts_map.list_parents()
    patch_list = ts_map.list_patches()

    return image_ID, ts_map, parent_list, patch_list

@pytest.fixture
def keys():
    return ["parent_id", "image_path", "shape", "name", "coord", "other"]

@pytest.fixture
def metadata_df():
    return pd.DataFrame({"name":["file1.png", "file2.png", "file3.png"], "coord":[(1.1,1.5),(2.1,1.0),(3.1,4.5)], "other":[1,2,3]})

@pytest.fixture
def matching_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png", "file3.png"]
    for file in files:
        rand_colour = (randint(0,255), randint(0,255), randint(0,255))
        Image.new("RGB",(9,9), color = rand_colour).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep=",")
    metadata_df.to_csv(f"{test_path}/metadata_df.tsv", sep="\t")
    metadata_df.to_excel(f"{test_path}/metadata_df.xlsx")
    return test_path

@pytest.fixture
def extra_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png"]
    for file in files:
        rand_colour = (randint(0,255), randint(0,255), randint(0,255))
        Image.new("RGB",(9,9), color = rand_colour).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep=",")
    return test_path 

@pytest.fixture
def missing_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png", "file3.png", "file4.png"]
    for file in files:
        rand_colour = (randint(0,255), randint(0,255), randint(0,255))
        Image.new("RGB",(9,9), color = rand_colour).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep=",")
    return test_path


# ---- tests ----

# ---- png tests (separate geo info) ---

def test_loader_png(sample_dir):
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    assert len(ts_map) == 1
    assert isinstance(ts_map, MapImages)

# add_metadata tests w/ png files
def test_loader_add_metadata(sample_dir):
    #metadata csv
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    assert "coordinates" in ts_map.images["parent"][image_ID].keys()
    assert ts_map.images["parent"][image_ID]["coordinates"] == approx(
        (-4.83, 55.80, -4.21, 56.059), rel=1e-2
    )
    #metadata tsv
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.add_metadata(f"{sample_dir}/ts_downloaded_maps.tsv", delimiter="\t")
    assert "coordinates" in ts_map.images["parent"][image_ID].keys()
    assert ts_map.images["parent"][image_ID]["coordinates"] == approx(
        (-4.83, 55.80, -4.21, 56.059), rel=1e-2
    )
    #metadata xlsx
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.add_metadata(f"{sample_dir}/ts_downloaded_maps.xlsx")
    assert "coordinates" in ts_map.images["parent"][image_ID].keys()
    assert ts_map.images["parent"][image_ID]["coordinates"] == approx(
        (-4.83,55.80, -4.21, 56.059), rel=1e-2
    )

#if metadata info matches 
def test_matching_metdata_csv(matching_metadata_dir, keys):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{matching_metadata_dir}/metadata_df.csv")
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys

def test_matching_metdata_xlsx(matching_metadata_dir, keys):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{matching_metadata_dir}/metadata_df.xlsx")
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys

def test_matching_metadata_df(matching_metadata_dir, metadata_df, keys):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(metadata_df)
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys

#if you pass index col - this should pick up if index.name is 'name' or 'image_id'
def test_matching_metadata_csv_w_index_col(matching_metadata_dir):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{matching_metadata_dir}/metadata_df.csv", index_col="name")
    keys = ["parent_id", "image_path", "shape", "Unnamed: 0", "coord", "other", "name"]
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys

#if you pass columns
def test_matching_metadata_csv_w_usecols(matching_metadata_dir):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{matching_metadata_dir}/metadata_df.csv", columns=["name","coord"])
    keys = ["parent_id", "image_path", "shape", "name", "coord"]
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys
        assert isinstance(my_files.images["parent"][parent_id]["coord"], tuple)

#if there is extra info in the metadata
def test_extra_metadata_csv_ignore_mismatch(extra_metadata_dir,keys):  
    my_files=loader(f"{extra_metadata_dir}/*png")
    assert len(my_files)==2
    my_files.add_metadata(f"{extra_metadata_dir}/metadata_df.csv", ignore_mismatch=True)
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys
    
def test_extra_metadata_csv_errors(extra_metadata_dir):  
    my_files=loader(f"{extra_metadata_dir}/*png")
    assert len(my_files)==2
    with pytest.raises(ValueError, match="information about non-existant images"):
        my_files.add_metadata(f"{extra_metadata_dir}/metadata_df.csv")
    
#if there is missing info in metadata
def test_missing_metadata_csv_ignore_mismatch(missing_metadata_dir, keys):  
    my_files=loader(f"{missing_metadata_dir}/*png")
    assert len(my_files)==4
    my_files.add_metadata(f"{missing_metadata_dir}/metadata_df.csv", ignore_mismatch=True)
    for parent_id in ["file1.png", "file2.png", "file3.png"]:
        assert list(my_files.images["parent"][parent_id].keys()) == keys
    assert list(my_files.images["parent"]["file4.png"].keys()) == ["parent_id", "image_path", "shape"]
    
def test_missing_metadata_csv_errors(missing_metadata_dir):  
    my_files=loader(f"{missing_metadata_dir}/*png")
    assert len(my_files)==4
    with pytest.raises(ValueError, match="missing information"):
        my_files.add_metadata(f"{missing_metadata_dir}/metadata_df.csv")

# other errors
#if csv file doesn't exist
def test_metadata_not_found(matching_metadata_dir):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    with pytest.raises(ValueError):
        my_files.add_metadata(f"{matching_metadata_dir}/fakefile.csv")

def test_metadata_missing_name_or_image_id(matching_metadata_dir):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    incomplete_metadata_df = pd.DataFrame({"coord":[(1.1,1.5),(2.1,1.0),(3.1,4.5)], "other":[1,2,3]})
    incomplete_metadata_df.to_csv(f"{matching_metadata_dir}/incomplete_metadata_df.csv", sep=",")
    with pytest.raises(ValueError, match = "'name' or 'image_id' should be one of the columns"):
        my_files.add_metadata(incomplete_metadata_df)
    with pytest.raises(ValueError, match = "'name' or 'image_id' should be one of the columns"):
        my_files.add_metadata(f"{matching_metadata_dir}/incomplete_metadata_df.csv")


# --- tiff tests (no geo info) ---

def test_loader_tiff(sample_dir):
    image_ID = "cropped_non_geo.tif"
    tiff = loader(f"{sample_dir}/{image_ID}")
    assert len(tiff) == 1
    assert isinstance(tiff, MapImages)

# --- geotiff tests (contains geo info) ---

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

# -- could add jpeg, IIIF, etc. here too ---

def test_loader_tiff_32bit(sample_dir):
    image_ID = "cropped_32bit.tif"
    with pytest.raises(NotImplementedError, match = "Image mode"): 
        loader(f"{sample_dir}/{image_ID}")

def test_loader_non_image(sample_dir):
    file_ID = "ts_downloaded_maps.csv"
    with pytest.raises(PIL.UnidentifiedImageError, match="not an image"): 
        loader(f"{sample_dir}/{file_ID}")

# --- test other functions ---

def test_loader_patchify_all(sample_dir, tmp_path):
    image_ID = "cropped_74488689.png"
    ts_map = loader(f"{sample_dir}/{image_ID}")
    ts_map.patchify_all(patch_size=3, path_save=tmp_path)
    parent_list = ts_map.list_parents()
    patch_list = ts_map.list_patches()
    assert len(parent_list) == 1
    assert len(patch_list) == 9
    assert os.path.isfile(f"{tmp_path}/patch-0-0-3-3-#{image_ID}#.png")

def test_loader_coord_functions(init_ts_maps, sample_dir):
    # test for png with added metadata
    image_ID, ts_map, _, patch_list = init_ts_maps
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

def test_loader_calc_pixel_stats(init_ts_maps, sample_dir, tmp_path):
    image_ID, ts_map, _, patch_list = init_ts_maps
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

def test_loader_convert_images(init_ts_maps):
    _, ts_map, _, _ = init_ts_maps
    parent_df, patch_df = ts_map.convert_images()
    assert parent_df.shape == (1, 13)
    assert patch_df.shape == (9, 7)
    parent_df, patch_df = ts_map.convert_images(save=True)
    assert os.path.isfile("./parent_df.csv")
    assert os.path.isfile("./patch_df.csv")
    os.remove("./parent_df.csv")
    os.remove("./patch_df.csv")
    parent_df, patch_df = ts_map.convert_images(save=True, save_format="excel")
    assert os.path.isfile("./parent_df.xlsx")
    assert os.path.isfile("./patch_df.xlsx")
    os.remove("./parent_df.xlsx")
    os.remove("./patch_df.xlsx")

def test_loader_convert_images_errors(init_ts_maps):
    _, ts_map, _, _ = init_ts_maps
    with pytest.raises(ValueError, match="``save_format`` should be one of"):
        ts_map.convert_images(save=True, save_format="json")

def test_loader_add_patch_polygons(init_ts_maps):
    _, ts_map, _, patch_list = init_ts_maps
    ts_map.add_patch_polygons()
    assert "polygon" in ts_map.patches[patch_list[0]].keys()
    assert isinstance(ts_map.patches[patch_list[0]]["polygon"], Polygon)
    
def test_loader_save_to_geojson(init_ts_maps, tmp_path):
    _, ts_map, _, _ = init_ts_maps
    ts_map.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    assert os.path.exists(f"{tmp_path}/patches.geojson")
    geo_df = geopd.read_file(f"{tmp_path}/patches.geojson")
    assert "geometry" in geo_df.columns
    assert str(geo_df.crs.to_string()) == "EPSG:4326"
    assert isinstance(geo_df["geometry"][0], Polygon)