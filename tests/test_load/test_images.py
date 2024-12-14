from __future__ import annotations

import os
import pathlib
from random import randint

import folium
import geopandas as gpd
import pandas as pd
import pytest
from PIL import Image
from pytest import approx
from shapely.geometry import Polygon

from mapreader.load.images import MapImages
from mapreader.utils.load_frames import load_from_csv, load_from_geojson


@pytest.fixture
def sample_dir():
    return pathlib.Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
def image_id():
    return "cropped_74488689.png"


@pytest.fixture
def init_maps(sample_dir, image_id, tmp_path):
    """Initializes MapImages object (with metadata from csv and patches).

    Returns
    -------
    tuple
        maps (MapImages object), parent_list (== image_id) and patch_list (list of patches).
    """
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=3, path_save=tmp_path)  # gives 9 patches
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()

    return maps, parent_list, patch_list


@pytest.fixture
def init_dataframes(sample_dir, image_id, tmp_path):
    """Initializes MapImages object (with metadata from csv and patches) and creates parent and patch dataframes.

    Returns
    -------
    tuple
        path to parent and patch dataframes
    """
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=3, path_save=tmp_path)  # gives 9 patches
    maps.add_center_coord(tree_level="parent")
    maps.add_patch_polygons()
    maps.calc_pixel_stats()
    _, _ = maps.convert_images(save=True)
    assert os.path.isfile("./parent_df.csv")
    assert os.path.isfile("./patch_df.csv")

    return "./parent_df.csv", "./patch_df.csv"


@pytest.fixture
def ts_metadata_keys():
    return [
        "name",
        "url",
        "coordinates",
        "crs",
        "published_date",
        "grid_bb",
        "geometry",  # polygon col renamed to geometry
    ]


# creating missing/matching/extra metadata


@pytest.fixture
def metadata_df():
    return pd.DataFrame(
        {
            "name": ["file1.png", "file2.png", "file3.png"],
            "coord": [(1.1, 1.5), (2.1, 1.0), (3.1, 4.5)],
            "other": [1, 2, 3],
        }
    )


@pytest.fixture
def metadata_keys():
    return ["parent_id", "image_path", "shape", "name", "coord", "other"]


@pytest.fixture
def matching_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png", "file3.png"]
    for file in files:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        Image.new("RGB", (9, 9), color=rand_color).save(f"{test_path}/{file}")
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
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        Image.new("RGB", (9, 9), color=rand_color).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep=",")
    return test_path


@pytest.fixture
def missing_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png", "file3.png", "file4.png"]
    for file in files:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        Image.new("RGB", (9, 9), color=rand_color).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep=",")
    return test_path


# --- test init ---


def test_init_png(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0
    assert isinstance(maps, MapImages)
    str(maps)
    len(maps)
    assert not maps.georeferenced


def test_init_png_grayscale(sample_dir):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0
    assert isinstance(maps, MapImages)
    maps.add_shape()
    assert maps.parents[image_id]["shape"] == (9, 9, 1)


def test_init_tiff(sample_dir):
    image_id = "cropped_non_geo.tif"
    tiffs = MapImages(f"{sample_dir}/{image_id}")
    assert len(tiffs) == 1
    assert isinstance(tiffs, MapImages)
    assert not tiffs.georeferenced


def test_init_geotiff(sample_dir):
    image_id = "cropped_geo.tif"
    geotiffs = MapImages(f"{sample_dir}/{image_id}")
    assert len(geotiffs) == 1
    assert isinstance(geotiffs, MapImages)
    assert geotiffs.georeferenced


def test_init_parent_path(sample_dir, image_id, capfd):
    maps = MapImages(
        f"{sample_dir}/{image_id}",
        tree_level="patch",
        parent_path=f"{sample_dir}/{image_id}",
    )
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 1

    # without passing tree level should get warning
    maps = MapImages(f"{sample_dir}/{image_id}", parent_path=f"{sample_dir}/{image_id}")
    out, _ = capfd.readouterr()
    assert (
        "[WARNING] Ignoring `parent_path` as `tree_level`  is set to 'parent'." in out
    )
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0


def test_init_tiff_32bit_error(sample_dir):
    image_id = "cropped_32bit.tif"
    with pytest.raises(NotImplementedError, match="Image mode"):
        MapImages(f"{sample_dir}/{image_id}")


def test_init_non_image_error(sample_dir):
    file_name = "ts_downloaded_maps.csv"
    with pytest.raises(ValueError, match="Non-image file types detected"):
        MapImages(f"{sample_dir}/{file_name}")


def test_init_fake_tree_level_error(sample_dir, image_id):
    with pytest.raises(ValueError, match="parent or patch"):
        MapImages(f"{sample_dir}/{image_id}", tree_level="fake")


# --- test ``add_metadata`` ---

# first test ``add_metadata`` works for png files


def test_add_metadata(sample_dir, image_id, ts_metadata_keys):
    # metadata csv
    maps_csv = MapImages(f"{sample_dir}/{image_id}")
    maps_csv.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")

    # metadata tsv
    maps_tsv = MapImages(f"{sample_dir}/{image_id}")
    maps_tsv.add_metadata(f"{sample_dir}/ts_downloaded_maps.tsv", delimiter="\t")

    # metadata xlsx
    maps_xlsx = MapImages(f"{sample_dir}/{image_id}")
    maps_xlsx.add_metadata(f"{sample_dir}/ts_downloaded_maps.xlsx")

    # metadata json
    maps_json = MapImages(f"{sample_dir}/{image_id}")
    maps_json.add_metadata(f"{sample_dir}/ts_downloaded_maps.geojson")

    for maps in [maps_csv, maps_tsv, maps_xlsx, maps_json]:
        assert all([k in maps.parents[image_id].keys() for k in ts_metadata_keys])
        assert isinstance(maps.parents[image_id]["coordinates"], tuple)
        assert maps.georeferenced
        assert maps.parents[image_id]["coordinates"] == approx(
            (-4.83, 55.80, -4.21, 56.059), rel=1e-2
        )


def test_add_metadata_pathlib(sample_dir, image_id, ts_metadata_keys):
    # metadata csv
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(pathlib.Path(f"{sample_dir}/ts_downloaded_maps.csv"))

    assert all([k in maps.parents[image_id].keys() for k in ts_metadata_keys])
    assert isinstance(maps.parents[image_id]["coordinates"], tuple)
    assert maps.georeferenced
    assert maps.parents[image_id]["coordinates"] == approx(
        (-4.83, 55.80, -4.21, 56.059), rel=1e-2
    )


def test_add_metadata_df(sample_dir, image_id, ts_metadata_keys):
    maps = MapImages(f"{sample_dir}/{image_id}")
    df = load_from_csv(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.add_metadata(df)

    assert all([k in maps.parents[image_id].keys() for k in ts_metadata_keys])
    assert isinstance(maps.parents[image_id]["coordinates"], tuple)
    assert maps.georeferenced
    assert maps.parents[image_id]["coordinates"] == approx(
        (-4.83, 55.80, -4.21, 56.059), rel=1e-2
    )


def test_add_metadata_geodf(sample_dir, image_id, ts_metadata_keys):
    maps = MapImages(f"{sample_dir}/{image_id}")
    gdf = load_from_geojson(f"{sample_dir}/ts_downloaded_maps.geojson")
    maps.add_metadata(gdf)

    assert all([k in maps.parents[image_id].keys() for k in ts_metadata_keys])
    assert isinstance(maps.parents[image_id]["coordinates"], tuple)
    assert maps.georeferenced
    assert maps.parents[image_id]["coordinates"] == approx(
        (-4.83, 55.80, -4.21, 56.059), rel=1e-2
    )


def test_add_metadata_errors(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    with pytest.raises(ValueError, match="a CSV/TSV/etc, Excel or JSON/GeoJSON file"):
        maps.add_metadata("fake.file")
    with pytest.raises(ValueError, match="file or a pandas DataFrame"):
        maps.add_metadata(123)


# check for mismatched metadata


def test_matching_metadata(matching_metadata_dir, metadata_df, metadata_keys):
    my_files_csv = MapImages(f"{matching_metadata_dir}/*png")
    my_files_csv.add_metadata(f"{matching_metadata_dir}/metadata_df.csv")

    my_files_xlsx = MapImages(f"{matching_metadata_dir}/*png")
    my_files_xlsx.add_metadata(f"{matching_metadata_dir}/metadata_df.xlsx")

    my_files_df = MapImages(f"{matching_metadata_dir}/*png")
    my_files_df.add_metadata(metadata_df)

    for my_files in [my_files_csv, my_files_xlsx, my_files_df]:
        for parent_id in my_files.list_parents():
            assert list(my_files.parents[parent_id].keys()) == metadata_keys


def test_missing_metadata_csv_ignore_mismatch(missing_metadata_dir, metadata_keys):
    my_files = MapImages(f"{missing_metadata_dir}/*png")
    assert len(my_files) == 4
    my_files.add_metadata(
        f"{missing_metadata_dir}/metadata_df.csv", ignore_mismatch=True
    )
    for parent_id in ["file1.png", "file2.png", "file3.png"]:
        assert list(my_files.parents[parent_id].keys()) == metadata_keys
    assert list(my_files.parents["file4.png"].keys()) == [
        "parent_id",
        "image_path",
        "shape",
    ]


def test_missing_metadata_csv_errors(missing_metadata_dir):
    my_files = MapImages(f"{missing_metadata_dir}/*png")
    assert len(my_files) == 4
    with pytest.raises(ValueError, match="missing information"):
        my_files.add_metadata(f"{missing_metadata_dir}/metadata_df.csv")


def test_extra_metadata_csv_ignore_mismatch(extra_metadata_dir, metadata_keys):
    my_files = MapImages(f"{extra_metadata_dir}/*png")
    assert len(my_files) == 2
    my_files.add_metadata(f"{extra_metadata_dir}/metadata_df.csv", ignore_mismatch=True)
    for parent_id in my_files.list_parents():
        assert list(my_files.parents[parent_id].keys()) == metadata_keys


def test_extra_metadata_csv_errors(extra_metadata_dir):
    my_files = MapImages(f"{extra_metadata_dir}/*png")
    assert len(my_files) == 2
    with pytest.raises(ValueError, match="information about non-existent images"):
        my_files.add_metadata(f"{extra_metadata_dir}/metadata_df.csv")


#  test other ``add_metadata`` args


def test_add_metadata_index_col(matching_metadata_dir):
    my_files = MapImages(f"{matching_metadata_dir}/*png")
    assert len(my_files) == 3
    my_files.add_metadata(
        f"{matching_metadata_dir}/metadata_df.csv", index_col="name"
    )  # pass index col arg
    metadata_keys = [
        "parent_id",
        "image_path",
        "shape",
        "Unnamed: 0",
        "coord",
        "other",
        "name",
    ]
    for parent_id in my_files.list_parents():
        assert list(my_files.parents[parent_id].keys()) == metadata_keys


def test_add_metadata_usecols(matching_metadata_dir, metadata_df):
    my_files_csv = MapImages(f"{matching_metadata_dir}/*png")
    my_files_csv.add_metadata(
        f"{matching_metadata_dir}/metadata_df.csv", usecols=["name", "coord"]
    )

    my_files_xlsx = MapImages(f"{matching_metadata_dir}/*png")
    my_files_xlsx.add_metadata(
        f"{matching_metadata_dir}/metadata_df.xlsx", usecols=["name", "coord"]
    )

    my_files_df = MapImages(f"{matching_metadata_dir}/*png")
    my_files_df.add_metadata(metadata_df, usecols=["name", "coord"])  # pass columns arg

    metadata_keys = set(["parent_id", "image_path", "shape", "name", "coord"])
    for my_files in [my_files_csv, my_files_xlsx, my_files_df]:
        for parent_id in my_files.list_parents():
            assert set(my_files.parents[parent_id].keys()) == metadata_keys
            assert isinstance(my_files.parents[parent_id]["coord"], tuple)


def test_add_metadata_parent(sample_dir, image_id, init_dataframes, ts_metadata_keys):
    parent_df, _ = init_dataframes
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(parent_df, tree_level="parent")
    assert all(
        [
            x in maps.parents[image_id].keys()
            for x in [*ts_metadata_keys, "center_lat", "center_lon"]
        ]
    )
    assert isinstance(maps.parents[image_id]["shape"], tuple)
    assert isinstance(maps.parents[image_id]["coordinates"], tuple)


def test_add_metadata_patch(sample_dir, image_id, init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.add_metadata(parent_df, tree_level="parent")  # add this too just in case
    maps.add_metadata(patch_df, tree_level="patch")
    patch_id = maps.list_patches()[0]
    expected_cols = [
        "parent_id",
        "shape",
        "pixel_bounds",
        "coordinates",
        "geometry",
        "mean_pixel_R",
        "mean_pixel_A",
        "std_pixel_R",
        "std_pixel_A",
    ]
    assert all([x in maps.patches[patch_id].keys() for x in expected_cols])
    for k in ["shape", "pixel_bounds", "coordinates"]:
        assert isinstance(maps.patches[patch_id][k], tuple)
    assert isinstance(
        "geometry", str
    )  # expect this to be a string, reformed into polygon later
    assert maps.georeferenced


def test_add_metadata_polygons(sample_dir, image_id, ts_metadata_keys):
    maps = MapImages(f"{sample_dir}/{image_id}")
    gdf = load_from_geojson(f"{sample_dir}/ts_downloaded_maps.geojson")
    gdf.rename(columns={"geometry": "polygon"}, inplace=True)
    maps.add_metadata(gdf)

    assert all([k in maps.parents[image_id].keys() for k in ts_metadata_keys])
    assert isinstance(maps.parents[image_id]["coordinates"], tuple)
    assert maps.georeferenced
    assert maps.parents[image_id]["coordinates"] == approx(
        (-4.83, 55.80, -4.21, 56.059), rel=1e-2
    )


# other ``add_metadata`` errors


def test_metadata_not_found(matching_metadata_dir):
    my_files = MapImages(f"{matching_metadata_dir}/*png")
    assert len(my_files) == 3
    with pytest.raises(FileNotFoundError):
        my_files.add_metadata(f"{matching_metadata_dir}/fakefile.csv")


def test_metadata_missing_name_or_image_id(matching_metadata_dir):
    my_files = MapImages(f"{matching_metadata_dir}/*png")
    assert len(my_files) == 3
    incomplete_metadata_df = pd.DataFrame(
        {"coord": [(1.1, 1.5), (2.1, 1.0), (3.1, 4.5)], "other": [1, 2, 3]}
    )
    incomplete_metadata_df.to_csv(
        f"{matching_metadata_dir}/incomplete_metadata_df.csv", sep=","
    )
    with pytest.raises(
        ValueError, match="'name' or 'image_id' should be one of the columns"
    ):
        my_files.add_metadata(incomplete_metadata_df)
    with pytest.raises(
        ValueError, match="'name' or 'image_id' should be one of the columns"
    ):
        my_files.add_metadata(f"{matching_metadata_dir}/incomplete_metadata_df.csv")


# --- test ``add_geo_info`` ---


def test_loader_add_geo_info(sample_dir):
    # check it works for geotiff
    image_id = "cropped_geo.tif"
    geotiffs = MapImages(f"{sample_dir}/{image_id}")
    geotiffs.add_geo_info()
    assert all(
        [k in geotiffs.parents[image_id].keys() for k in ["shape", "coordinates"]]
    )
    assert geotiffs.parents[image_id]["coordinates"] == approx(
        (-0.061, 51.6142, -0.0610, 51.614), rel=1e-2
    )
    assert geotiffs.georeferenced

    # check nothing happens for png/tiff (no metadata)
    image_id = "cropped_74488689.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    keys = list(maps.parents[image_id].keys())
    maps.add_geo_info()
    assert list(maps.parents[image_id].keys()) == keys

    image_id = "cropped_non_geo.tif"
    tiff = MapImages(f"{sample_dir}/{image_id}")
    keys = list(tiff.parents[image_id].keys())
    tiff.add_geo_info()
    assert list(tiff.parents[image_id].keys()) == keys
    assert not tiff.georeferenced


# --- test patchify ---


def test_patchify_pixels(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    assert len(parent_list) == 1
    assert len(patch_list) == 9
    assert os.path.isfile(f"{tmp_path}/patch-0-0-3-3-#{image_id}#.png")


def test_patchify_pixels_skip_blank_rgb(sample_dir, tmp_path):
    maps = MapImages(f"{sample_dir}/cropped_blank_corners_rgb.tif")
    maps.patchify_all(patch_size=3, path_save=tmp_path, skip_blank_patches=True)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    assert len(parent_list) == 1
    assert len(patch_list) == 5
    assert os.path.isfile(f"{tmp_path}/patch-0-3-3-6-#cropped_blank_corners_rgb.tif#.png")


def test_patchify_pixels_skip_blank_rgba(sample_dir, tmp_path):
    maps = MapImages(f"{sample_dir}/cropped_blank_corners_rgba.tif")
    maps.patchify_all(patch_size=3, path_save=tmp_path, skip_blank_patches=True)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    assert len(parent_list) == 1
    assert len(patch_list) == 5
    assert os.path.isfile(f"{tmp_path}/patch-0-3-3-6-#cropped_blank_corners_rgba.tif#.png")


def test_patchify_pixels_square(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=5, path_save=f"{tmp_path}_square", square_cuts=True)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    print(patch_list, flush=True)
    assert len(parent_list) == 1
    assert len(patch_list) == 4
    assert os.path.isfile(f"{tmp_path}_square/patch-0-0-5-5-#{image_id}#.png")
    assert os.path.isfile(f"{tmp_path}_square/patch-4-4-9-9-#{image_id}#.png")


def test_patchify_meters(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=10000, method="meters", path_save=f"{tmp_path}_meters")
    assert os.path.isfile(f"{tmp_path}_meters/patch-0-0-2-2-#{image_id}#.png")
    assert len(maps.list_patches()) == 25


def test_patchify_grayscale(sample_dir, tmp_path):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    assert len(parent_list) == 1
    assert len(patch_list) == 9
    assert os.path.isfile(f"{tmp_path}/patch-0-0-3-3-#{image_id}#.png")


def test_patchify_meters_errors(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    with pytest.raises(ValueError, match="add coordinate information"):
        maps.patchify_all(patch_size=10000, method="meters", path_save=tmp_path)


def test_patchify_pixels_overlap(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=4, path_save=tmp_path, overlap=0.5)
    parent_list = maps.list_parents()
    patch_list = maps.list_patches()
    print(patch_list, flush=True)
    assert len(parent_list) == 1
    assert len(patch_list) == 25
    assert os.path.isfile(f"{tmp_path}/patch-0-0-4-4-#{image_id}#.png")
    assert os.path.isfile(f"{tmp_path}/patch-2-2-6-6-#{image_id}#.png")
    assert os.path.isfile(f"{tmp_path}/patch-6-6-9-9-#{image_id}#.png")
    assert os.path.isfile(f"{tmp_path}/patch-8-8-9-9-#{image_id}#.png")


# --- test other functions ---


def test_check_georeferencing(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.check_georeferencing()
    assert maps.georeferenced is False  # no metadata yet

    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv",
        usecols=["name", "url", "coordinates", "crs", "published_date", "grid_bb"],
    )
    assert maps.georeferenced is True  # have added georeferencing via coordinates col
    assert (
        "geometry" in maps.parents["cropped_74488689.png"].keys()
    )  # should also have added polygon
    maps.patchify_all(patch_size=3, path_save=f"{tmp_path}/patches_3_pixel")
    parent_df, patch_df = maps.convert_images()  # save these for loading later
    assert (
        "coordinates" in maps.patches["patch-0-0-3-3-#cropped_74488689.png#.png"].keys()
    )
    assert "geometry" in maps.patches["patch-0-0-3-3-#cropped_74488689.png#.png"].keys()

    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv",
        usecols=["name", "url", "crs", "published_date", "grid_bb"],
    )
    assert maps.georeferenced is True  # have added georeferencing from grid_bb
    assert (
        "geometry" in maps.parents["cropped_74488689.png"].keys()
    )  # should also have added polygon

    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv",
        usecols=["name", "url", "published_date", "grid_bb"],
    )
    assert maps.georeferenced is True  # have added georeferencing from grid_bb
    assert (
        "geometry" in maps.parents["cropped_74488689.png"].keys()
    )  # should also have added polygon

    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv",
        usecols=["name", "url", "published_date"],
    )
    maps.load_patches(patch_paths=f"{tmp_path}/patches_3_pixel/*png")
    assert len(maps.patches) == 9
    assert maps.georeferenced is False  # still no georeferencing info
    maps.add_metadata(patch_df, tree_level="patch")
    assert maps.georeferenced is True  # now have georeferencing info
    assert "coordinates" in maps.parents["cropped_74488689.png"].keys()
    assert "geometry" in maps.parents["cropped_74488689.png"].keys()
    assert (
        "coordinates" in maps.patches["patch-0-0-3-3-#cropped_74488689.png#.png"].keys()
    )
    assert "geometry" in maps.patches["patch-0-0-3-3-#cropped_74488689.png#.png"].keys()


def test_load_patches(init_maps, sample_dir, tmp_path):
    maps, _, _ = init_maps

    # create tiff patches
    geotiff_path = f"{sample_dir}/cropped_geo.tif"
    tiff_maps = MapImages(geotiff_path)
    tiff_maps.add_geo_info()
    assert tiff_maps.georeferenced
    tiff_maps.patchify_all(patch_size=3, path_save=f"{tmp_path}_tiffs")

    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 9
    maps.load_patches(f"{tmp_path}_tiffs", parent_paths=geotiff_path, add_geo_info=True)
    assert "coordinates" in maps.parents["cropped_geo.tif"].keys()
    assert maps.georeferenced
    assert len(maps.list_parents()) == 2
    assert len(maps.list_patches()) == 18  # 9 for each

    maps.load_patches(f"{tmp_path}_tiffs", parent_paths=geotiff_path, clear_images=True)
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 9
    assert not maps.georeferenced


def test_infer_coords_from_patches(sample_dir, tmp_path):
    # create tiff patches
    geotiff_path = f"{sample_dir}/cropped_geo.tif"
    tiff_maps = MapImages(geotiff_path)
    tiff_maps.add_geo_info()
    assert tiff_maps.georeferenced
    tiff_maps.patchify_all(patch_size=3, path_save=f"{tmp_path}_tiffs")
    parent_df, patch_df = tiff_maps.convert_images()

    maps = MapImages()  # create new instance
    maps.load_patches(f"{tmp_path}_tiffs", parent_paths=geotiff_path, add_geo_info=True)
    maps.check_georeferencing()
    assert (
        "coordinates" in maps.patches["patch-0-0-3-3-#cropped_geo.tif#.png"].keys()
    )  # need coordinates in patches
    assert "coordinates" in maps.parents["cropped_geo.tif"].keys()
    assert maps.georeferenced
    maps.parents["cropped_geo.tif"].pop("coordinates")
    maps.infer_parent_coords_from_patches()
    assert "coordinates" in maps.parents["cropped_geo.tif"].keys()
    maps.check_georeferencing()
    assert maps.georeferenced


def test_load_parents(init_maps, image_id, sample_dir):
    maps, _, _ = init_maps

    geotiff_path = f"{sample_dir}/cropped_geo.tif"
    assert len(maps.list_parents()) == 1
    maps.load_parents(geotiff_path, overwrite=False, add_geo_info=True)
    assert len(maps.list_parents()) == 2
    assert all(map in maps.list_parents() for map in [image_id, "cropped_geo.tif"])
    assert "coordinates" in maps.parents["cropped_geo.tif"].keys()
    assert maps.georeferenced

    tiff_path = f"{sample_dir}/cropped_non_geo.tif"
    maps.load_parents(tiff_path, overwrite=True)
    assert len(maps.list_parents()) == 1
    assert maps.list_parents() == ["cropped_non_geo.tif"]
    assert not maps.georeferenced


def test_add_shape(init_maps, image_id):
    maps, _, patch_list = init_maps

    maps.parents[image_id].pop("shape")
    assert "shape" not in maps.parents[image_id].keys()
    maps.add_shape(tree_level="parent")
    assert "shape" in maps.parents[image_id].keys()

    maps.patches[patch_list[0]].pop("shape")
    assert "shape" not in maps.patches[patch_list[0]].keys()
    maps.add_shape(tree_level="patch")
    assert "shape" in maps.patches[patch_list[0]].keys()


def test_calc_coords_from_grid_bb(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv", usecols=["name", "grid_bb", "crs"]
    )
    for parent in maps.list_parents():
        maps.parents[parent].pop("coordinates")  # remove coordinates so we can test
    assert "coordinates" not in maps.parents[image_id]
    maps.add_coords_from_grid_bb()
    assert "coordinates" in maps.parents[image_id]
    assert maps.parents[image_id]["coordinates"] == approx(
        (-4.83, 55.80, -4.21, 56.059), rel=1e-2
    )
    assert maps.georeferenced


def test_calc_coords_from_grid_bb_error(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    assert all([x not in maps.parents[image_id] for x in ["coordinates", "grid_bb"]])
    with pytest.raises(ValueError, match="No grid bounding box"):
        maps.add_coords_from_grid_bb()


def test_calc_coords_from_grid_bb_format_error(sample_dir, image_id, capfd):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv", usecols=["name", "grid_bb", "crs"]
    )
    for parent in maps.list_parents():
        maps.parents[parent].pop("coordinates")  # remove coordinates so we can test
    assert "coordinates" not in maps.parents[image_id]
    maps.parents[image_id]["grid_bb"] = 123  # change to some other format/string
    with pytest.raises(ValueError, match="Unexpected grid_bb"):
        maps.add_coords_from_grid_bb()
    maps.parents[image_id]["grid_bb"] = "17,0,0"  # change to some other format/string
    with pytest.raises(ValueError, match="Unexpected grid_bb"):
        maps.add_coords_from_grid_bb()


def test_coord_functions(init_maps, image_id, sample_dir, capfd):
    # test for png with added metadata
    maps, _, patch_list = init_maps
    maps.add_center_coord()
    assert "dlon" in maps.parents[image_id].keys()
    assert "center_lon" in maps.patches[patch_list[0]].keys()

    # test for geotiff with added geoinfo
    image_id = "cropped_geo.tif"
    geotiffs = MapImages(f"{sample_dir}/{image_id}")
    geotiffs.add_geo_info()
    geotiffs.add_coord_increments()
    geotiffs.add_center_coord(tree_level="parent")
    assert "dlon" in geotiffs.parents[image_id].keys()
    assert "center_lon" in geotiffs.parents[image_id].keys()


def test_coord_functions_errors(sample_dir, image_id, tmp_path):
    # test for tiff with no geo info, no coordinates so should raise error
    tiffs = MapImages(f"{sample_dir}/cropped_non_geo.tif")
    with pytest.raises(ValueError, match="No coordinates"):
        tiffs.add_coord_increments()
    with pytest.raises(ValueError, match="'coordinates' could not be found"):
        tiffs.add_center_coord(tree_level="parent")
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    with pytest.raises(ValueError, match="'coordinates' could not be found"):
        maps.add_center_coord(tree_level="patch")


def test_add_patch_coords(init_maps):
    maps, _, patch_list = init_maps
    maps.patches[patch_list[0]].pop("coordinates")
    assert "coordinates" not in maps.patches[patch_list[0]].keys()
    maps.add_patch_coords()
    assert "coordinates" in maps.patches[patch_list[0]].keys()


def test_add_patch_polygons(init_maps):
    maps, _, patch_list = init_maps
    maps.patches[patch_list[0]].pop("geometry")
    assert "geometry" not in maps.patches[patch_list[0]].keys()
    maps.add_patch_polygons()
    assert "geometry" in maps.patches[patch_list[0]].keys()
    assert isinstance(maps.patches[patch_list[0]]["geometry"], Polygon)


def test_add_parent_polygons(init_maps):
    maps, parent_list, _ = init_maps
    for parent in parent_list:
        maps.parents[parent].pop("geometry")
    assert "geometry" not in maps.parents[parent_list[0]].keys()
    maps.add_parent_polygons()
    assert "geometry" in maps.parents[parent_list[0]].keys()
    assert isinstance(maps.parents[parent_list[0]]["geometry"], Polygon)
    maps.check_georeferencing()
    assert maps.georeferenced


def test_add_parent_polygons_errors(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    with pytest.raises(ValueError, match="No coordinate information found"):
        maps.add_parent_polygons()


def test_save_patches_as_geotiffs(init_maps):
    maps, _, _ = init_maps
    maps.save_patches_as_geotiffs()
    patch_id = maps.list_patches()[0]
    assert "geotiff_path" in maps.patches[patch_id].keys()
    assert os.path.isfile(maps.patches[patch_id]["geotiff_path"])


def test_save_patches_as_geotiffs_edge_patches(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    maps.patchify_all(patch_size=8, path_save=tmp_path)
    patch_list = maps.list_patches()
    assert len(maps.list_patches()) == 4
    maps.save_patches_as_geotiffs()
    for patch in patch_list:
        pixel_bounds = maps.patches[patch]["pixel_bounds"]
        patch_width = pixel_bounds[2] - pixel_bounds[0]
        patch_height = pixel_bounds[3] - pixel_bounds[1]
        assert "geotiff_path" in maps.patches[patch].keys()
        assert os.path.isfile(maps.patches[patch]["geotiff_path"])
        img = Image.open(maps.patches[patch]["geotiff_path"])
        assert img.height == patch_height
        assert img.width == patch_width


def test_save_patches_as_geotiffs_grayscale(sample_dir, tmp_path):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    metadata = pd.read_csv(f"{sample_dir}/ts_downloaded_maps.csv", index_col=0)
    metadata.loc[0, "name"] = "cropped_L.png"
    maps.add_metadata(metadata)
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.save_patches_as_geotiffs()
    patch_id = maps.list_patches()[0]
    assert "geotiff_path" in maps.patches[patch_id].keys()
    assert os.path.isfile(maps.patches[patch_id]["geotiff_path"])


def test_save_to_geojson(init_maps, tmp_path, capfd):
    maps, _, _ = init_maps
    maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    assert os.path.exists(f"{tmp_path}/patches.geojson")
    geo_df = gpd.read_file(f"{tmp_path}/patches.geojson", engine="pyogrio")
    assert "geometry" in geo_df.columns
    assert str(geo_df.crs.to_string()) == "EPSG:4326"
    assert isinstance(geo_df["geometry"][0], Polygon)

    maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    out, _ = capfd.readouterr()
    assert "[WARNING] File already exists" in out


def test_save_to_geojson_error(sample_dir, image_id, tmp_path, capfd):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    with pytest.raises(ValueError, match="No patches"):
        maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    # remove coordinates
    for parent in maps.list_parents():
        maps.parents[parent].pop("geometry")  # remove geometry
        maps.parents[parent].pop(
            "coordinates"
        )  # remove coordinates (for add_parent_polygons)
        maps.parents[parent].pop(
            "grid_bb"
        )  # remove grid_bb (for add_coords_from_grid_bb)
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.check_georeferencing()
    assert not maps.georeferenced
    with pytest.raises(ValueError, match="No geographic information"):
        maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")


def test_save_to_geojson_missing_data(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.add_metadata(
        f"{sample_dir}/ts_downloaded_maps.csv", usecols=["name", "coordinates", "crs"]
    )
    maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    assert os.path.exists(f"{tmp_path}/patches.geojson")
    geo_df = gpd.read_file(f"{tmp_path}/patches.geojson", engine="pyogrio")
    assert "geometry" in geo_df.columns
    assert str(geo_df.crs.to_string()) == "EPSG:4326"
    assert isinstance(geo_df["geometry"][0], Polygon)


def test_save_to_geojson_polygon_strings(
    sample_dir, image_id, init_dataframes, tmp_path
):
    parent_df, patch_df = init_dataframes
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    maps.add_metadata(parent_df, tree_level="parent")
    maps.add_metadata(patch_df, tree_level="patch")
    patch_id = maps.list_patches()[0]
    assert isinstance(
        maps.patches[patch_id]["geometry"], Polygon
    )  # should be a polygon
    maps.save_patches_to_geojson(geojson_fname=f"{tmp_path}/patches.geojson")
    assert os.path.exists(f"{tmp_path}/patches.geojson")
    geo_df = gpd.read_file(f"{tmp_path}/patches.geojson", engine="pyogrio")
    assert "geometry" in geo_df.columns
    assert str(geo_df.crs.to_string()) == "EPSG:4326"
    assert isinstance(geo_df["geometry"][0], Polygon)  # should be a polygon


def test_calc_pixel_stats(init_maps, sample_dir, tmp_path):
    maps, _, patch_list = init_maps
    maps.calc_pixel_stats()
    expected_cols = [
        "mean_pixel_R",
        "mean_pixel_G",
        "mean_pixel_B",
        "mean_pixel_A",
        "std_pixel_R",
        "std_pixel_G",
        "std_pixel_B",
        "std_pixel_A",
    ]
    assert all([col in maps.patches[patch_list[0]].keys() for col in expected_cols])

    # geotiff/tiff will not have alpha channel, so only RGB returned
    image_id = "cropped_geo.tif"
    geotiffs = MapImages(f"{sample_dir}/{image_id}")
    geotiffs.patchify_all(patch_size=3, path_save=tmp_path)
    patch_list = geotiffs.list_patches()
    geotiffs.calc_pixel_stats()
    expected_cols = [
        "mean_pixel_R",
        "mean_pixel_G",
        "mean_pixel_B",
        "std_pixel_R",
        "std_pixel_G",
        "std_pixel_B",
    ]
    assert all([col in geotiffs.patches[patch_list[0]].keys() for col in expected_cols])


def test_loader_convert_images(init_maps):
    maps, _, _ = init_maps
    parent_df, patch_df = maps.convert_images()
    assert parent_df.shape == (1, 13)
    assert patch_df.shape == (9, 7)
    parent_df, patch_df = maps.convert_images(save=True)
    assert os.path.isfile("./parent_df.csv")
    assert os.path.isfile("./patch_df.csv")
    os.remove("./parent_df.csv")
    os.remove("./patch_df.csv")
    parent_df, patch_df = maps.convert_images(save=True, save_format="excel")
    assert os.path.isfile("./parent_df.xlsx")
    assert os.path.isfile("./patch_df.xlsx")
    os.remove("./parent_df.xlsx")
    os.remove("./patch_df.xlsx")
    assert maps.georeferenced
    parent_df, patch_df = maps.convert_images(save=True, save_format="geojson")
    assert os.path.isfile("./parent_df.geojson")
    assert os.path.isfile("./patch_df.geojson")
    os.remove("./parent_df.geojson")
    os.remove("./patch_df.geojson")


def test_loader_convert_images_geojson_errors(sample_dir, image_id, tmp_path):
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    assert not maps.georeferenced
    with pytest.raises(ValueError, match="no coordinate information found"):
        maps.convert_images(save=True, save_format="geojson")


def test_loader_convert_images_errors(init_maps):
    maps, _, _ = init_maps
    with pytest.raises(ValueError, match="``save_format`` should be one of"):
        maps.convert_images(save=True, save_format="fakeformat")


def test_save_parents_as_geotiffs(init_maps, sample_dir, image_id):
    maps, _, _ = init_maps
    maps.save_parents_as_geotiffs()
    image_id = image_id.split(".")[0]
    assert os.path.isfile(f"{sample_dir}/{image_id}.tif")


def test_save_parents_as_geotiffs_grayscale(sample_dir, tmp_path):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    metadata = pd.read_csv(f"{sample_dir}/ts_downloaded_maps.csv", index_col=0)
    metadata.loc[0, "name"] = "cropped_L.png"
    maps.add_metadata(metadata)
    maps.save_parents_as_geotiffs()
    assert "geotiff_path" in maps.parents[image_id].keys()
    assert os.path.isfile(maps.parents[image_id]["geotiff_path"])


def test_save_parents_as_geotiffs_error(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    assert "coordinates" not in maps.parents[image_id].keys()
    with pytest.raises(ValueError, match="Cannot locate coordinates"):
        maps.save_parents_as_geotiffs(rewrite=True)


def test_show_sample_png(init_maps, monkeypatch):
    maps, parent_list, patch_list = init_maps
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    maps.show_sample(num_samples=1, tree_level="parent")
    maps.show_sample(num_samples=1, tree_level="patch")


def test_show_sample_grayscale(sample_dir, tmp_path, monkeypatch):
    image_id = "cropped_L.png"
    maps = MapImages(f"{sample_dir}/{image_id}")
    maps.patchify_all(patch_size=3, path_save=tmp_path)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    maps.show_sample(num_samples=1, tree_level="parent")
    maps.show_sample(num_samples=1, tree_level="patch")


def test_explore_patches(init_maps):
    maps, parent_list, _ = init_maps
    out = maps.explore_patches(parent_list[0])
    assert isinstance(out, folium.Map)
    assert [*out.to_dict()["children"].keys()][
        0
    ] == "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

    # using tilelayer
    out = maps.explore_patches(
        parent_list[0], xyz_url="https://tile.opentopomap.org/{z}/{x}/{y}.png"
    )
    assert isinstance(out, folium.Map)
    assert [*out.to_dict()["children"].keys()][
        0
    ] == "https://tile.opentopomap.org/{z}/{x}/{y}.png"


def test_explore_patches_errors(init_maps):
    maps, parent_list, _ = init_maps
    maps.georeferenced = False  # set to False to test error
    with pytest.raises(
        NotImplementedError, match="only works with georeferenced images"
    ):
        maps.explore_patches(parent_list[0])
    maps.check_georeferencing()
    with pytest.raises(ValueError, match="not found in parent list"):
        maps.explore_patches("fake_id")


def test_load_parents_errors(sample_dir, image_id):
    maps = MapImages(f"{sample_dir}/{image_id}")
    with pytest.raises(ValueError, match="Please pass one of"):
        maps.load_parents()


def test_load_df(init_dataframes, image_id):
    # set up
    parent_df, patch_df = init_dataframes
    parent_df = load_from_csv(parent_df)
    parent_df.rename(columns={"geometry": "polygon"}, inplace=True)
    assert "polygon" in parent_df.columns
    patch_df = load_from_csv(patch_df)
    patch_df.rename(columns={"geometry": "polygon"}, inplace=True)
    assert "polygon" in patch_df.columns

    maps = MapImages()
    # test loading parent df
    maps.load_df(parent_df=parent_df)
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0
    assert maps.georeferenced
    assert "geometry" in maps.parents[image_id].keys()
    # test loading patch df
    maps.load_df(patch_df=patch_df, clear_images=False)
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 9
    maps.check_georeferencing()
    assert maps.georeferenced
    assert "geometry" in maps.parents[image_id].keys()
    assert "geometry" in maps.patches[maps.list_patches()[0]].keys()
    # test clear images
    maps.load_df(parent_df=parent_df, clear_images=True)
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0


def test_load_csv(init_dataframes, image_id):
    # set up
    patch_df, parent_df = init_dataframes
    parent_df = load_from_csv(parent_df)
    parent_df.rename(columns={"geometry": "polygon"}, inplace=True)
    assert "polygon" in parent_df.columns
    parent_df.to_csv("./patch_df.csv")  # save to csv again
    patch_df = load_from_csv(patch_df)
    patch_df.rename(columns={"geometry": "polygon"}, inplace=True)
    assert "polygon" in patch_df.columns
    patch_df.to_csv("./parent_df.csv")  # save to csv again

    maps = MapImages()
    # test loading parent df
    maps.load_csv(parent_path="./parent_df.csv")
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0
    assert maps.georeferenced
    assert "geometry" in maps.parents[image_id].keys()
    # test loading patch df
    maps.load_csv(patch_path="./patch_df.csv")
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 9
    assert maps.georeferenced
    assert "geometry" in maps.parents[image_id].keys()
    assert "geometry" in maps.patches[maps.list_patches()[0]].keys()
    # test clear images
    maps.load_csv(parent_path="./parent_df.csv", clear_images=True)
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0
    # test pathlib
    maps.load_csv(parent_path=pathlib.Path("./parent_df.csv"), clear_images=True)
    assert len(maps.list_parents()) == 1
    assert len(maps.list_patches()) == 0
