from __future__ import annotations

import pathlib

import geopandas as gpd
import pandas as pd
import pytest
from shapely import Polygon

from mapreader.load.images import MapImages
from mapreader.utils.load_frames import (
    eval_dataframe,
    get_geodataframe,
    get_load_function,
    load_from_csv,
    load_from_excel,
    load_from_geojson,
)


@pytest.fixture
def sample_dir():
    return pathlib.Path(__file__).resolve().parent / "sample_files"


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


def test_eval_dataframe(init_dataframes, tmp_path):
    parent_df, _ = init_dataframes
    parent_df.to_csv(f"{tmp_path}/parent_df.csv")
    parent_df_csv = pd.read_csv(f"{tmp_path}/parent_df.csv", index_col=0)
    assert isinstance(parent_df_csv.iloc[0]["shape"], str)
    assert isinstance(parent_df_csv.iloc[0]["geometry"], str)
    assert isinstance(parent_df_csv.iloc[0]["patches"], str)
    parent_df_csv = eval_dataframe(parent_df_csv)
    assert isinstance(parent_df_csv.iloc[0]["shape"], tuple)
    assert isinstance(parent_df_csv.iloc[0]["geometry"], str)  # still a string
    assert isinstance(parent_df_csv.iloc[0]["patches"], list)


def test_load_from_csv(init_dataframes, tmp_path):
    parent_df, _ = init_dataframes
    parent_df.rename(
        columns={"geometry": "polygon"}, inplace=True
    )  # rename geometry column to polygon for testing purposes
    parent_df.to_csv(f"{tmp_path}/parent_df.csv")
    parent_df.drop(columns=["polygon", "crs"]).to_csv(
        f"{tmp_path}/parent_df_no_geo.csv"
    )

    # load geo df
    parent_df_csv = pd.read_csv(
        f"{tmp_path}/parent_df.csv",
        index_col=0,
    )
    assert "polygon" in parent_df_csv.columns
    assert "geometry" not in parent_df_csv.columns
    assert isinstance(parent_df_csv.iloc[0]["shape"], str)
    assert isinstance(parent_df_csv.iloc[0]["polygon"], str)
    assert isinstance(parent_df_csv.iloc[0]["patches"], str)
    parent_df_csv = load_from_csv(f"{tmp_path}/parent_df.csv")
    assert "geometry" in parent_df_csv.columns
    assert "polygon" not in parent_df_csv.columns
    assert isinstance(parent_df_csv, gpd.GeoDataFrame)
    assert isinstance(parent_df_csv.iloc[0]["shape"], tuple)
    assert isinstance(
        parent_df_csv.iloc[0]["geometry"], Polygon
    )  # should be shapely Polygon
    assert isinstance(parent_df_csv.iloc[0]["patches"], list)

    # load non-geo df
    parent_df_csv = load_from_csv(f"{tmp_path}/parent_df_no_geo.csv")
    assert "geometry" not in parent_df_csv.columns
    assert "polygon" not in parent_df_csv.columns
    assert isinstance(parent_df_csv, pd.DataFrame)
    assert isinstance(parent_df_csv.iloc[0]["shape"], tuple)
    assert isinstance(parent_df_csv.iloc[0]["patches"], list)


def test_get_geodataframe(init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    parent_df = pd.DataFrame(parent_df)
    parent_df["crs"] = "EPSG:27700"  # change crs to EPSG:27700
    parent_df["geometry"] = parent_df["geometry"].apply(
        str
    )  # change geometry to string
    assert not isinstance(parent_df, gpd.GeoDataFrame)

    # convert to GeoDataFrame (with geom and crs)
    df = get_geodataframe(parent_df.copy())
    assert isinstance(df, gpd.GeoDataFrame)
    assert df.geometry.name == "geometry"
    assert df.crs.to_string() == "EPSG:27700"
    assert isinstance(df.iloc[0]["geometry"], Polygon)

    # convert to GeoDataFrame (with geom, but no crs)
    df = get_geodataframe(parent_df.drop(columns=["crs"]))
    assert isinstance(df, gpd.GeoDataFrame)
    assert df.geometry.name == "geometry"
    assert df.crs.to_string() == "EPSG:4326"  # default is EPSG:4326
    assert isinstance(df.iloc[0]["geometry"], Polygon)

    # convert to GeoDataFrame (multiple crs)
    patch_df = pd.DataFrame(patch_df)
    patch_df.at[0, "crs"] = "EPSG:27700"  # set one different
    assert patch_df["crs"].nunique() > 1
    assert not isinstance(patch_df, gpd.GeoDataFrame)
    with pytest.raises(ValueError, match="Multiple CRS found"):
        get_geodataframe(patch_df.copy())

    # return DataFrame (no geometry)
    df = get_geodataframe(parent_df.drop(columns=["geometry"]))
    assert not isinstance(df, gpd.GeoDataFrame)


def test_load_from_excel(init_dataframes, tmp_path):
    parent_df, _ = init_dataframes
    parent_df.rename(
        columns={"geometry": "polygon"}, inplace=True
    )  # rename geometry column to polygon for testing purposes
    parent_df.to_excel(f"{tmp_path}/parent_df.xlsx")
    parent_df.drop(columns=["polygon", "crs"]).to_excel(
        f"{tmp_path}/parent_df_no_geo.xlsx"
    )

    # load geo df
    parent_df_excel = pd.read_excel(f"{tmp_path}/parent_df.xlsx", index_col=0)
    assert "polygon" in parent_df_excel.columns
    assert "geometry" not in parent_df_excel.columns
    assert isinstance(parent_df_excel.iloc[0]["shape"], str)
    assert isinstance(parent_df_excel.iloc[0]["polygon"], str)
    assert isinstance(parent_df_excel.iloc[0]["patches"], str)
    parent_df_excel = load_from_excel(f"{tmp_path}/parent_df.xlsx")
    assert "geometry" in parent_df_excel.columns
    assert "polygon" not in parent_df_excel.columns
    assert isinstance(parent_df_excel, gpd.GeoDataFrame)
    assert isinstance(parent_df_excel.iloc[0]["shape"], tuple)
    assert isinstance(
        parent_df_excel.iloc[0]["geometry"], Polygon
    )  # should be shapely Polygon
    assert isinstance(parent_df_excel.iloc[0]["patches"], list)

    # load non-geo df
    parent_df_excel = load_from_excel(f"{tmp_path}/parent_df_no_geo.xlsx")
    assert "geometry" not in parent_df_excel.columns
    assert "polygon" not in parent_df_excel.columns
    assert isinstance(parent_df_excel, pd.DataFrame)
    assert isinstance(parent_df_excel.iloc[0]["shape"], tuple)
    assert isinstance(parent_df_excel.iloc[0]["patches"], list)


def test_load_from_geojson(init_dataframes, tmp_path):
    parent_df, _ = init_dataframes
    parent_df.to_file(
        f"{tmp_path}/parent_df.geojson", driver="GeoJSON", engine="pyogrio"
    )
    parent_df_geojson = gpd.read_file(f"{tmp_path}/parent_df.geojson", engine="pyogrio")
    assert parent_df_geojson.index == range(len(parent_df))  # should be numeric index
    assert "image_id" in parent_df_geojson.columns  # image_id should be in columns
    assert parent_df_geojson.geometry.name == "geometry"
    assert isinstance(parent_df_geojson.iloc[0]["shape"], str)
    assert isinstance(
        parent_df_geojson.iloc[0]["geometry"], Polygon
    )  # should be shapely Polygon (always)
    parent_df_geojson = load_from_geojson(f"{tmp_path}/parent_df.geojson")
    assert parent_df_geojson.index.name == "image_id"
    assert isinstance(parent_df_geojson.iloc[0].name, str)
    assert (
        "image_id" not in parent_df_geojson.columns
    )  # image_id should not be in columns, is now index
    assert isinstance(parent_df_geojson, gpd.GeoDataFrame)
    assert parent_df_geojson.geometry.name == "geometry"
    assert parent_df_geojson.crs == "EPSG:4326"
    assert isinstance(parent_df_geojson.iloc[0]["shape"], tuple)
    assert isinstance(
        parent_df_geojson.iloc[0]["geometry"], Polygon
    )  # should be shapely Polygon (always)


def test_get_load_function(sample_dir):
    assert (
        get_load_function(f"{sample_dir}/post_processing_patch_df.csv") == load_from_csv
    )
    assert get_load_function(f"{sample_dir}/ts_downloaded_maps.tsv") == load_from_csv
    assert get_load_function(f"{sample_dir}/ts_downloaded_maps.xlsx") == load_from_excel
    assert get_load_function(f"{sample_dir}/land_annots.geojson") == load_from_geojson
    with pytest.raises(ValueError):
        get_load_function(f"{sample_dir}/cropped_geo.tif")
    with pytest.raises(FileNotFoundError):
        get_load_function(f"{sample_dir}/non_existent_file.csv")
