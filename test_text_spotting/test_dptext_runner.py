from __future__ import annotations

import os
import pathlib

import adet
import geopandas as gpd
import pandas as pd
import pytest
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances

from mapreader import DPTextDETRRunner
from mapreader.load import MapImages

print(adet.__version__)

# use cloned DPText-DETR path if running in github actions
ADET_PATH = (
    pathlib.Path("./DPText-DETR/").resolve()
    if os.getenv("GITHUB_ACTIONS") == "true"
    else pathlib.Path(adet.__path__[0]).resolve().parent
)


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
    maps = MapImages(f"{sample_dir}/mapreader_text.png")
    maps.add_metadata(f"{sample_dir}/mapreader_text_metadata.csv")
    maps.patchify_all(patch_size=800, path_save=tmp_path)
    maps.check_georeferencing()
    parent_df, patch_df = maps.convert_images()
    return parent_df, patch_df


@pytest.fixture
def init_runner(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{ADET_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    return runner


def test_dptext_init(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{ADET_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    assert isinstance(runner, DPTextDETRRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_dptext_init_str(init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    parent_df = parent_df.to_csv(f"{tmp_path}/parent_df.csv")
    patch_df = patch_df.to_csv(f"{tmp_path}/patch_df.csv")
    runner = DPTextDETRRunner(
        f"{tmp_path}/patch_df.csv",
        parent_df=f"{tmp_path}/parent_df.csv",
        cfg_file=f"{ADET_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    assert isinstance(runner, DPTextDETRRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_dptext_init_pathlib(init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    parent_df = parent_df.to_csv(f"{tmp_path}/parent_df.csv")
    patch_df = patch_df.to_csv(f"{tmp_path}/patch_df.csv")
    runner = DPTextDETRRunner(
        pathlib.Path(f"{tmp_path}/patch_df.csv"),
        parent_df=pathlib.Path(f"{tmp_path}/parent_df.csv"),
        cfg_file=f"{ADET_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    assert isinstance(runner, DPTextDETRRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_dptext_init_tsv(init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    parent_df = parent_df.to_csv(f"{tmp_path}/parent_df.tsv", sep="\t")
    patch_df = patch_df.to_csv(f"{tmp_path}/patch_df.tsv", sep="\t")
    runner = DPTextDETRRunner(
        f"{tmp_path}/patch_df.tsv",
        parent_df=f"{tmp_path}/parent_df.tsv",
        delimiter="\t",
        cfg_file=f"{ADET_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    assert isinstance(runner, DPTextDETRRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_dptext_run_all(init_runner):
    runner = init_runner
    # dict
    out = runner.run_all()
    assert isinstance(out, dict)
    assert "patch-0-0-800-40-#mapreader_text.png#.png" in out.keys()
    assert isinstance(out["patch-0-0-800-40-#mapreader_text.png#.png"], list)
    # dataframe
    runner.patch_predictions = {}
    out = runner.run_all(return_dataframe=True)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(["image_id", "geometry", "score"])
    assert "patch-0-0-800-40-#mapreader_text.png#.png" in out["image_id"].values


def test_dptext_convert_to_parent(init_runner):
    runner = init_runner
    _ = runner.run_all()
    # dict
    out = runner.convert_to_parent_pixel_bounds()
    assert isinstance(out, dict)
    assert "mapreader_text.png" in out.keys()
    assert isinstance(out["mapreader_text.png"], list)
    # dataframe
    runner.parent_predictions = {}
    out = runner.convert_to_parent_pixel_bounds(return_dataframe=True)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(["image_id", "patch_id", "geometry", "score"])
    assert "mapreader_text.png" in out["image_id"].values


def test_dptext_convert_to_parent_coords(init_runner):
    runner = init_runner
    _ = runner.run_all()
    # dict
    out = runner.convert_to_coords()
    assert isinstance(out, dict)
    assert "mapreader_text.png" in out.keys()
    assert isinstance(out["mapreader_text.png"], list)
    # dataframe
    runner.parent_predictions = {}
    out = runner.convert_to_coords(return_dataframe=True)
    assert isinstance(out, gpd.GeoDataFrame)
    assert set(out.columns) == set(["image_id", "patch_id", "geometry", "crs", "score"])
    assert "mapreader_text.png" in out["image_id"].values
    assert out.crs == runner.parent_df.crs


def test_dptext_deduplicate(sample_dir, tmp_path):
    maps = MapImages(f"{sample_dir}/mapreader_text.png")
    maps.add_metadata(f"{sample_dir}/mapreader_text_metadata.csv")
    maps.patchify_all(patch_size=800, path_save=tmp_path, overlap=0.5)
    maps.check_georeferencing()
    parent_df, patch_df = maps.convert_images()
    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{ADET_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    _ = runner.run_all()
    out = runner.convert_to_parent_pixel_bounds(deduplicate=False)
    len_before = len(out["mapreader_text.png"])
    runner.patch_predictions = {}
    out_07 = runner.convert_to_parent_pixel_bounds(deduplicate=True)
    len_07 = len(out_07["mapreader_text.png"])
    print(len_before, len_07)
    assert len_before >= len_07
    runner.patch_predictions = {}
    out_05 = runner.convert_to_parent_pixel_bounds(deduplicate=True, min_ioa=0.5)
    len_05 = len(out_05["mapreader_text.png"])
    print(len_before, len_05)
    assert len_before >= len_05
    assert len_07 >= len_05


def test_dptext_run_on_image(init_runner):
    runner = init_runner
    out = runner.run_on_image(
        runner.patch_df.iloc[0]["image_path"], return_outputs=True
    )
    assert isinstance(out, dict)
    assert "instances" in out.keys()
    assert isinstance(out["instances"], Instances)


def test_dptext_save_to_geojson(init_runner, tmp_path):
    runner = init_runner
    _ = runner.run_all()
    _ = runner.convert_to_coords()
    runner.save_to_geojson(f"{tmp_path}/text.geojson")
    assert os.path.exists(f"{tmp_path}/text.geojson")
    gdf = gpd.read_file(f"{tmp_path}/text.geojson")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert set(gdf.columns) == set(["image_id", "patch_id", "geometry", "crs", "score"])
