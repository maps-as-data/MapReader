from __future__ import annotations

import os
import pathlib
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from deepsolo.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances
from shapely import Polygon

from mapreader import DeepSoloRunner
from mapreader.load import MapImages
from mapreader.spot_text.dataclasses import (
    GeoPrediction,
    ParentPrediction,
    PatchPrediction,
)

# use cloned DeepSolo path if running in github actions
DEEPSOLO_PATH = (
    pathlib.Path("./DeepSolo/").resolve()
    if os.getenv("GITHUB_ACTIONS") == "true"
    else pathlib.Path(os.getenv("DEEPSOLO_PATH")).resolve()
)


@pytest.fixture
def sample_dir():
    return pathlib.Path(__file__).resolve().parent.parent / "sample_files"


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
    assert maps.georeferenced
    parent_df, patch_df = maps.convert_images()
    return parent_df, patch_df


@pytest.fixture(scope="function")
def mock_response(monkeypatch, sample_dir):
    def mock_pred(self, *args, **kwargs):
        with open(f"{sample_dir}/patch-0-0-800-40-deepsolo-pred.pkl", "rb") as f:
            outputs = pickle.load(f)
        return outputs

    monkeypatch.setattr(DefaultPredictor, "__call__", mock_pred)


@pytest.fixture
def init_runner(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = DeepSoloRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    return runner


@pytest.fixture
def runner_run_all(init_runner, mock_response):
    runner = init_runner
    _ = runner.run_all()
    return runner


def test_get_cfg():
    cfg = get_cfg()
    assert "TEMPERATURE" in cfg["MODEL"]["TRANSFORMER"].keys()


def test_deepsolo_init(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = DeepSoloRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    assert isinstance(runner, DeepSoloRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_deepsolo_init_str(init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    parent_df = parent_df.to_csv(f"{tmp_path}/parent_df.csv")
    patch_df = patch_df.to_csv(f"{tmp_path}/patch_df.csv")
    runner = DeepSoloRunner(
        f"{tmp_path}/patch_df.csv",
        parent_df=f"{tmp_path}/parent_df.csv",
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    assert isinstance(runner, DeepSoloRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_deepsolo_init_pathlib(init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    parent_df = parent_df.to_csv(f"{tmp_path}/parent_df.csv")
    patch_df = patch_df.to_csv(f"{tmp_path}/patch_df.csv")
    runner = DeepSoloRunner(
        pathlib.Path(f"{tmp_path}/patch_df.csv"),
        parent_df=pathlib.Path(f"{tmp_path}/parent_df.csv"),
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    assert isinstance(runner, DeepSoloRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_deepsolo_init_tsv(init_dataframes, tmp_path):
    parent_df, patch_df = init_dataframes
    parent_df = parent_df.to_csv(f"{tmp_path}/parent_df.tsv", sep="\t")
    patch_df = patch_df.to_csv(f"{tmp_path}/patch_df.tsv", sep="\t")
    runner = DeepSoloRunner(
        f"{tmp_path}/patch_df.tsv",
        parent_df=f"{tmp_path}/parent_df.tsv",
        delimiter="\t",
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    assert isinstance(runner, DeepSoloRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_deepsolo_init_geojson(init_dataframes, tmp_path, mock_response):
    parent_df, patch_df = init_dataframes
    parent_df.to_file(f"{tmp_path}/parent_df.geojson", driver="GeoJSON")
    patch_df.to_file(f"{tmp_path}/patch_df.geojson", driver="GeoJSON")
    runner = DeepSoloRunner(
        f"{tmp_path}/patch_df.geojson",
        parent_df=f"{tmp_path}/parent_df.geojson",
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    assert isinstance(runner, DeepSoloRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["geometry"], Polygon)
    out = runner.run_all()
    assert isinstance(out, dict)
    assert "patch-0-0-800-40-#mapreader_text.png#.png" in out.keys()
    assert isinstance(out["patch-0-0-800-40-#mapreader_text.png#.png"], list)
    assert isinstance(
        out["patch-0-0-800-40-#mapreader_text.png#.png"][0], PatchPrediction
    )


def test_deepsolo_init_errors(init_dataframes):
    parent_df, patch_df = init_dataframes
    with pytest.raises(ValueError, match="path to a CSV/TSV/etc or geojson"):
        DeepSoloRunner(
            patch_df="fake_file.txt",
            parent_df=parent_df,
            cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
        )
    with pytest.raises(ValueError, match="path to a CSV/TSV/etc or geojson"):
        DeepSoloRunner(
            patch_df=patch_df,
            parent_df="fake_file.txt",
            cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
        )
    with pytest.raises(ValueError, match="path to a CSV/TSV/etc or geojson"):
        DeepSoloRunner(
            patch_df=np.array([1, 2, 3]),
            parent_df=parent_df,
        )
    with pytest.raises(ValueError, match="path to a CSV/TSV/etc or geojson"):
        DeepSoloRunner(
            patch_df=patch_df,
            parent_df=np.array([1, 2, 3]),
        )


def test_check_georeferencing(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = DeepSoloRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    runner.check_georeferencing()
    assert runner.georeferenced

    runner = DeepSoloRunner(
        patch_df,
        parent_df=parent_df.drop(columns=["dlat", "dlon"]),
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    runner.check_georeferencing()
    assert runner.georeferenced

    runner = DeepSoloRunner(
        patch_df,
        parent_df=parent_df.drop(columns=["coordinates"]),
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    runner.check_georeferencing()
    assert not runner.georeferenced


def test_deepsolo_run_all(init_runner, mock_response):
    runner = init_runner
    # dict
    out = runner.run_all()
    assert isinstance(out, dict)
    assert "patch-0-0-800-40-#mapreader_text.png#.png" in out.keys()
    assert isinstance(out["patch-0-0-800-40-#mapreader_text.png#.png"], list)
    assert isinstance(
        out["patch-0-0-800-40-#mapreader_text.png#.png"][0], PatchPrediction
    )
    # dataframe
    out = runner._dict_to_dataframe(runner.patch_predictions)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(
        [
            "image_id",
            "pixel_geometry",
            "text",
            "score",
        ]
    )
    assert "patch-0-0-800-40-#mapreader_text.png#.png" in out["image_id"].values


def test_deepsolo_convert_to_parent(runner_run_all, mock_response):
    runner = runner_run_all
    # dict
    out = runner.convert_to_parent_pixel_bounds()
    assert isinstance(out, dict)
    assert "mapreader_text.png" in out.keys()
    assert isinstance(out["mapreader_text.png"], list)
    assert isinstance(out["mapreader_text.png"][0], ParentPrediction)
    # dataframe
    out = runner._dict_to_dataframe(runner.parent_predictions)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(
        ["image_id", "patch_id", "pixel_geometry", "text", "score"]
    )
    assert "mapreader_text.png" in out["image_id"].values


def test_deepsolo_convert_to_parent_coords(runner_run_all, mock_response):
    runner = runner_run_all
    # dict
    out = runner.convert_to_coords()
    assert isinstance(out, dict)
    assert "mapreader_text.png" in out.keys()
    assert isinstance(out["mapreader_text.png"], list)
    assert isinstance(out["mapreader_text.png"][0], GeoPrediction)
    # dataframe
    out = runner._dict_to_dataframe(runner.geo_predictions)
    assert isinstance(out, gpd.GeoDataFrame)
    assert set(out.columns) == set(
        ["image_id", "patch_id", "pixel_geometry", "geometry", "crs", "text", "score"]
    )
    assert "mapreader_text.png" in out["image_id"].values
    assert out.crs == runner.parent_df.crs


def test_deepsolo_deduplicate(sample_dir, tmp_path, mock_response):
    maps = MapImages(f"{sample_dir}/mapreader_text.png")
    maps.add_metadata(f"{sample_dir}/mapreader_text_metadata.csv")
    maps.patchify_all(patch_size=800, path_save=tmp_path, overlap=0.5)
    maps.check_georeferencing()
    parent_df, patch_df = maps.convert_images()
    runner = DeepSoloRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{DEEPSOLO_PATH}/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml",
    )
    _ = runner.run_all()
    out = runner.convert_to_parent_pixel_bounds(deduplicate=False)
    len_before = len(out["mapreader_text.png"])
    runner.parent_predictions = {}
    out_07 = runner.convert_to_parent_pixel_bounds(deduplicate=True)
    len_07 = len(out_07["mapreader_text.png"])
    print(len_before, len_07)
    assert len_before >= len_07
    runner.parent_predictions = {}
    out_05 = runner.convert_to_parent_pixel_bounds(deduplicate=True, min_ioa=0.5)
    len_05 = len(out_05["mapreader_text.png"])
    print(len_before, len_05)
    assert len_before >= len_05
    assert len_07 >= len_05


def test_deepsolo_run_on_image(init_runner, mock_response):
    runner = init_runner
    out = runner.run_on_image(
        runner.patch_df.iloc[0]["image_path"], return_outputs=True
    )
    assert isinstance(out, dict)
    assert "instances" in out.keys()
    assert isinstance(out["instances"], Instances)


def test_deepsolo_save_to_geojson(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    _ = runner.convert_to_coords()
    runner.save_to_geojson(f"{tmp_path}/text.geojson")
    assert os.path.exists(f"{tmp_path}/text.geojson")
    gdf = gpd.read_file(f"{tmp_path}/text.geojson")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert set(gdf.columns) == set(
        ["image_id", "patch_id", "pixel_geometry", "geometry", "crs", "text", "score"]
    )
    runner.save_to_geojson(f"{tmp_path}/text_centroid.geojson", centroid=True)
    assert os.path.exists(f"{tmp_path}/text_centroid.geojson")
    gdf_centroid = gpd.read_file(f"{tmp_path}/text_centroid.geojson")
    assert isinstance(gdf_centroid, gpd.GeoDataFrame)
    assert set(gdf_centroid.columns) == set(
        [
            "image_id",
            "patch_id",
            "pixel_geometry",
            "geometry",
            "crs",
            "text",
            "score",
            "polygon",
        ]
    )


def test_deepsolo_load_geo_predictions(runner_run_all, tmp_path):
    runner = runner_run_all
    _ = runner.convert_to_coords()
    runner.save_to_geojson(f"{tmp_path}/text.geojson")
    runner.geo_predictions = {}
    runner.load_geo_predictions(f"{tmp_path}/text.geojson")
    assert len(runner.geo_predictions)
    assert "mapreader_text.png" in runner.geo_predictions.keys()
    assert isinstance(runner.geo_predictions["mapreader_text.png"], list)
    assert isinstance(runner.geo_predictions["mapreader_text.png"][0], GeoPrediction)


def test_deepsolo_load_geo_predictions_errors(runner_run_all, tmp_path):
    runner = runner_run_all
    with pytest.raises(ValueError, match="must be a path to a geojson file"):
        runner.load_geo_predictions("fakefile.csv")


def test_deepsolo_save_to_csv_polygon(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    # patch
    runner.save_to_csv(tmp_path)
    assert os.path.exists(f"{tmp_path}/patch_predictions.csv")
    # parent
    _ = runner.convert_to_parent_pixel_bounds()
    runner.save_to_csv(tmp_path)
    assert os.path.exists(f"{tmp_path}/patch_predictions.csv")
    assert os.path.exists(f"{tmp_path}/parent_predictions.csv")
    # geo
    _ = runner.convert_to_coords()
    runner.save_to_csv(tmp_path)
    assert os.path.exists(f"{tmp_path}/patch_predictions.csv")
    assert os.path.exists(f"{tmp_path}/parent_predictions.csv")
    assert os.path.exists(f"{tmp_path}/geo_predictions.csv")


def test_deepsolo_save_to_csv_centroid(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    # patch
    runner.save_to_csv(tmp_path, centroid=True)
    assert os.path.exists(f"{tmp_path}/patch_predictions.csv")
    # parent
    _ = runner.convert_to_parent_pixel_bounds()
    runner.save_to_csv(tmp_path, centroid=True)
    assert os.path.exists(f"{tmp_path}/patch_predictions.csv")
    assert os.path.exists(f"{tmp_path}/parent_predictions.csv")
    # geo
    _ = runner.convert_to_coords()
    runner.save_to_csv(tmp_path, centroid=True)
    assert os.path.exists(f"{tmp_path}/patch_predictions.csv")
    assert os.path.exists(f"{tmp_path}/parent_predictions.csv")
    assert os.path.exists(f"{tmp_path}/geo_predictions.csv")


def test_deepsolo_save_to_csv_errors(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    runner.patch_predictions = {}
    with pytest.raises(ValueError, match="No patch predictions found"):
        runner.save_to_csv(tmp_path)


def test_deepsolo_load_patch_predictions(runner_run_all, tmp_path):
    runner = runner_run_all
    _ = runner.convert_to_coords()
    assert len(runner.geo_predictions)  # this will be empty after reloading
    runner.save_to_csv(tmp_path)
    runner.load_patch_predictions(f"{tmp_path}/patch_predictions.csv")
    assert len(runner.patch_predictions)
    assert len(runner.geo_predictions) == 0
    assert (
        "patch-0-0-800-40-#mapreader_text.png#.png" in runner.patch_predictions.keys()
    )
    assert isinstance(
        runner.patch_predictions["patch-0-0-800-40-#mapreader_text.png#.png"], list
    )
    assert isinstance(
        runner.patch_predictions["patch-0-0-800-40-#mapreader_text.png#.png"][0],
        PatchPrediction,
    )


def test_deepsolo_load_patch_predictions_dataframe(runner_run_all):
    runner = runner_run_all
    patch_preds = runner._dict_to_dataframe(runner.patch_predictions)
    _ = runner.convert_to_coords()
    assert len(runner.geo_predictions)  # this will be empty after reloading
    runner.load_patch_predictions(patch_preds)
    assert len(runner.patch_predictions)
    assert len(runner.geo_predictions) == 0
    assert (
        "patch-0-0-800-40-#mapreader_text.png#.png" in runner.patch_predictions.keys()
    )
    assert isinstance(
        runner.patch_predictions["patch-0-0-800-40-#mapreader_text.png#.png"], list
    )
    assert isinstance(
        runner.patch_predictions["patch-0-0-800-40-#mapreader_text.png#.png"][0],
        PatchPrediction,
    )


def test_deepsolo_load_patch_predictions_centroid(runner_run_all, tmp_path):
    runner = runner_run_all
    _ = runner.convert_to_coords()
    assert len(runner.geo_predictions)
    runner.save_to_csv(tmp_path, centroid=True)
    runner.load_patch_predictions(f"{tmp_path}/patch_predictions.csv")
    assert len(runner.patch_predictions)
    assert len(runner.geo_predictions) == 0
    assert (
        "patch-0-0-800-40-#mapreader_text.png#.png" in runner.patch_predictions.keys()
    )
    assert isinstance(
        runner.patch_predictions["patch-0-0-800-40-#mapreader_text.png#.png"], list
    )
    assert isinstance(
        runner.patch_predictions["patch-0-0-800-40-#mapreader_text.png#.png"][0],
        PatchPrediction,
    )


def test_deepsolo_load_patch_predictions_errors(runner_run_all, tmp_path):
    runner = runner_run_all
    with pytest.raises(
        ValueError, match="must be a pandas DataFrame or path to a CSV file"
    ):
        runner.load_patch_predictions("fake_file.geojson")


def test_deepsolo_search_preds(runner_run_all, mock_response):
    runner = runner_run_all
    _ = runner.convert_to_parent_pixel_bounds()
    out = runner.search_preds("map", ignore_case=True)
    assert isinstance(out, dict)
    assert "mapreader_text.png" in out.keys()
    # test dataframe
    out = runner.search_preds("map", ignore_case=True, return_dataframe=True)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(
        ["image_id", "patch_id", "pixel_geometry", "text", "score"]
    )
    assert "mapreader_text.png" in out["image_id"].values
    out = runner.search_preds("somethingelse", ignore_case=True, return_dataframe=True)
    assert len(out) == 0


def test_deepsolo_search_preds_errors(runner_run_all, mock_response):
    runner = runner_run_all
    with pytest.raises(ValueError, match="No parent predictions found"):
        runner.search_preds("maps", ignore_case=True)


def test_deepsolo_save_search_results(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    _ = runner.convert_to_parent_pixel_bounds()
    out = runner.search_preds("map", ignore_case=True)
    assert isinstance(out, dict)
    runner.save_search_results_to_geojson(f"{tmp_path}/search_results.geojson")
    assert os.path.exists(f"{tmp_path}/search_results.geojson")
    gdf = gpd.read_file(f"{tmp_path}/search_results.geojson")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert set(gdf.columns) == set(
        ["image_id", "patch_id", "pixel_geometry", "geometry", "crs", "text", "score"]
    )
    assert "mapreader_text.png" in gdf["image_id"].values


def test_deepsolo_save_search_results_errors(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    with pytest.raises(ValueError, match="No results to save"):
        runner.save_search_results_to_geojson(f"{tmp_path}/test.geojson")
