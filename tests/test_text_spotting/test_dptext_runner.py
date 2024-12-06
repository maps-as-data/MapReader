from __future__ import annotations

import os
import pathlib
import pickle

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances
from dptext_detr.config import get_cfg
from shapely import Polygon

from mapreader import DPTextDETRRunner
from mapreader.load import MapImages
from mapreader.spot_text.dataclasses import (
    GeoPrediction,
    ParentPrediction,
    PatchPrediction,
)

# use cloned DPText-DETR path if running in github actions
DPTEXT_DETR_PATH = (
    pathlib.Path("./DPText-DETR/").resolve()
    if os.getenv("GITHUB_ACTIONS") == "true"
    else pathlib.Path(os.getenv("DPTEXT_DETR_PATH")).resolve()
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
        with open(f"{sample_dir}/patch-0-0-800-40-dptext-detr-pred.pkl", "rb") as f:
            outputs = pickle.load(f)
        return outputs

    monkeypatch.setattr(DefaultPredictor, "__call__", mock_pred)


@pytest.fixture
def init_runner(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    return runner


@pytest.fixture
def runner_run_all(init_runner, mock_response):
    runner = init_runner
    _ = runner.run_all()
    return runner


def test_get_cfg():
    cfg = get_cfg()
    assert "USE_POLYGON" in cfg["MODEL"]["TRANSFORMER"].keys()
    assert "DET_ONLY" in cfg["TEST"].keys()


def test_dptext_init(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
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
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
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
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
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
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    assert isinstance(runner, DPTextDETRRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["coordinates"], tuple)
    assert isinstance(runner.patch_df.iloc[0]["coordinates"], tuple)


def test_dptext_init_geojson(init_dataframes, tmp_path, mock_response):
    parent_df, patch_df = init_dataframes
    parent_df.to_file(f"{tmp_path}/parent_df.geojson", driver="GeoJSON")
    patch_df.to_file(f"{tmp_path}/patch_df.geojson", driver="GeoJSON")
    runner = DPTextDETRRunner(
        f"{tmp_path}/patch_df.geojson",
        parent_df=f"{tmp_path}/parent_df.geojson",
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    assert isinstance(runner, DPTextDETRRunner)
    assert isinstance(runner.predictor, DefaultPredictor)
    assert isinstance(runner.parent_df.iloc[0]["geometry"], Polygon)
    out = runner.run_all()
    assert isinstance(out, dict)
    assert "patch-0-0-800-40-#mapreader_text.png#.png" in out.keys()
    assert isinstance(out["patch-0-0-800-40-#mapreader_text.png#.png"], list)
    assert isinstance(
        out["patch-0-0-800-40-#mapreader_text.png#.png"][0], PatchPrediction
    )


def test_dptext_init_errors(init_dataframes):
    parent_df, patch_df = init_dataframes
    with pytest.raises(ValueError, match="path to a CSV/TSV/etc or geojson"):
        DPTextDETRRunner(
            patch_df="fake_file.txt",
            parent_df=parent_df,
            cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
        )
    with pytest.raises(ValueError, match="path to a CSV/TSV/etc or geojson"):
        DPTextDETRRunner(
            patch_df=patch_df,
            parent_df="fake_file.txt",
            cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
        )
    with pytest.raises(ValueError, match="path to a CSV/TSV/etc or geojson"):
        DPTextDETRRunner(
            patch_df=np.array([1, 2, 3]),
            parent_df=parent_df,
        )
    with pytest.raises(ValueError, match="path to a CSV/TSV/etc or geojson"):
        DPTextDETRRunner(
            patch_df=patch_df,
            parent_df=np.array([1, 2, 3]),
        )


def test_dptext_check_georeferencing(init_dataframes):
    parent_df, patch_df = init_dataframes
    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    runner.check_georeferencing()
    assert runner.georeferenced

    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df.drop(columns=["dlat", "dlon"]),
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    runner.check_georeferencing()
    assert runner.georeferenced

    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df.drop(columns=["coordinates"]),
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
    )
    runner.check_georeferencing()
    assert not runner.georeferenced


def test_dptext_run_all(init_runner, mock_response):
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
    assert set(out.columns) == set(["image_id", "pixel_geometry", "score"])
    assert "patch-0-0-800-40-#mapreader_text.png#.png" in out["image_id"].values


def test_dptext_convert_to_parent(runner_run_all, mock_response):
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
    assert set(out.columns) == set(["image_id", "patch_id", "pixel_geometry", "score"])
    assert "mapreader_text.png" in out["image_id"].values


def test_dptext_convert_to_parent_coords(runner_run_all, mock_response):
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
        ["image_id", "patch_id", "pixel_geometry", "geometry", "crs", "score"]
    )
    assert "mapreader_text.png" in out["image_id"].values
    assert out.crs == runner.parent_df.crs


def test_dptext_deduplicate(sample_dir, tmp_path, mock_response):
    maps = MapImages(f"{sample_dir}/mapreader_text.png")
    maps.add_metadata(f"{sample_dir}/mapreader_text_metadata.csv")
    maps.patchify_all(patch_size=800, path_save=tmp_path, overlap=0.5)
    maps.check_georeferencing()
    parent_df, patch_df = maps.convert_images()
    runner = DPTextDETRRunner(
        patch_df,
        parent_df=parent_df,
        cfg_file=f"{DPTEXT_DETR_PATH}/configs/DPText_DETR/ArT/R_50_poly.yaml",
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


def test_dptext_run_on_image(init_runner, mock_response):
    runner = init_runner
    out = runner.run_on_image(
        runner.patch_df.iloc[0]["image_path"], return_outputs=True
    )
    assert isinstance(out, dict)
    assert "instances" in out.keys()
    assert isinstance(out["instances"], Instances)


def test_dptext_save_to_geojson(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    _ = runner.convert_to_coords()
    runner.save_to_geojson(f"{tmp_path}/text.geojson")
    assert os.path.exists(f"{tmp_path}/text.geojson")
    gdf = gpd.read_file(f"{tmp_path}/text.geojson")
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert set(gdf.columns) == set(
        ["image_id", "patch_id", "pixel_geometry", "geometry", "crs", "score"]
    )


def test_dptext_save_to_geojson_centroid(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    _ = runner.convert_to_coords()
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
            "score",
            "polygon",
        ]
    )


def test_dptext_load_geo_predictions(runner_run_all, tmp_path):
    runner = runner_run_all
    _ = runner.convert_to_coords()
    runner.save_to_geojson(f"{tmp_path}/text.geojson")
    runner.geo_predictions = {}
    runner.load_geo_predictions(f"{tmp_path}/text.geojson")
    assert len(runner.geo_predictions)
    assert "mapreader_text.png" in runner.geo_predictions.keys()
    assert isinstance(runner.geo_predictions["mapreader_text.png"], list)
    assert isinstance(runner.geo_predictions["mapreader_text.png"][0], GeoPrediction)


def test_dptext_load_geo_predictions_errors(runner_run_all, tmp_path):
    runner = runner_run_all
    with pytest.raises(ValueError, match="must be a path to a geojson file"):
        runner.load_geo_predictions("fakefile.csv")


def test_dptext_save_to_csv_polygon(runner_run_all, tmp_path, mock_response):
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


def test_dptext_save_to_csv_centroid(runner_run_all, tmp_path, mock_response):
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


def test_dptext_save_to_csv_errors(runner_run_all, tmp_path, mock_response):
    runner = runner_run_all
    runner.patch_predictions = {}
    with pytest.raises(ValueError, match="No patch predictions found"):
        runner.save_to_csv(tmp_path)


def test_dptext_load_patch_predictions(runner_run_all, tmp_path):
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


def test_dptext_load_patch_predictions_dataframe(runner_run_all):
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


def test_dptext_load_patch_predictions_centroid(runner_run_all, tmp_path):
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


def test_dptext_load_patch_predictions_errors(runner_run_all, tmp_path):
    runner = runner_run_all
    with pytest.raises(
        ValueError, match="must be a pandas DataFrame or path to a CSV file"
    ):
        runner.load_patch_predictions("fake_file.geojson")
