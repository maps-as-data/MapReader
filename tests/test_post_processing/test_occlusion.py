from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import timm
import torch
from PIL import Image
from torchvision import models, transforms

from mapreader.process.occlusion_analysis import OcclusionAnalyzer


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
def patch_df(sample_dir):
    return pd.read_csv(f"{sample_dir}/post_processing_patch_df.csv", index_col=0)


@pytest.fixture
def model():
    return timm.create_model(
        "hf_hub:Livingwithmachines/mr_resnest101e_finetuned_OS_6inch_2nd_ed",
        pretrained=True,
    )


def test_init_dataframe(patch_df, model):
    analyzer = OcclusionAnalyzer(patch_df, model)
    assert isinstance(analyzer, OcclusionAnalyzer)
    assert len(analyzer) == 81


def test_init_path(sample_dir, model):
    patch_df = f"{sample_dir}/post_processing_patch_df.csv"
    analyzer = OcclusionAnalyzer(patch_df, model)
    assert isinstance(analyzer, OcclusionAnalyzer)
    assert len(analyzer) == 81
    assert isinstance(analyzer.patch_df.iloc[0]["pixel_bounds"], tuple)


def test_init_dataframe_transform(patch_df, model):
    transform = transforms.ToTensor()
    analyzer = OcclusionAnalyzer(patch_df, model, transform=transform)
    assert isinstance(analyzer, OcclusionAnalyzer)
    assert isinstance(analyzer.transform, Callable)
    img = Image.new("RGB", (10, 10))
    assert isinstance(analyzer.transform(img), torch.Tensor)


def test_init_fake_path_error(model):
    patch_df = "fake_df.csv"
    with pytest.raises(ValueError, match="cannot be found"):
        OcclusionAnalyzer(patch_df, model)


def test_init_error(model):
    with pytest.raises(
        ValueError, match="as a string (path to csv file) or pd.DataFrame"
    ):
        OcclusionAnalyzer({"image_id": "patch"}, model)


def test_init_dataframe_no_predictions(patch_df, model):
    patch_df.drop(columns="predicted_label", inplace=True)
    with pytest.raises(
        ValueError, match="patch dataframe should contain predicted labels"
    ):
        OcclusionAnalyzer(patch_df, model)
    patch_df.drop(columns="pred", inplace=True)
    with pytest.raises(
        ValueError, match="patch dataframe should contain predicted labels"
    ):
        OcclusionAnalyzer(patch_df, model)


def test_init_reindex_dataframe(patch_df, model):
    patch_df.reset_index(inplace=True, drop=False)
    assert "image_id" in patch_df.columns
    analyzer = OcclusionAnalyzer(patch_df, model)
    assert isinstance(analyzer, OcclusionAnalyzer)
    assert len(analyzer) == 81
    assert analyzer.patch_df.index.name == "image_id"
    assert "image_id" not in analyzer.patch_df.columns


def test_init_models_string(patch_df):
    for model2test in [
        ["resnet18", models.ResNet],
        ["alexnet", models.AlexNet],
        ["vgg11", models.VGG],
        ["squeezenet1_0", models.SqueezeNet],
        ["densenet121", models.DenseNet],
        ["inception_v3", models.Inception3],
    ]:
        model, model_type = model2test
        analyzer = OcclusionAnalyzer(patch_df, model)
        assert isinstance(analyzer.model, model_type)


def test_add_criterion(patch_df, model):
    analyzer = OcclusionAnalyzer(patch_df, model)
    analyzer.add_criterion("bce")  # loss function as str
    assert isinstance(analyzer.criterion, torch.nn.BCELoss)
    my_criterion = torch.nn.L1Loss()
    analyzer.add_criterion(my_criterion)
    assert isinstance(analyzer.criterion, torch.nn.L1Loss)


def test_criterion_errors(patch_df, model):
    analyzer = OcclusionAnalyzer(patch_df, model)
    with pytest.raises(NotImplementedError, match="criterion can only be"):
        analyzer.add_criterion("a fake criterion")
    with pytest.raises(ValueError, match="Please pass"):
        analyzer.add_criterion(0.01)


def test_run_occlusion(patch_df, model):
    analyzer = OcclusionAnalyzer(patch_df, model)
    out = analyzer.run_occlusion("railspace", 2)
    assert len(out) == 2
    assert isinstance(out[0], Image)
