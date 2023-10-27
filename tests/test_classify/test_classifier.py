from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import timm
import torch
import transformers
from torchvision import models
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from mapreader import AnnotationsLoader, ClassifierContainer
from mapreader.classify.datasets import PatchDataset


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
@pytest.mark.dependency(depends=["load_annots_csv", "dataloaders"], scope="session")
def inputs(sample_dir):
    annots = AnnotationsLoader()
    annots.load(
        f"{sample_dir}/test_annots.csv",
        reset_index=True,
        remove_broken=False,
        ignore_broken=True,
    )
    dataloaders = annots.create_dataloaders(batch_size=8)
    return annots, dataloaders


@pytest.fixture
def infer_inputs(sample_dir):
    infer_dict = {
        "image_id": ["cropped_74488689.png"],
        "image_path": [f"{sample_dir}/cropped_74488689.png"],
    }
    infer_df = pd.DataFrame.from_dict(infer_dict, orient="columns")
    infer = PatchDataset(infer_df, transform="val")
    return infer


@pytest.fixture
def load_classifier(sample_dir):
    classifier = ClassifierContainer(
        None, None, None, load_path=f"{sample_dir}/test.pkl"
    )
    return classifier


# test loading model using model name as string


@pytest.mark.dependency(name="models_by_string", scope="session")
def test_init_models_string(inputs, infer_inputs):
    annots, dataloaders = inputs
    for model2test in [
        ["resnet18", models.ResNet],
        ["alexnet", models.AlexNet],
        ["vgg11", models.VGG],
        ["squeezenet1_0", models.SqueezeNet],
        ["densenet121", models.DenseNet],
        ["inception_v3", models.Inception3],
    ]:
        model, model_type = model2test
        classifier = ClassifierContainer(
            model, labels_map=annots.labels_map, dataloaders=dataloaders
        )
        assert isinstance(classifier.model, model_type)


def test_init_models_string_errors(inputs):
    annots, dataloaders = inputs
    with pytest.raises(NotImplementedError, match="Invalid model name"):
        ClassifierContainer(
            "resnext101_32x8d", labels_map=annots.labels_map,  dataloaders=dataloaders
        )


# test loading model (e.g. resnet18) using torch load


def test_init_resnet18_torch(inputs):
    annots, dataloaders = inputs
    my_model = models.resnet18(pretrained=True)
    assert isinstance(my_model, models.ResNet)  # sanity check
    num_input_features = my_model.fc.in_features
    my_model.fc = torch.nn.Linear(num_input_features, len(annots.labels_map))
    classifier = ClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )  # resnet18 as nn.Module
    assert isinstance(classifier.model, models.ResNet)


# test loading model from pickle file using torch load


def test_init_resnet18_pickle(inputs, sample_dir):
    annots, dataloaders = inputs
    my_model = torch.load(f"{sample_dir}/model_test.pkl")
    assert isinstance(my_model, models.ResNet)  # sanity check
    classifier = ClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )  # resnet18 as pkl (from sample files)
    assert isinstance(classifier.model, models.ResNet)


# test loading model from hugging face


@pytest.mark.dependency(name="hf_models", scope="session")
def test_init_resnet18_hf(inputs):
    annots, dataloaders = inputs
    AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    my_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    model_type = transformers.models.resnet.ResNetForImageClassification
    assert isinstance(my_model, model_type)  # sanity check
    classifier = ClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )
    assert isinstance(classifier.model, model_type)


# test loading model using timm


def test_init_resnet18_timm(inputs):
    annots, dataloaders = inputs
    my_model = timm.create_model(
        "resnet18", pretrained=True, num_classes=len(annots.labels_map)
    )
    assert isinstance(my_model, timm.models.ResNet)  # sanity check
    classifier = ClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )
    assert isinstance(classifier.model, timm.models.ResNet)


@pytest.mark.dependency(name="timm_models", scope="session")
def test_init_models_timm(inputs):
    annots, dataloaders = inputs
    for model2test in [
        ["resnest50d_4s2x40d", timm.models.ResNet],
        ["resnest101e", timm.models.ResNet],
        ["swsl_resnext101_32x8d", timm.models.ResNet],
        ["resnet152", timm.models.ResNet],
        ["tf_efficientnet_b3_ns", timm.models.EfficientNet],
        ["swin_base_patch4_window7_224", timm.models.swin_transformer.SwinTransformer],
        ["vit_base_patch16_224", timm.models.vision_transformer.VisionTransformer],
    ]:  # these are models from 2021 paper
        model, model_type = model2test
        my_model = timm.create_model(
            model, pretrained=True, num_classes=len(annots.labels_map)
        )
        assert isinstance(my_model, model_type)
        classifier = ClassifierContainer(
            my_model, labels_map=annots.labels_map, dataloaders=dataloaders
        )
        assert isinstance(classifier.model, model_type)


# test loading object from pickle file


def test_init_load(inputs, load_classifier):
    annots, dataloaders = inputs
    classifier = load_classifier
    assert list(classifier.dataloaders.keys()) == list(dataloaders.keys())
    assert classifier.labels_map == annots.labels_map
    assert isinstance(classifier.model, models.ResNet)


def test_add_criterion(load_classifier):
    classifier = load_classifier
    classifier.add_criterion("bce")  # loss function as str
    assert isinstance(classifier.criterion, torch.nn.BCELoss)
    my_criterion = torch.nn.L1Loss()
    classifier.add_criterion(my_criterion)
    assert isinstance(classifier.criterion, torch.nn.L1Loss)


def test_initialize_optimizer(load_classifier):
    classifier = load_classifier
    classifier.initialize_optimizer("sgd")
    assert isinstance(classifier.optimizer, torch.optim.SGD)

    params2optimize = classifier.generate_layerwise_lrs(
        min_lr=1e-4, max_lr=1e-3, spacing="geomspace"
    )
    classifier.initialize_optimizer("adam", params2optimize)
    assert isinstance(classifier.optimizer, torch.optim.Adam)


def test_initialize_scheduler(load_classifier):
    classifier = load_classifier
    classifier.initialize_optimizer()
    classifier.initialize_scheduler(
        scheduler_param_dict={"step_size": 5, "gamma": 0.02}
    )
    assert isinstance(classifier.scheduler, torch.optim.lr_scheduler.StepLR)
    assert classifier.scheduler.step_size == 5
    assert classifier.scheduler.gamma == 0.02


def test_calculate_add_metrics(load_classifier):
    classifier = load_classifier
    y_true = np.ones(10)
    np.random.seed(0)
    y_pred = np.random.randint(0, 2, 10)
    y_score = np.random.random_sample((10, 1))
    classifier.calculate_add_metrics(y_true, y_pred, y_score, phase="pytest")
    assert len(classifier.metrics) == 20
    assert "epoch_fscore_0_pytest" in classifier.metrics


def test_save(load_classifier, tmp_path):
    classifier = load_classifier
    classifier.save(save_path=f"{tmp_path}/out.obj")
    assert os.path.isfile(f"{tmp_path}/out.obj")
    assert os.path.isfile(f"{tmp_path}/model_out.obj")


def test_load_dataset(load_classifier, sample_dir):
    classifier = load_classifier
    dataset = PatchDataset(f"{sample_dir}/test_annots_append.csv", "test")
    classifier.load_dataset(dataset, "pytest_set", batch_size=8, shuffle=True)


# errors


def test_init_errors(sample_dir):
    with pytest.raises(ValueError, match="cannot be used together"):
        ClassifierContainer("VGG", None, None, load_path=f"{sample_dir}/test.pkl")
    with pytest.raises(ValueError, match="Unless passing ``load_path``"):
        ClassifierContainer("VGG", None, None)


def test_criterion_errors(load_classifier):
    classifier = load_classifier
    with pytest.raises(NotImplementedError, match="criterion can only be"):
        classifier.add_criterion("a fake criterion")
    with pytest.raises(ValueError, match="Please pass"):
        classifier.add_criterion(0.01)


def test_optimizer_errors(load_classifier):
    classifier = load_classifier
    with pytest.raises(NotImplementedError, match="At present, only"):
        classifier.initialize_optimizer("a fake optimizer")
    with pytest.raises(NotImplementedError, match="must be one of"):
        classifier.generate_layerwise_lrs(1e-4, 1e-3, "a fake spacing")


def test_scheduler_errors(load_classifier):
    classifier = load_classifier
    with pytest.raises(ValueError, match="not yet defined"):
        classifier.initialize_scheduler()
    classifier.initialize_optimizer()
    with pytest.raises(NotImplementedError, match="can only be"):
        classifier.initialize_scheduler("a fake scheduler type")


# dont test train here due to fake file paths

# test inference w/ various models and model-types


@pytest.mark.dependency(depends=["models_by_string"], scope="session")
def test_infer_models_by_string(inputs, infer_inputs):
    annots, dataloaders = inputs
    for model in [
        "resnet18",
        "alexnet",
        "vgg11",
        "squeezenet1_0",
        "densenet121",
        "inception_v3",
    ]:
        classifier = ClassifierContainer(
            model, labels_map=annots.labels_map, dataloaders=dataloaders
        )
        classifier.add_criterion()
        classifier.initialize_optimizer()
        classifier.initialize_scheduler()
        classifier.load_dataset(infer_inputs, set_name="infer")
        classifier.inference("infer")


@pytest.mark.dependency(depends=["hf_models"], scope="session")
def test_infer_hf_models(inputs, infer_inputs):
    annots, dataloaders = inputs
    AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    my_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    classifier = ClassifierContainer(
        my_model, labels_map=annots.labels_map,  dataloaders=dataloaders
    )
    classifier.add_criterion()
    classifier.initialize_optimizer()
    classifier.initialize_scheduler()
    classifier.load_dataset(infer_inputs, set_name="infer")
    classifier.inference("infer")


@pytest.mark.dependency(depends=["timm_models"], scope="session")
def test_infer_timm_models(inputs, infer_inputs):
    annots, dataloaders = inputs
    for model in [
        "resnest50d_4s2x40d",
        "resnest101e",
        "swsl_resnext101_32x8d",
        "resnet152",
        "tf_efficientnet_b3_ns",
        "swin_base_patch4_window7_224",
        "vit_base_patch16_224",
    ]:  # these are models from 2021 paper
        my_model = timm.create_model(
            model, pretrained=True, num_classes=len(annots.labels_map)
        )
        classifier = ClassifierContainer(
            my_model, labels_map=annots.labels_map, dataloaders=dataloaders
        )
        classifier.add_criterion()
        classifier.initialize_optimizer()
        classifier.initialize_scheduler()
        classifier.load_dataset(infer_inputs, set_name="infer")
        classifier.inference("infer")
