from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import timm
import torch
import transformers
from lightning.pytorch import Trainer
from torchvision import models
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from mapreader import AnnotationsLoader
from mapreader.classify.datasets import PatchDataset
from mapreader.classify.lightning_classifier import LightningClassifierContainer


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
@pytest.mark.dependency(depends=["load_annots_csv", "dataloaders"], scope="session")
def inputs(sample_dir):
    annots = AnnotationsLoader()
    annots.load(
        f"{sample_dir}/test_annots.csv",
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
    classifier = LightningClassifierContainer(
        None, None, None, load_path=f"{sample_dir}/test.pkl"
    )
    return classifier


@pytest.fixture
def ready_classifier(sample_dir):
    """Classifier with loss/optimizer/scheduler set up, ready for training."""
    classifier = LightningClassifierContainer(
        None, None, None, load_path=f"{sample_dir}/test.pkl"
    )
    classifier.add_loss_fn("cross entropy")
    classifier.initialize_optimizer("adam")
    classifier.initialize_scheduler()
    return classifier


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


@pytest.mark.dependency(name="lc_models_by_string", scope="session")
def test_init_models_string(inputs):
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
        classifier = LightningClassifierContainer(
            model, labels_map=annots.labels_map, dataloaders=dataloaders
        )
        assert isinstance(classifier.model, model_type)
        assert all(k in classifier.dataloaders.keys() for k in ["train", "test", "val"])

        classifier = LightningClassifierContainer(model, labels_map=annots.labels_map)
        assert isinstance(classifier.model, model_type)
        assert classifier.dataloaders == {}


def test_init_inception_input_size(inputs):
    """Regression: inception_v3 must set input_size to (299, 299), not 299."""
    annots, _ = inputs
    classifier = LightningClassifierContainer(
        "inception_v3", labels_map=annots.labels_map
    )
    assert classifier.input_size == (299, 299)
    assert classifier.is_inception is True


def test_init_resnet18_torch(inputs):
    annots, dataloaders = inputs
    my_model = models.resnet18(weights="DEFAULT")
    assert isinstance(my_model, models.ResNet)  # sanity check
    num_input_features = my_model.fc.in_features
    my_model.fc = torch.nn.Linear(num_input_features, len(annots.labels_map))
    classifier = LightningClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )
    assert isinstance(classifier.model, models.ResNet)
    assert all(k in classifier.dataloaders.keys() for k in ["train", "test", "val"])

    classifier = LightningClassifierContainer(my_model, labels_map=annots.labels_map)
    assert isinstance(classifier.model, models.ResNet)
    assert classifier.dataloaders == {}


def test_init_resnet18_pickle(inputs, sample_dir):
    annots, dataloaders = inputs
    my_model = torch.load(f"{sample_dir}/model_test.pkl")
    assert isinstance(my_model, models.ResNet)  # sanity check
    classifier = LightningClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )
    assert isinstance(classifier.model, models.ResNet)
    assert all(k in classifier.dataloaders.keys() for k in ["train", "test", "val"])
    classifier = LightningClassifierContainer(my_model, labels_map=annots.labels_map)
    assert isinstance(classifier.model, models.ResNet)
    assert classifier.dataloaders == {}


@pytest.mark.dependency(name="lc_hf_models", scope="session")
def test_init_resnet18_hf(inputs):
    annots, dataloaders = inputs
    AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    my_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    model_type = transformers.models.resnet.ResNetForImageClassification
    assert isinstance(my_model, model_type)  # sanity check
    classifier = LightningClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )
    assert isinstance(classifier.model, model_type)
    assert all(k in classifier.dataloaders.keys() for k in ["train", "test", "val"])
    classifier = LightningClassifierContainer(my_model, labels_map=annots.labels_map)
    assert isinstance(classifier.model, model_type)
    assert classifier.dataloaders == {}


def test_init_resnet18_timm(inputs):
    annots, dataloaders = inputs
    my_model = timm.create_model(
        "resnet18", pretrained=True, num_classes=len(annots.labels_map)
    )
    assert isinstance(my_model, timm.models.ResNet)  # sanity check
    classifier = LightningClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )
    assert isinstance(classifier.model, timm.models.ResNet)
    assert all(k in classifier.dataloaders.keys() for k in ["train", "test", "val"])
    classifier = LightningClassifierContainer(my_model, labels_map=annots.labels_map)
    assert isinstance(classifier.model, timm.models.ResNet)
    assert classifier.dataloaders == {}


def test_init_errors(sample_dir):
    with pytest.raises(
        ValueError, match="``model`` and ``labels_map`` must be defined"
    ):
        LightningClassifierContainer("resnet18", None, None)


def test_init_models_string_errors(inputs):
    annots, dataloaders = inputs
    with pytest.raises(NotImplementedError, match="Invalid model name"):
        LightningClassifierContainer(
            "resnext101_32x8d", labels_map=annots.labels_map, dataloaders=dataloaders
        )


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------


def test_load_no_dataloaders(inputs, sample_dir):
    annots, dataloaders = inputs
    classifier = LightningClassifierContainer(
        None, None, None, load_path=f"{sample_dir}/test.pkl"
    )
    assert all(k in classifier.dataloaders.keys() for k in ["train", "test", "val"])
    assert classifier.labels_map == annots.labels_map
    assert isinstance(classifier.model, models.ResNet)

    # without explicitly passing dataloaders as None
    classifier = LightningClassifierContainer(
        None, None, load_path=f"{sample_dir}/test.pkl"
    )
    assert all(k in classifier.dataloaders.keys() for k in ["train", "test", "val"])
    assert classifier.labels_map == annots.labels_map
    assert isinstance(classifier.model, models.ResNet)


def test_load_w_dataloaders(inputs, sample_dir):
    annots, dataloaders = inputs
    dataloaders["new_train"] = dataloaders.pop("train")
    dataloaders["new_val"] = dataloaders.pop("val")
    dataloaders["new_test"] = dataloaders.pop("test")

    classifier = LightningClassifierContainer(
        None, None, dataloaders=dataloaders, load_path=f"{sample_dir}/test.pkl"
    )
    assert all(
        k in classifier.dataloaders.keys()
        for k in ["train", "test", "val", "new_train", "new_test", "new_val"]
    )
    assert classifier.labels_map == annots.labels_map
    assert isinstance(classifier.model, models.ResNet)


def test_init_load(inputs, load_classifier):
    annots, dataloaders = inputs
    classifier = load_classifier
    assert all(k in classifier.dataloaders.keys() for k in ["train", "test", "val"])
    assert classifier.labels_map == annots.labels_map
    assert isinstance(classifier.model, models.ResNet)


def test_save(load_classifier, tmp_path):
    classifier = load_classifier
    classifier.save(save_path=f"{tmp_path}/out.obj")
    assert os.path.isfile(f"{tmp_path}/out.obj")
    assert os.path.isfile(f"{tmp_path}/model_out.obj")


def test_save_load_roundtrip(inputs, tmp_path):
    """Save a fresh classifier and load it back; check model and labels_map survive."""
    annots, _ = inputs
    classifier = LightningClassifierContainer("resnet18", labels_map=annots.labels_map)
    classifier.save(save_path=f"{tmp_path}/rt.obj")

    loaded = LightningClassifierContainer(None, None, load_path=f"{tmp_path}/rt.obj")
    assert isinstance(loaded.model, models.ResNet)
    assert loaded.labels_map == annots.labels_map


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def test_add_loss_fn(load_classifier):
    classifier = load_classifier
    classifier.add_loss_fn("bce")
    assert isinstance(classifier.loss_fn, torch.nn.BCELoss)
    loss_fn = torch.nn.L1Loss()
    classifier.add_loss_fn(loss_fn)
    assert isinstance(classifier.loss_fn, torch.nn.L1Loss)


def test_loss_fn_errors(load_classifier):
    classifier = load_classifier
    with pytest.raises(NotImplementedError, match="loss function can only be"):
        classifier.add_loss_fn("a fake loss_fn")
    with pytest.raises(ValueError, match="Please pass"):
        classifier.add_loss_fn(0.01)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def test_initialize_optimizer(load_classifier):
    classifier = load_classifier
    classifier.initialize_optimizer("sgd")
    assert isinstance(classifier.optimizer, torch.optim.SGD)

    params2optimize = classifier.generate_layerwise_lrs(
        min_lr=1e-4, max_lr=1e-3, spacing="geomspace"
    )
    classifier.initialize_optimizer("adam", params2optimize)
    assert isinstance(classifier.optimizer, torch.optim.Adam)


def test_optimizer_errors(load_classifier):
    classifier = load_classifier
    with pytest.raises(NotImplementedError, match="At present, only"):
        classifier.initialize_optimizer("a fake optimizer")
    with pytest.raises(NotImplementedError, match="must be one of"):
        classifier.generate_layerwise_lrs(1e-4, 1e-3, "a fake spacing")


def test_generate_layerwise_lrs_uses_lr_key(load_classifier):
    """Regression: param groups must use 'lr' not 'learning rate'."""
    classifier = load_classifier
    params2optimize = classifier.generate_layerwise_lrs(min_lr=1e-4, max_lr=1e-3)
    for group in params2optimize:
        assert "lr" in group, "param group missing 'lr' key"
        assert "learning rate" not in group, "param group has wrong key 'learning rate'"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_initialize_scheduler(load_classifier):
    classifier = load_classifier
    classifier.initialize_optimizer()
    classifier.initialize_scheduler(
        scheduler_param_dict={"step_size": 5, "gamma": 0.02}
    )
    assert isinstance(classifier.scheduler, torch.optim.lr_scheduler.StepLR)
    assert classifier.scheduler.step_size == 5
    assert classifier.scheduler.gamma == 0.02


def test_scheduler_errors(load_classifier):
    classifier = load_classifier
    with pytest.raises(ValueError, match="not yet defined"):
        classifier.initialize_scheduler()
    classifier.initialize_optimizer()
    with pytest.raises(NotImplementedError, match="only StepLR"):
        classifier.initialize_scheduler("a fake scheduler type")


# ---------------------------------------------------------------------------
# configure_optimizers (Lightning hook)
# ---------------------------------------------------------------------------


def test_configure_optimizers_no_scheduler(load_classifier):
    classifier = load_classifier
    classifier.initialize_optimizer("adam")
    result = classifier.configure_optimizers()
    assert isinstance(result, torch.optim.Adam)


def test_configure_optimizers_with_scheduler(load_classifier):
    classifier = load_classifier
    classifier.initialize_optimizer("adam")
    classifier.initialize_scheduler()
    result = classifier.configure_optimizers()
    assert isinstance(result, dict)
    assert "optimizer" in result
    assert "lr_scheduler" in result
    assert isinstance(result["optimizer"], torch.optim.Adam)


def test_configure_optimizers_no_optimizer(load_classifier):
    classifier = load_classifier
    classifier.optimizer = None
    with pytest.raises(ValueError, match="optimizer should be defined"):
        classifier.configure_optimizers()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_calculate_add_metrics(load_classifier):
    classifier = load_classifier
    y_true = np.ones(10)
    np.random.seed(0)
    y_pred = np.random.randint(0, 2, 10)
    y_score = np.random.random_sample((10, 1))
    classifier.calculate_add_metrics(y_true, y_pred, y_score, phase="pytest")
    assert "pytest" in classifier.metrics.keys()
    for metric in ["precision", "recall", "fscore", "support"]:
        for suffix in ["0", "micro", "macro", "weighted"]:
            assert f"{metric}_{suffix}" in classifier.metrics["pytest"].keys()
            assert len(classifier.metrics["pytest"][f"{metric}_{suffix}"]) == 1


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def test_load_dataset(load_classifier, sample_dir):
    classifier = load_classifier
    dataset = PatchDataset(f"{sample_dir}/test_annots_append.csv", "test")
    classifier.load_dataset(dataset, "pytest_set", batch_size=8, shuffle=True)
    assert "pytest_set" in classifier.dataloaders


# ---------------------------------------------------------------------------
# Inference (direct, no Trainer)
# ---------------------------------------------------------------------------


@pytest.mark.dependency(depends=["lc_models_by_string"], scope="session")
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
        classifier = LightningClassifierContainer(
            model, labels_map=annots.labels_map, dataloaders=dataloaders
        )
        classifier.add_loss_fn()
        classifier.initialize_optimizer()
        classifier.initialize_scheduler()
        classifier.load_dataset(infer_inputs, set_name="infer")
        classifier.inference("infer")


@pytest.mark.dependency(depends=["lc_hf_models"], scope="session")
def test_infer_hf_models(inputs, infer_inputs):
    annots, dataloaders = inputs
    AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    my_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    classifier = LightningClassifierContainer(
        my_model, labels_map=annots.labels_map, dataloaders=dataloaders
    )
    classifier.add_loss_fn()
    classifier.initialize_optimizer()
    classifier.initialize_scheduler()
    classifier.load_dataset(infer_inputs, set_name="infer")
    classifier.inference("infer")


def test_infer_timm_models(inputs, infer_inputs):
    annots, dataloaders = inputs
    for model in [
        "resnest50d_4s2x40d",
        "resnest101e",
        "resnext101_32x8d.fb_swsl_ig1b_ft_in1k",
        "resnet152",
        "tf_efficientnet_b3.ns_jft_in1k",
        "swin_base_patch4_window7_224",
        "vit_base_patch16_224",
    ]:
        my_model = timm.create_model(
            model, pretrained=True, num_classes=len(annots.labels_map)
        )
        classifier = LightningClassifierContainer(
            my_model, labels_map=annots.labels_map, dataloaders=dataloaders
        )
        classifier.add_loss_fn()
        classifier.initialize_optimizer()
        classifier.initialize_scheduler()
        classifier.load_dataset(infer_inputs, set_name="infer")
        classifier.inference("infer")


# ---------------------------------------------------------------------------
# Training via Lightning Trainer (fast_dev_run=True — 1 batch only)
# ---------------------------------------------------------------------------


def test_training_step(inputs):
    """Smoke-test: Trainer.fit() with fast_dev_run=True runs one batch without error."""
    annots, dataloaders = inputs
    classifier = LightningClassifierContainer("resnet18", labels_map=annots.labels_map)
    classifier.add_loss_fn("cross entropy")
    classifier.initialize_optimizer("adam")
    classifier.initialize_scheduler()

    trainer = Trainer(
        max_epochs=1,
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(
        classifier,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
    )


def test_predict_step(inputs, infer_inputs):
    """Smoke-test: Trainer.predict() populates pred_label."""
    annots, dataloaders = inputs
    classifier = LightningClassifierContainer("resnet18", labels_map=annots.labels_map)
    classifier.add_loss_fn("cross entropy")
    classifier.initialize_optimizer("adam")
    classifier.initialize_scheduler()

    from torch.utils.data import DataLoader

    infer_loader = DataLoader(infer_inputs, batch_size=1)

    trainer = Trainer(
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.predict(classifier, dataloaders=infer_loader)
    # After predict, predictions should have been collected
    assert len(classifier.pred_label) > 0 or len(classifier.pred_label_indices) > 0
