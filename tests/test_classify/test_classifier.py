from mapreader import AnnotationsLoader, ClassifierContainer
from mapreader.classify.datasets import PatchDataset
import pytest
from pathlib import Path
import numpy as np
from torchvision import models
import torch
import os

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent / "sample_files"

@pytest.fixture
@pytest.mark.dependency(depends=["load_annots_csv", "dataloaders"], scope="session")
def inputs(sample_dir):
    annots = AnnotationsLoader()
    annots.load(f"{sample_dir}/test_annots.csv", reset_index=True)
    dataloaders = annots.create_dataloaders(batch_size=8)
    return annots, dataloaders

@pytest.fixture
def load_classifier(sample_dir):
    classifier = ClassifierContainer(None, None, None, load_path=f"{sample_dir}/test.pkl")
    return classifier

#test loading model using model name as string

def test_init_models_string(inputs):
    annots, dataloaders = inputs
    for model_info in [
        ["resnet18", models.ResNet],
        ["alexnet", models.AlexNet],
        ["vgg11", models.VGG], 
        ["squeezenet", models.SqueezeNet],
        ["densenet121", models.DenseNet],
        ["inception", models.Inception3],
    ]:
        model, model_type = model_info
        assert isinstance(model, model_type) # sanity check
        classifier = ClassifierContainer(model, dataloaders=dataloaders, labels_map=annots.labels_map)
        assert isinstance(classifier.model, model_type)

#test loading model using torch load

def test_init_resnet18_torch(inputs):
    annots, dataloaders = inputs
    my_model = models.resnet18(pretrained=True)
    num_input_features = my_model.fc.in_features
    my_model.fc = torch.nn.Linear(num_input_features, len(annots.labels_map))
    classifier = ClassifierContainer(my_model, dataloaders, annots.labels_map) #resnet18 as nn.Module
    assert isinstance(classifier.model, models.ResNet)

#test loading model from pickle file using torch load

def test_init_resnet18_pickle(inputs, sample_dir):
    annots, dataloaders = inputs
    my_model = torch.load(f"{sample_dir}/model_test.pkl")
    classifier = ClassifierContainer(my_model, dataloaders=dataloaders, labels_map=annots.labels_map) #resnet18 as pkl (from sample files)
    assert isinstance(classifier.model, models.ResNet)

#test loading object from pickle file

def test_init_load(inputs, load_classifier):
    annots, dataloaders = inputs
    classifier = load_classifier
    assert list(classifier.dataloaders.keys()) == list(dataloaders.keys())
    assert classifier.labels_map == annots.labels_map
    assert isinstance(classifier.model, models.ResNet)

def test_add_criterion(load_classifier):
    classifier = load_classifier
    classifier.add_criterion("bce") #loss function as str
    assert isinstance(classifier.criterion, torch.nn.BCELoss)
    my_criterion = torch.nn.L1Loss()
    classifier.add_criterion(my_criterion)
    assert isinstance(classifier.criterion, torch.nn.L1Loss)

def test_initialize_optimiser(load_classifier):
    classifier = load_classifier
    classifier.initialize_optimizer("sgd")
    assert isinstance(classifier.optimizer, torch.optim.SGD)
    
    params2optimise = classifier.generate_layerwise_lrs(min_lr=1e-4, max_lr=1e-3, spacing="geomspace")
    classifier.initialize_optimizer("adam", params2optimise)
    assert isinstance(classifier.optimizer, torch.optim.Adam)

def test_initialize_scheduler(load_classifier):
    classifier = load_classifier
    classifier.initialize_optimizer()
    classifier.initialize_scheduler(scheduler_param_dict= {'step_size': 5, 'gamma': 0.02})
    assert isinstance(classifier.scheduler, torch.optim.lr_scheduler.StepLR)
    assert classifier.scheduler.step_size == 5
    assert classifier.scheduler.gamma == 0.02

def test_calculate_add_metrics(load_classifier):
    classifier = load_classifier
    y_true=np.ones(10)
    np.random.seed(0)
    y_pred=np.random.randint(0,2,10)
    y_score=np.random.random_sample((10,1))
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

#errors

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
    with pytest.raises(NotImplementedError, match = "At present, only"):
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

#dont test train/infer here due to fake file paths
