from mapreader import AnnotationsLoader
from mapreader.classify.datasets import PatchDataset
import pytest
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent / "sample_files"

@pytest.fixture
def load_annots(sample_dir):
    annots = AnnotationsLoader()
    annots.load(f"{sample_dir}/test_annots.csv", reset_index=True)
    return annots

@pytest.mark.dependency(name="load_annots_csv", scope="session")
def test_load_csv(load_annots, sample_dir):
    annots = load_annots
    assert len(annots.annotations) == 29
    assert isinstance(annots.annotations, pd.DataFrame)
    assert annots.labels_map == {0: 'stuff', 1: 'nothing'}
    annots.load(f"{sample_dir}/test_annots_append.csv", append=True) #test append
    assert len(annots.annotations) == 31
    assert annots.unique_labels == ["stuff", "nothing", "new"]
    assert annots.labels_map == {0: 'stuff', 1: 'nothing', 2: 'new'}

def test_load_df(sample_dir):
    annots = AnnotationsLoader()
    df = pd.read_csv(f"{sample_dir}/test_annots.csv", sep=",", index_col=0)
    annots.load(df)
    assert len(annots.annotations) == 29
    assert isinstance(annots.annotations, pd.DataFrame)
    assert annots.labels_map == {0: 'stuff', 1: 'nothing'}

def test_create_datsets_default_transforms(load_annots):
    annots = load_annots
    annots.create_datasets(0.5, 0.3, 0.2)
    assert annots.dataset_sizes == {'train': 14, 'val': 9, 'test': 6}
    assert isinstance(annots.datasets["train"], PatchDataset)
    assert isinstance(annots.datasets["train"].patch_df, pd.DataFrame)
    for v in annots.datasets.values():
        assert list(v.patch_df.columns) == ['image_id', 'image_path', 'label', 'label_index']

def test_create_datasets_custom_transforms(load_annots):
    annots = load_annots
    my_transform = transforms.Compose([transforms.ToTensor()])
    annots.create_datasets(train_transform=my_transform, val_transform=my_transform, test_transform=my_transform)
    assert annots.dataset_sizes == {'train': 20, 'val': 4, 'test': 5}
    assert isinstance(annots.datasets["train"], PatchDataset)
    for v in annots.datasets.values():
        assert v.transform == my_transform

@pytest.mark.dependency(name="dataloaders", scope="session")
def test_create_dataloaders_default_sampler(load_annots):
    annots = load_annots
    dataloaders = annots.create_dataloaders(batch_size=8)
    assert dataloaders == annots.dataloaders
    assert isinstance(dataloaders["train"], DataLoader)
    assert dataloaders["train"].batch_size == 8

def test_create_dataloaders_custom_sampler(load_annots):
    annots = load_annots
    annots.create_datasets()
    sampler = RandomSampler(annots.datasets["train"])
    dataloaders = annots.create_dataloaders(sampler=sampler)
    assert dataloaders == annots.dataloaders
    assert isinstance(dataloaders["train"], DataLoader)
    assert dataloaders["train"].sampler == sampler
    
def test_create_dataloaders_no_sampler(load_annots):
    annots = load_annots
    dataloaders = annots.create_dataloaders(batch_size=8,  sampler=None, shuffle=True)
    assert dataloaders == annots.dataloaders
    assert isinstance(dataloaders["train"], DataLoader)
    assert dataloaders["train"].batch_size == 8

#errors

def test_load_csv_errors():
    annots=AnnotationsLoader()
    with pytest.raises(ValueError, match="cannot be found"):
        annots.load("a_fake_file.csv")

def test_create_datasets_errors(load_annots):
    annots = AnnotationsLoader()
    with pytest.raises(ValueError, match="No annotations"):
        annots.create_datasets()
    annots = load_annots
    with pytest.raises(ValueError, match="do not add"):
        annots.create_datasets(0.1, 0.2, 0.3)

def test_create_dataloaders_errors(load_annots):
    annots = load_annots
    with pytest.raises(ValueError):
        annots.create_dataloaders(sampler="a test string")
    
