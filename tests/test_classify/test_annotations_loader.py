from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from mapreader import AnnotationsLoader
from mapreader.classify.datasets import PatchContextDataset, PatchDataset


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
def load_annots(sample_dir):
    annots = AnnotationsLoader()
    annots.load(
        f"{sample_dir}/test_annots.csv",
        reset_index=True,
        remove_broken=False,
        ignore_broken=True,
    )
    return annots


@pytest.mark.dependency(name="load_annots_csv", scope="session")
def test_load_csv(load_annots, sample_dir):
    annots = load_annots
    assert len(annots.annotations) == 81
    assert isinstance(annots.annotations, pd.DataFrame)
    assert annots.labels_map == {0: "no", 1: "railspace"}
    annots.load(
        f"{sample_dir}/test_annots_append.csv",
        append=True,
        remove_broken=False,
        ignore_broken=True,
    )  # test append
    assert len(annots.annotations) == 83
    assert annots.unique_labels == ["no", "railspace", "building"]
    assert annots.labels_map == {0: "no", 1: "railspace", 2: "building"}


def test_labels_map(sample_dir):
    annots = AnnotationsLoader()
    annots.load(
        f"{sample_dir}/test_annots.csv",
        reset_index=True,
        remove_broken=False,
        ignore_broken=True,
        labels_map={1: "no", 0: "railspace"},
    )
    assert len(annots.annotations) == 81
    assert annots.labels_map == {0: "railspace", 1: "no"}
    # test append
    annots.load(
        f"{sample_dir}/test_annots_append.csv",
        append=True,
        remove_broken=False,
        ignore_broken=True,
    )
    assert len(annots.annotations) == 83
    assert annots.unique_labels == ["no", "railspace", "building"]
    assert annots.labels_map == {0: "railspace", 1: "no", 2: "building"}


@pytest.mark.dependency(name="load_annots_df", scope="session")
def test_load_df(sample_dir):
    annots = AnnotationsLoader()
    df = pd.read_csv(f"{sample_dir}/test_annots.csv", sep=",", index_col=0)
    annots.load(df, remove_broken=False, ignore_broken=True)
    assert len(annots.annotations) == 81
    assert isinstance(annots.annotations, pd.DataFrame)
    assert annots.labels_map == {0: "no", 1: "railspace"}


def test_init_images_dir(sample_dir):
    annots = AnnotationsLoader()
    annots.load(
        f"{sample_dir}/test_annots.csv",
        remove_broken=False,
        ignore_broken=True,
        images_dir=sample_dir,
    )
    assert annots.annotations.iloc[0]["image_path"].startswith(str(sample_dir))


def test_scramble_frame(sample_dir, load_annots):
    annots = AnnotationsLoader()
    annots.load(
        f"{sample_dir}/test_annots.csv",
        reset_index=False,
        scramble_frame=True,
        remove_broken=False,
        ignore_broken=True,
    )
    assert len(annots.annotations) == 81
    assert not annots.annotations.index.equals(load_annots.annotations.index)

    # with reset_index
    annots = AnnotationsLoader()
    annots.load(
        f"{sample_dir}/test_annots.csv",
        reset_index=True,
        scramble_frame=True,
        remove_broken=False,
        ignore_broken=True,
    )
    assert len(annots.annotations) == 81
    assert annots.annotations.index[0] == 0


def test_create_datasets_default_transforms(load_annots):
    annots = load_annots
    annots.create_datasets(0.5, 0.3, 0.2)
    assert annots.dataset_sizes == {"train": 40, "val": 24, "test": 17}
    assert isinstance(annots.datasets["train"], PatchDataset)
    assert isinstance(annots.datasets["train"].patch_df, pd.DataFrame)


def test_create_context_datasets_default_transforms(load_annots):
    annots = load_annots
    annots.create_datasets(
        0.5, 0.3, 0.2, context_datasets=True, context_df=annots.annotations
    )
    assert annots.dataset_sizes == {"train": 40, "val": 24, "test": 17}
    assert isinstance(annots.datasets["train"], PatchContextDataset)
    assert isinstance(annots.datasets["train"].patch_df, pd.DataFrame)


def test_create_context_datasets_missing_cols(load_annots):
    annots = load_annots
    annotations = annots.annotations.drop(columns=["pixel_bounds", "parent_id"])
    annots.create_datasets(0.5, 0.3, 0.2, context_datasets=True, context_df=annotations)
    assert annots.dataset_sizes == {"train": 40, "val": 24, "test": 17}
    assert isinstance(annots.datasets["train"], PatchContextDataset)
    assert isinstance(annots.datasets["train"].patch_df, pd.DataFrame)


def test_create_datasets_custom_transforms(load_annots):
    annots = load_annots
    my_transform = transforms.Compose([transforms.ToTensor()])
    annots.create_datasets(
        train_transform=my_transform,
        val_transform=my_transform,
        test_transform=my_transform,
    )
    assert annots.dataset_sizes == {"train": 56, "val": 12, "test": 13}
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
    dataloaders = annots.create_dataloaders(batch_size=8, sampler=None, shuffle=True)
    assert dataloaders == annots.dataloaders
    assert isinstance(dataloaders["train"], DataLoader)
    assert dataloaders["train"].batch_size == 8


# errors


def test_labels_map_errors(sample_dir):
    # csv
    annots = AnnotationsLoader()
    with pytest.raises(ValueError, match="not in the labels map"):
        annots.load(
            f"{sample_dir}/test_annots.csv",
            reset_index=True,
            remove_broken=False,
            ignore_broken=True,
            labels_map={0: "no"},
        )
    # dataframe
    annots = AnnotationsLoader()
    df = pd.read_csv(f"{sample_dir}/test_annots.csv", sep=",", index_col=0)
    with pytest.raises(ValueError, match="not in the labels map"):
        annots.load(
            df,
            reset_index=True,
            remove_broken=False,
            ignore_broken=True,
            labels_map={0: "no"},
        )


def test_load_fake_csv_errors():
    annots = AnnotationsLoader()
    with pytest.raises(ValueError, match="cannot be found"):
        annots.load("a_fake_file.csv")


def test_load_csv_errors(sample_dir):
    annots = AnnotationsLoader()
    with pytest.raises(ValueError, match="No annotations remaining"):
        annots.load(f"{sample_dir}/test_annots.csv")


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
