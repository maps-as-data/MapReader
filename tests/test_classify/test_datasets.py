from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from mapreader import AnnotationsLoader, loader
from mapreader.classify.datasets import PatchContextDataset, PatchDataset


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
def annots(sample_dir):
    annots = AnnotationsLoader()
    annots.load(
        f"{sample_dir}/test_annots.csv", remove_broken=False, ignore_broken=True
    )
    return annots


@pytest.fixture
def load_dfs(sample_dir, tmp_path):
    my_maps = loader(f"{sample_dir}/cropped_74488689.png")
    my_maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    my_maps.patchify_all(
        patch_size=3, path_save=f"{tmp_path}/patches/"
    )  # creates 9 patches
    _, patch_df = my_maps.convert_images()
    patch_df.to_csv(f"{tmp_path}/patch_df.csv")
    return patch_df, tmp_path


# patch dataset


def test_patch_dataset_init_df(load_dfs):
    patch_df, tmp_path = load_dfs
    patch_dataset = PatchDataset(patch_df=patch_df, transform="test")
    assert isinstance(patch_dataset, PatchDataset)
    assert patch_dataset.patch_df.equals(patch_df)
    assert patch_dataset.label_col is None


def test_patch_dataset_init_string(load_dfs):
    patch_df, tmp_path = load_dfs
    patch_dataset = PatchDataset(
        patch_df=f"{tmp_path}/patch_df.csv",
        transform="test",
    )
    assert isinstance(patch_dataset, PatchDataset)
    for col in patch_df.columns:
        if col == "polygon":
            continue  # polygon column is converted to polygon type
        assert patch_df[col].equals(patch_dataset.patch_df[col])
    assert patch_dataset.label_col is None


def test_patch_dataset_init_annots(annots):
    patch_dataset = PatchDataset(
        patch_df=annots.annotations, transform="test", label_col="label"
    )
    assert isinstance(patch_dataset, PatchDataset)
    assert patch_dataset.patch_df.equals(annots.annotations)
    assert patch_dataset.label_col == "label"
    assert patch_dataset.unique_labels == ["no", "railspace"]


def test_create_dataloaders(load_dfs):
    patch_df, tmp_path = load_dfs
    patch_dataset = PatchDataset(
        patch_df=patch_df,
        transform="test",
    )
    dataloaders = patch_dataset.create_dataloaders(
        "a_test",
        batch_size=2,
        shuffle=False,
    )
    assert isinstance(dataloaders["a_test"], DataLoader)
    assert dataloaders["a_test"].batch_size == 2


# patch context dataset


def test_patch_context_dataset_init_df(load_dfs):
    patch_df, tmp_path = load_dfs
    patch_dataset = PatchContextDataset(
        patch_df=patch_df[:-3],
        total_df=patch_df,
        transform="test",
        create_context=True,
    )
    assert isinstance(patch_dataset, PatchContextDataset)
    assert patch_dataset.patch_df.equals(patch_df[:-3])
    assert patch_dataset.label_col is None


def test_patch_context_dataset_init_string(load_dfs):
    patch_df, tmp_path = load_dfs
    patch_dataset = PatchContextDataset(
        patch_df=f"{tmp_path}/patch_df.csv",
        total_df=f"{tmp_path}/patch_df.csv",
        transform="test",
        create_context=True,
    )
    assert isinstance(patch_dataset, PatchContextDataset)
    for col in patch_df.columns:
        if col == "polygon":
            continue  # polygon column is not converted to polygon type
        assert patch_df[col].equals(patch_dataset.patch_df[col])
    assert patch_dataset.label_col is None


def test_patch_context_dataset_init_annots(annots):
    patch_dataset = PatchContextDataset(
        patch_df=annots.annotations,
        total_df=annots.annotations,
        transform="test",
        label_col="label",
        create_context=True,
    )
    assert isinstance(patch_dataset, PatchContextDataset)
    assert patch_dataset.patch_df.equals(annots.annotations)
    assert patch_dataset.label_col == "label"
    assert patch_dataset.unique_labels == ["no", "railspace"]


def test_create_context_dataloaders(load_dfs):
    patch_df, tmp_path = load_dfs
    patch_dataset = PatchContextDataset(
        patch_df=patch_df,
        total_df=patch_df,
        transform="test",
    )
    dataloaders = patch_dataset.create_dataloaders(
        "a_test",
        batch_size=2,
        shuffle=False,
    )
    assert isinstance(dataloaders["a_test"], DataLoader)
    assert dataloaders["a_test"].batch_size == 2
