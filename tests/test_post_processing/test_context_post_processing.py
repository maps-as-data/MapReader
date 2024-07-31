from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mapreader.process.context_post_process import ContextPostProcessor


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
def patch_df(sample_dir):
    return pd.read_csv(f"{sample_dir}/post_processing_patch_df.csv", index_col=0)


@pytest.fixture
def labels_map():
    return {0: "no", 1: "railspace", 2: "building", 3: "railspace&building"}


def test_init(labels_map, patch_df):
    patches = ContextPostProcessor(patch_df, labels_map=labels_map)
    assert isinstance(patches, ContextPostProcessor)
    assert len(patches) == 81
    assert patches.labels_map == labels_map


def test_init_errors(patch_df, labels_map):
    with pytest.raises(ValueError, match="must contain the following columns"):
        ContextPostProcessor(patch_df.drop(columns=["parent_id", "pred"]), labels_map)


def test_init_imaged_id_col(patch_df, labels_map):
    # e.g. if you have integer index
    patches = ContextPostProcessor(patch_df.reset_index(drop=False), labels_map)
    assert isinstance(patches, ContextPostProcessor)
    assert patches.patch_df.index.name == "image_id"

    patch_df["image_id"] = patch_df.index
    patches = ContextPostProcessor(patch_df, labels_map)
    assert isinstance(patches, ContextPostProcessor)
    assert patches.patch_df.index.name == "image_id"
    assert len(patches) == 81
    assert "image_id" in patches.patch_df.columns


def test_get_context(patch_df, labels_map):
    patches = ContextPostProcessor(patch_df, labels_map=labels_map)
    # labels as str
    patches.get_context("railspace")
    assert len(patches.context) == 10
    # labels as list
    patches.get_context(["railspace", "railspace&building"])
    assert len(patches.context) == 13


def test_update_preds_railspace(patch_df, labels_map):
    patches = ContextPostProcessor(patch_df, labels_map=labels_map)
    patches.get_context(["railspace"])
    remap = {"railspace": "no"}
    patches.update_preds(remap)
    assert len(patches.patch_df[patches.patch_df["new_predicted_label"].notna()]) == 1
    assert (
        patches.patch_df.loc[
            "patch-4-7-5-8-#cropped_74488689.png#.png", "new_predicted_label"
        ]
        == "no"
    )

    patches.patch_df.drop(columns=["new_predicted_label", "new_pred"], inplace=True)
    patches.update_preds(remap, conf=0.8)
    assert len(patches.patch_df[patches.patch_df["new_predicted_label"].notna()]) == 2
    assert (
        patches.patch_df.loc[
            "patch-1-7-2-8-#cropped_74488689.png#.png", "new_predicted_label"
        ]
        == "no"
    )

    patches.patch_df.drop(columns=["new_predicted_label", "new_pred"], inplace=True)
    patches.update_preds(remap, conf=1)  # all conf == 1 should remain unchanged
    assert len(patches.patch_df[patches.patch_df["new_predicted_label"].notna()]) == 3
    assert (
        patches.patch_df.loc[
            "patch-1-4-2-5-#cropped_74488689.png#.png", "new_predicted_label"
        ]
        == "no"
    )


def test_update_preds_railspace_railspace_building(patch_df, labels_map):
    patches = ContextPostProcessor(patch_df, labels_map=labels_map)
    patches.get_context(["railspace", "railspace&building"])
    remap = {"railspace": "no", "railspace&building": "building"}
    patches.update_preds(remap)
    assert len(patches.patch_df[patches.patch_df["new_predicted_label"].notna()]) == 1
    assert (
        patches.patch_df.loc[
            "patch-7-7-8-8-#cropped_74488689.png#.png", "new_predicted_label"
        ]
        == "building"
    )

    patches.patch_df.drop(columns=["new_predicted_label", "new_pred"], inplace=True)
    patches.update_preds(remap, conf=0.8)
    assert len(patches.patch_df[patches.patch_df["new_predicted_label"].notna()]) == 2
    assert (
        patches.patch_df.loc[
            "patch-1-7-2-8-#cropped_74488689.png#.png", "new_predicted_label"
        ]
        == "no"
        
    )
    assert (
        patches.patch_df.loc[
            "patch-7-7-8-8-#cropped_74488689.png#.png", "new_predicted_label"
        ]
        == "building"
    )


def test_update_preds_inplace(patch_df, labels_map):
    patches = ContextPostProcessor(patch_df, labels_map=labels_map)
    patches.get_context(["railspace"])
    remap = {"railspace": "no"}
    patches.update_preds(remap, inplace=True)
    assert "new_predicted_label" not in patches.patch_df.columns
    assert (
        patches.patch_df.loc[
            "patch-4-7-5-8-#cropped_74488689.png#.png", "predicted_label"
        ]
        == "no"
    )


def test_update_preds_new_label(patch_df, labels_map):
    patches = ContextPostProcessor(patch_df, labels_map=labels_map)
    patches.get_context(["railspace"])
    remap = {"railspace": "new"}
    patches.update_preds(remap)
    assert len(patches.patch_df[patches.patch_df["new_predicted_label"].notna()]) == 1
    assert (
        patches.patch_df.loc[
            "patch-4-7-5-8-#cropped_74488689.png#.png", "new_predicted_label"
        ]
        == "new"
    )
    assert (
        patches.patch_df.loc["patch-4-7-5-8-#cropped_74488689.png#.png", "new_pred"]
        == 4
    )
    assert patches.labels_map[4] == "new"


def test_update_preds_errors(patch_df, labels_map):
    patches = ContextPostProcessor(patch_df, labels_map=labels_map)
    remap = {"railspace": "no"}
    with pytest.raises(ValueError, match="run `get_context` first"):
        patches.update_preds(remap)

    patches.get_context(["fake"])
    with pytest.raises(ValueError, match="No patches to update"):
        patches.update_preds(remap)

    patches.get_context(["railspace"])
    with pytest.raises(ValueError, match="must specify a remap"):
        patches.update_preds(remap={"fake": "no"})
