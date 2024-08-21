from __future__ import annotations

import os
from pathlib import Path

import pytest

from mapreader import Annotator, loader


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent.parent / "sample_files"


@pytest.fixture
def load_dfs(sample_dir, tmp_path):
    my_maps = loader(f"{sample_dir}/cropped_74488689.png")
    my_maps.add_metadata(f"{sample_dir}/ts_downloaded_maps.csv")
    my_maps.patchify_all(
        patch_size=3, path_save=f"{tmp_path}/patches/"
    )  # creates 9 patches
    parent_df, patch_df = my_maps.convert_images()
    parent_df.to_csv(f"{tmp_path}/parent_df.csv")
    patch_df.to_csv(f"{tmp_path}/patch_df.csv")
    return parent_df, patch_df, tmp_path


def test_init_with_dfs(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
    )
    assert len(annotator) == 9
    assert isinstance(annotator.patch_df.iloc[0]["coordinates"], tuple)


def test_init_defaults(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    # test defaults
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
    )
    assert len(annotator) == 9
    assert annotator._labels == ["a", "b"]
    assert annotator.show_context is False
    assert annotator.border is False
    assert annotator.auto_save is True
    assert annotator._sortby is None
    assert annotator._ascending is True
    assert annotator.task_name == "task"
    assert annotator._min_values == {}
    assert annotator._max_values == {}
    assert annotator._filter_for is None
    assert annotator.surrounding == 1
    assert annotator.max_size == 1000
    assert annotator.resize_to is None

    annotator.annotate()
    assert annotator.show_context is False
    assert annotator.border is False
    assert annotator._sortby is None
    assert annotator._ascending is True
    assert annotator._min_values == {}
    assert annotator._max_values == {}
    assert annotator.surrounding == 1
    assert annotator.max_size == 1000
    assert annotator.resize_to is None
    assert annotator.show_vals is None

    annotator.annotate(
        show_context=True,
        border=True,
        sortby="min_x",
        ascending=False,
        min_values={"min_x": 0},
        max_values={"min_x": 10},
        surrounding=2,
        resize_to=300,
        max_size=500,
        show_vals=["label"],
    )
    assert annotator.show_context is True
    assert annotator.border is True
    assert annotator._sortby == "min_x"
    assert annotator._ascending is False
    assert annotator._min_values == {"min_x": 0}
    assert annotator._max_values == {"min_x": 10}
    assert annotator.surrounding == 2
    assert annotator.max_size == 500
    assert annotator.resize_to == 300
    assert annotator.show_vals == ["label"]


def test_init_optional_args(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    # add optional arguments
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        show_context=True,
        border=True,
        auto_save=False,
        sortby="min_x",
        ascending=False,
        username="rosie",
        task_name="test",
        surrounding=2,
        max_size=500,
        resize_to=300,
    )
    assert len(annotator) == 9
    assert annotator._labels == ["a", "b"]
    assert annotator.show_context is True
    assert annotator.border is True
    assert annotator.auto_save is False
    assert annotator._sortby == "min_x"
    assert annotator._ascending is False
    assert annotator.username == "rosie"
    assert annotator.task_name == "test"
    assert annotator.surrounding == 2
    assert annotator.max_size == 500
    assert annotator.resize_to == 300

    annotator.annotate()
    assert annotator.show_context is True
    assert annotator.border is True
    assert annotator._sortby == "min_x"
    assert annotator._ascending is False
    assert annotator.surrounding == 2
    assert annotator.max_size == 500
    assert annotator.resize_to == 300

    annotator.annotate(
        show_context=False,
        border=False,
        sortby="max_x",
        ascending=True,
        surrounding=1,
        resize_to=400,
        max_size=1000,
    )
    assert annotator.show_context is False
    assert annotator.border is False
    assert annotator._sortby == "max_x"
    assert annotator._ascending is True
    assert annotator.surrounding == 1
    assert annotator.max_size == 1000
    assert annotator.resize_to == 400


def test_init_dfs_value_error(load_dfs):
    with pytest.raises(ValueError, match="path to a csv or a pandas DataFrame"):
        Annotator(
            patch_df=1,
            parent_df=1,
        )
    _, _, tmp_path = load_dfs
    with pytest.raises(ValueError, match="path to a csv or a pandas DataFrame"):
        Annotator(
            patch_df=f"{tmp_path}/patch_df.csv",
            parent_df=1,
        )
    with pytest.raises(ValueError, match="Please specify one"):
        Annotator()


def test_init_n_labels(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    for i in range(9):
        labels = [str(x) for x in range(i)]
        if (i % 2) == 0:
            if i > 4:
                expected = 4
            else:
                expected = 2
        else:
            if i == 3:
                expected = 3
            else:
                expected = 5

        annotator = Annotator(
            patch_df=patch_df,
            parent_df=parent_df,
            labels=labels,
            annotations_dir=f"{tmp_path}/annotations/",
            auto_save=False,
        )
        assert len(annotator) == 9
        assert annotator._labels == labels
        assert annotator.buttons_per_row == expected


def test_init_with_csvs(load_dfs):
    _, _, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=f"{tmp_path}/patch_df.csv",
        parent_df=f"{tmp_path}/parent_df.csv",
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
    )
    assert len(annotator) == 9
    assert isinstance(annotator.patch_df.iloc[0]["coordinates"], tuple)


def test_init_with_fpaths(load_dfs, sample_dir):
    _, _, tmp_path = load_dfs
    annotator = Annotator(
        patch_paths=f"{tmp_path}/patches/*png",
        parent_paths=f"{sample_dir}/cropped_74488689.png",
        metadata_path=f"{sample_dir}/ts_downloaded_maps.csv",
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
    )
    assert len(annotator) == 9
    assert "mean_pixel_R" in annotator.patch_df.columns


def test_incorrect_csv_paths(load_dfs):
    with pytest.raises(FileNotFoundError):
        Annotator(
            patch_df="fake_df.csv",
            parent_df="fake_df.csv",
        )
    _, _, tmp_path = load_dfs
    with pytest.raises(FileNotFoundError):
        Annotator(
            patch_df=f"{tmp_path}/patch_df.csv",
            parent_df="fake_df.csv",
        )


def test_fpaths_metadata_filenotfound_error(load_dfs, sample_dir):
    _, _, tmp_path = load_dfs
    with pytest.raises(FileNotFoundError):
        Annotator(
            patch_paths=f"{tmp_path}/patches/*png",
            parent_paths=f"{sample_dir}/cropped_74488689.png",
            metadata_path="fake_df.csv",
        )


def test_init_with_fpaths_tsv(load_dfs, sample_dir):
    _, _, tmp_path = load_dfs
    annotator = Annotator(
        patch_paths=f"{tmp_path}/patches/*png",
        parent_paths=f"{sample_dir}/cropped_74488689.png",
        metadata_path=f"{sample_dir}/ts_downloaded_maps.tsv",
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
        delimiter="\t",
    )
    assert len(annotator) == 9
    assert "mean_pixel_R" in annotator.patch_df.columns


def test_incorrect_delimiter(load_dfs):
    _, _, tmp_path = load_dfs
    with pytest.raises(ValueError):
        Annotator(
            patch_df=f"{tmp_path}/patch_df.csv",
            parent_df=f"{tmp_path}/parent_df.csv",
            delimiter="|",
        )


def test_no_labels(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
    )
    assert len(annotator) == 9
    assert annotator._labels == []

    annotator._labels = ["a", "b"]
    assert annotator._labels == ["a", "b"]


def test_str_labels(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    with pytest.raises(SyntaxError):
        Annotator(
            patch_df=patch_df,
            parent_df=parent_df,
            labels="a, b",
            annotations_dir=f"{tmp_path}/annotations/",
            auto_save=False,
        )


def test_duplicate_labels(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b", "a"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
    )
    assert len(annotator) == 9
    assert annotator._labels == ["a", "b"]


def test_labels_sorting(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["b", "a"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
    )
    assert len(annotator) == 9
    assert annotator._labels == ["b", "a"]


def test_sortby(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
        sortby="min_x",
        ascending=False,
    )
    queue = annotator._get_queue()
    assert len(queue) == 9
    assert queue[0] == "patch-6-0-9-3-#cropped_74488689.png#.png"
    assert queue[-1] == "patch-0-6-3-9-#cropped_74488689.png#.png"


def test_sortby_value_errors(load_dfs):
    parent_df, patch_df, _ = load_dfs
    with pytest.raises(ValueError, match="not a column"):
        Annotator(
            patch_df=patch_df,
            parent_df=parent_df,
            sortby="fake_col",
        )
    with pytest.raises(ValueError, match="must be a string or None"):
        Annotator(
            patch_df=patch_df,
            parent_df=parent_df,
            sortby=1,
        )


def test_min_values(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
        min_values={"min_x": 3},
        sortby="min_x",  # no shuffle
    )
    queue = annotator._get_queue()
    assert len(queue) == 6
    assert queue[0] == "patch-3-0-6-3-#cropped_74488689.png#.png"
    assert queue[-1] == "patch-6-6-9-9-#cropped_74488689.png#.png"


def test_max_values(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
        max_values={"min_x": 0},
        sortby="min_x",  # no shuffle
    )
    queue = annotator._get_queue()
    assert len(queue) == 3
    assert queue[0] == "patch-0-0-3-3-#cropped_74488689.png#.png"
    assert queue[-1] == "patch-0-6-3-9-#cropped_74488689.png#.png"


def test_filter_for(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
        filter_for={"min_y": 0},
        sortby="min_x",  # no shuffle
    )
    queue = annotator._get_queue()
    assert len(queue) == 3
    assert queue[0] == "patch-0-0-3-3-#cropped_74488689.png#.png"
    assert queue[-1] == "patch-6-0-9-3-#cropped_74488689.png#.png"


def test_max_size(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
        max_size=2,
    )
    assert annotator.max_size == 2
    annotator.annotate()


def test_no_image_path_col(load_dfs):
    parent_df, patch_df, _ = load_dfs
    patch_df = patch_df.drop(columns=["image_path"])
    with pytest.raises(ValueError, match="does not have the image paths column"):
        Annotator(
            patch_df=patch_df,
            parent_df=parent_df,
        )


def test_unknown_arg_error(load_dfs):
    parent_df, patch_df, _ = load_dfs
    with pytest.raises(TypeError):
        Annotator(
            patch_df=patch_df,
            parent_df=parent_df,
            fake_arg=1,
        )


def test_filtered(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=False,
    )
    out = annotator.filtered
    assert len(out) == 0
    annotator.patch_df["label"] = "a"
    out = annotator.filtered
    assert len(out) == 9


def test_add_annotation(load_dfs):
    parent_df, patch_df, tmp_path = load_dfs
    annotator = Annotator(
        patch_df=patch_df,
        parent_df=parent_df,
        labels=["a", "b"],
        annotations_dir=f"{tmp_path}/annotations/",
        auto_save=True,
    )
    annotator.annotate()
    patch = annotator._queue[0]
    patch_idx = annotator.current_index
    annotator._add_annotation("a")
    assert annotator.patch_df.loc[patch, "label"] == "a"
    assert annotator.current_index != patch_idx
    assert os.path.isfile(annotator.annotations_file)
