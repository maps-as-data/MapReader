from __future__ import annotations

import os
import pathlib
from random import randint

import pytest
from PIL import Image

from mapreader import loader


@pytest.fixture
def dirs(tmp_path):
    parent_path = tmp_path / "test_parent_dir"
    patch_path = parent_path / "patch_dir"
    os.mkdir(parent_path)
    files = ["file1.png", "file2.png", "file3.png"]
    for file in files:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        Image.new("RGB", (9, 9), color=rand_color).save(f"{parent_path}/{file}")
    pathlib.Path(f"{parent_path}/file4.csv").touch()
    os.mkdir(patch_path)
    files = [
        "patch1-0-1-2-3-#file1.png#.png",
        "patch2-4-5-6-7-#file1.png#.png",
        "patch1-0-1-2-3-#file2.png#.png",
    ]
    for file in files:
        rand_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        Image.new("RGB", (9, 9), color=rand_color).save(f"{patch_path}/{file}")
    return parent_path, patch_path


@pytest.fixture
def empty_dir(tmp_path):
    dir_path = tmp_path / "empty_dir"
    os.mkdir(dir_path)
    return dir_path


# load_parents


def test_unmixed_paths(dirs):
    _, patch_path = dirs
    my_files = loader()
    my_files.load_parents(patch_path)
    assert len(my_files) == 3
    my_files = loader()
    my_files.load_parents(f"{patch_path}/")
    assert len(my_files) == 3
    my_files = loader()
    my_files.load_parents(f"{patch_path}/*")
    assert len(my_files) == 3


def test_file_ext_w_mixed_paths(dirs):
    parent_path, _ = dirs
    my_files = loader()
    my_files.load_parents(parent_path, parent_file_ext="png")
    assert len(my_files) == 3
    my_files = loader()
    my_files.load_parents(f"{parent_path}/", parent_file_ext="png")
    assert len(my_files) == 3


# errors


def test_multiple_file_types_errors(dirs):
    parent_path, _ = dirs
    with pytest.raises(ValueError, match="Non-image file types"):
        my_files = loader()
        my_files.load_parents(parent_path)
    with pytest.raises(ValueError, match="Non-image file types"):
        my_files = loader()
        my_files.load_parents(f"{parent_path}/")
    with pytest.raises(ValueError, match="Non-image file types"):
        my_files = loader()
        my_files.load_parents(f"{parent_path}/*")


def test_no_files_found_errors(dirs):
    parent_path, _ = dirs
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.load_parents(parent_path, parent_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.load_parents(f"{parent_path}/*tif")


def test_ignore_file_ext(dirs):
    parent_path, _ = dirs
    my_files = loader()
    my_files.load_parents(f"{parent_path}/*png", parent_file_ext="tif")
    assert len(my_files) == 3


def test_load_parents_empty_dir(empty_dir):
    with pytest.raises(ValueError):
        my_files = loader()
        my_files.load_parents(empty_dir)
    with pytest.raises(ValueError):
        my_files = loader()
        my_files.load_parents(f"{empty_dir}/")
    with pytest.raises(ValueError):
        my_files = loader()
        my_files.load_parents(empty_dir, parent_file_ext="png")
    with pytest.raises(ValueError):
        my_files = loader()
        my_files.load_parents(f"{empty_dir}/*")
