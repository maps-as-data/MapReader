from mapreader import loader, load_patches
import pytest
import os
import pathlib
from PIL import Image
from random import randint

@pytest.fixture
def test_dirs(tmp_path):
    parent_path = tmp_path / "test_parent_dir"
    patch_path = parent_path / "patch_dir"
    os.mkdir(parent_path)
    files = ["file1.png", "file2.png", "file3.png"]
    for file in files:
        rand_colour = (randint(0,255), randint(0,255), randint(0,255))
        Image.new("RGB",(9,9), color = rand_colour).save(f"{parent_path}/{file}")
    pathlib.Path(f"{parent_path}/file4.csv").touch()
    os.mkdir(patch_path)
    files = [
        "patch1-0-1-2-3-#file1.png#.png",
        "patch2-4-5-6-7-#file1.png#.png",
        "patch1-0-1-2-3-#file2.png#.png",
    ]
    for file in files:
        rand_colour = (randint(0,255), randint(0,255), randint(0,255))
        Image.new("RGB",(9,9), color = rand_colour).save(f"{patch_path}/{file}")
    return parent_path, patch_path

@pytest.fixture
def empty_dir(tmp_path):
    dir_path = tmp_path / "empty_dir"
    os.mkdir(dir_path)
    return dir_path

#loadParents

def test_unmixed_paths(test_dirs):
    _, patch_path = test_dirs
    my_files = loader()
    my_files.loadParents(patch_path)
    assert len(my_files) == 3
    my_files = loader()
    my_files.loadParents(f"{patch_path}/")
    assert len(my_files) == 3
    my_files = loader()
    my_files.loadParents(f"{patch_path}/*")
    assert len(my_files) == 3

def test_file_ext_w_mixed_paths(test_dirs):
    parent_path, _ = test_dirs
    my_files = loader()
    my_files.loadParents(parent_path, parent_file_ext = "png")
    assert len(my_files) == 3
    my_files = loader()
    my_files.loadParents(f"{parent_path}/", parent_file_ext = "png")
    assert len(my_files) == 3
    my_files = loader()
    my_files.loadParents(f"{parent_path}/*", parent_file_ext = "png")
    assert len(my_files) == 3

# errors 

def test_multiple_file_types_errors(test_dirs):
    parent_path, _ = test_dirs
    with pytest.raises(ValueError, match="multiple file types"):
        my_files = loader()
        my_files.loadParents(parent_path)
    with pytest.raises(ValueError, match="multiple file types"):
        my_files = loader()
        my_files.loadParents(f"{parent_path}/")
    with pytest.raises(ValueError, match="multiple file types"):
        my_files = loader()
        my_files.loadParents(f"{parent_path}/*")

def test_no_files_found_errors(test_dirs):
    parent_path, _ = test_dirs
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.loadParents(parent_path, parent_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.loadParents(f"{parent_path}/*", parent_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.loadParents(f"{parent_path}/*png", parent_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.loadParents(f"{parent_path}/*tif")
        
def test_loadParents_empty_dir(empty_dir):
    with pytest.raises(ValueError):
        my_files = loader()
        my_files.loadParents(empty_dir)
    with pytest.raises(ValueError):
        my_files = loader()
        my_files.loadParents(f"{empty_dir}/")
    with pytest.raises(ValueError):
        my_files = loader()
        my_files.loadParents(empty_dir, parent_file_ext="png")
    with pytest.raises(ValueError):
        my_files = loader()
        my_files.loadParents(f"{empty_dir}/*")
