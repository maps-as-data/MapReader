from mapreader import load_patches
import pytest
import os
import pathlib
from PIL import Image
from random import randint

@pytest.fixture
def dirs(tmp_path):
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

#load_patches() is just a wrapper for MapImages.load_patches()

#load just patches

def test_no_parents(dirs):
    _, patch_path = dirs
    my_files = load_patches(patch_path)
    assert len(my_files) == 5 #file1.png and file2.png will be added as parents (w/ no file path)
    my_files = load_patches(f"{patch_path}/")
    assert len(my_files) == 5
    my_files = load_patches(f"{patch_path}/*")
    assert len(my_files) == 5
    my_files = load_patches(f"{patch_path}/*png")
    assert len(my_files) == 5
    my_files = load_patches(patch_path, patch_file_ext="png")
    assert len(my_files) == 5
    my_files = load_patches(f"{patch_path}/*", patch_file_ext="png")
    assert len(my_files) == 5

#load patches and parents

def test_w_parent_dir_and_parent_file_ext(dirs):
    parent_path, patch_path = dirs
    my_files = load_patches(patch_path, parent_paths=parent_path, parent_file_ext="png")
    assert len(my_files) == 6
    my_files = load_patches(patch_path, parent_paths=f"{parent_path}/", parent_file_ext="png")
    assert len(my_files) == 6
    my_files = load_patches(patch_path, parent_paths=f"{parent_path}/*", parent_file_ext="png")
    assert len(my_files) == 6

def test_w_parent_file_paths(dirs):
    parent_path, patch_path = dirs
    my_files = load_patches(patch_path, parent_paths=f"{parent_path}/*png")
    assert len(my_files) == 6

# other test cases

# errors

def test_multiple_file_types_errors(dirs):
    parent_path, patch_path = dirs
    with pytest.raises(ValueError, match="multiple file types"):
        load_patches(parent_path)
    with pytest.raises(ValueError, match="multiple file types"):
        load_patches(f"{parent_path}/")
    with pytest.raises(ValueError, match="multiple file types"):
        load_patches(f"{parent_path}/*")

def test_no_files_found_errors(dirs):
    _, patch_path = dirs
    with pytest.raises(ValueError, match="No files found"):
        load_patches(patch_path, patch_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        load_patches(f"{patch_path}/*", patch_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        load_patches(f"{patch_path}/*png", patch_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        load_patches(f"{patch_path}/*tif")

def test_empty_dir_errors(empty_dir):
    with pytest.raises(ValueError, match="No files found"):
        load_patches(empty_dir)
    with pytest.raises(ValueError, match="No files found"):
        load_patches(f"{empty_dir}/")
    with pytest.raises(ValueError, match="No files found"):
        load_patches(empty_dir, patch_file_ext="png")
    with pytest.raises(ValueError, match="No files found"):
        load_patches(f"{empty_dir}/*")
