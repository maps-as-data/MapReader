from mapreader import loader
import pytest
import os
import pathlib

@pytest.fixture
def test_dirs(tmp_path):
    parent_path = tmp_path / "test_parent_dir"
    patch_path = parent_path / "patch_dir"
    os.mkdir(parent_path)
    files = ["file1.png", "file2.png", "file3.png", "file5.csv"]
    for file in files:
        pathlib.Path(f"{parent_path}/{file}").touch()
    os.mkdir(patch_path)
    files = [
        "patch1-0-1-2-3-#file1.png#.png",
        "patch2-4-5-6-7-#file1.png#.png",
        "patch1-0-1-2-3-#file2.png#.png",
    ]
    for file in files:
        pathlib.Path(f"{patch_path}/{file}").touch()
    return parent_path, patch_path

@pytest.fixture
def empty_dir(tmp_path):
    dir_path = tmp_path / "empty_dir"
    os.mkdir(dir_path)
    return dir_path

#load_parents

def test_unmixed_paths(test_dirs):
    _, patch_path = test_dirs
    my_files = loader()
    my_files.load_parents(patch_path)
    assert len(my_files) == 3
    my_files = loader()
    my_files.load_parents(f"{patch_path}/")
    assert len(my_files) == 3
    my_files = loader()
    my_files.load_parents(f"{patch_path}/*")
    assert len(my_files) == 3

def test_file_ext_w_mixed_paths(test_dirs):
    parent_path, _ = test_dirs
    my_files = loader()
    my_files.load_parents(parent_path, parent_file_ext = "png")
    assert len(my_files) == 3
    my_files = loader()
    my_files.load_parents(f"{parent_path}/", parent_file_ext = "png")
    assert len(my_files) == 3
    my_files = loader()
    my_files.load_parents(f"{parent_path}/*", parent_file_ext = "png")
    assert len(my_files) == 3

# errors 

def test_multiple_file_types_errors(test_dirs):
    parent_path, _ = test_dirs
    with pytest.raises(ValueError, match="multiple file types"):
        my_files = loader()
        my_files.load_parents(parent_path)
    with pytest.raises(ValueError, match="multiple file types"):
        my_files = loader()
        my_files.load_parents(f"{parent_path}/")
    with pytest.raises(ValueError, match="multiple file types"):
        my_files = loader()
        my_files.load_parents(f"{parent_path}/*")

def test_no_files_found_errors(test_dirs):
    parent_path, _ = test_dirs
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.load_parents(parent_path, parent_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.load_parents(f"{parent_path}/*", parent_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.load_parents(f"{parent_path}/*png", parent_file_ext="tif")
    with pytest.raises(ValueError, match="No files found"):
        my_files = loader()
        my_files.load_parents(f"{parent_path}/*tif")
        
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
