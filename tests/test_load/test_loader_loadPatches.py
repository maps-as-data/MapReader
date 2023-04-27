from mapreader import load_patches
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

#load_patches() is just a wrapper for MapImages.load_patches()

#load just patches

def test_no_parents(test_dirs):
    _, patch_path = test_dirs
    my_files = load_patches(patch_path)
    assert len(my_files) == 3
    my_files = load_patches(f"{patch_path}/")
    assert len(my_files) == 3
    my_files = load_patches(f"{patch_path}/*")
    assert len(my_files) == 3
    my_files = load_patches(f"{patch_path}/*png")
    assert len(my_files) == 3
    my_files = load_patches(patch_path, patch_file_ext="png")
    assert len(my_files) == 3
    my_files = load_patches(f"{patch_path}/*", patch_file_ext="png")
    assert len(my_files) == 3

#load patches and parents

def test_w_parent_dir_and_parent_file_ext(test_dirs):
    parent_path, patch_path = test_dirs
    my_files = load_patches(patch_path, parent_paths=parent_path, parent_file_ext="png")
    assert len(my_files) == 6
    my_files = load_patches(patch_path, parent_paths=f"{parent_path}/", parent_file_ext="png")
    assert len(my_files) == 6
    my_files = load_patches(patch_path, parent_paths=f"{parent_path}/*", parent_file_ext="png")
    assert len(my_files) == 6

def test_w_parent_file_paths(test_dirs):
    parent_path, patch_path = test_dirs
    my_files = load_patches(patch_path, parent_paths=f"{parent_path}/*png")
    assert len(my_files) == 6

# other test cases

# errors

def test_multiple_file_types_errors(test_dirs):
    parent_path, patch_path = test_dirs
    with pytest.raises(ValueError, match="multiple file types"):
        load_patches(parent_path)
    with pytest.raises(ValueError, match="multiple file types"):
        load_patches(f"{parent_path}/")
    with pytest.raises(ValueError, match="multiple file types"):
        load_patches(f"{parent_path}/*")

def test_no_files_found_errors(test_dirs):
    _, patch_path = test_dirs
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