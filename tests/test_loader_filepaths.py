from mapreader import loader
import pytest
import os
import pathlib

@pytest.fixture
def mixed_dir(tmp_path):
    dir_path = tmp_path / "mixed_dir"
    os.mkdir(dir_path)
    files = ["file1.png", "file2.png", "file3.png", "file4.tif", "file5.csv"]
    for file in files:
        pathlib.Path(f"{dir_path}/{file}").touch()
    return dir_path

@pytest.fixture
def png_dir(tmp_path):
    dir_path = tmp_path / "png_dir"
    os.mkdir(dir_path)
    files = ["file1.png", "file2.png"]
    for file in files:
        pathlib.Path(f"{dir_path}/{file}").touch()
    return dir_path

@pytest.fixture
def empty_dir(tmp_path):
    dir_path = tmp_path / "empty_dir"
    os.mkdir(dir_path)
    return dir_path

# mixed dir

def test_mixed_dir_w_file_ext(mixed_dir):
    my_files = loader(mixed_dir, file_ext="png")
    assert len(my_files) == 3
    my_files = loader(f"{mixed_dir}/", file_ext="png")
    assert len(my_files) == 3

def test_mixed_dir_w_star_w_file_ext(mixed_dir):
    my_files = loader(f"{mixed_dir}/*", file_ext="png")
    assert len(my_files) == 3

def test_mixed_dir_w_file_path(mixed_dir):
    my_files = loader(f"{mixed_dir}/*png")
    assert len(my_files) == 3

def test_mixed_dir_errors(mixed_dir):
    with pytest.raises(ValueError):
        loader(mixed_dir)
    with pytest.raises(ValueError):
        loader(f"{mixed_dir}/")
        
# unmixed dir

def test_unmixed_dir_no_file_ext(png_dir):
    my_files = loader(png_dir)
    assert len(my_files) == 2
    my_files = loader(f"{png_dir}/")
    assert len(my_files) == 2

def test_unmixed_dir_w_star_no_file_ext(png_dir):
    my_files = loader(f"{png_dir}/*")
    assert len(my_files) == 2

def test_unmixed_dir_w_file_path(png_dir):
    my_files = loader(f"{png_dir}/*png")
    assert len(my_files) == 2

# other test cases?

def test_file_path_w_file_ext(mixed_dir):
    my_files = loader(f"{mixed_dir}/*png", file_ext="tif")
    assert len(my_files) == 3

def test_empty_dir_errors(empty_dir):
    with pytest.raises(ValueError):
        loader(empty_dir)
    with pytest.raises(ValueError):
        loader(f"{empty_dir}/")
    with pytest.raises(ValueError):
        loader(empty_dir, file_ext="png")
    with pytest.raises(ValueError):
        loader(f"{empty_dir}/*")
