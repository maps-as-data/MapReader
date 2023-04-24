from mapreader import loader
import pytest
import os
from pathlib import Path
import shutil

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent / "sample_files"

@pytest.fixture
def mixed_dir(sample_dir, tmp_path):
    dir_path = tmp_path / "mixed_dir"
    os.mkdir(dir_path)
    files = [f"{sample_dir}/map_74488693.png", f"{sample_dir}/101101409.1_JPEG.tif", f"{sample_dir}/101200740.27_JPEG.tif", f"{sample_dir}/ts_downloaded_maps.csv"]
    for file in files:
        shutil.copy(file, dir_path)
    return dir_path

@pytest.fixture
def tiff_dir(tmp_path, sample_dir):
    dir_path = tmp_path / "tiff_dir"
    os.mkdir(dir_path)
    files = [f"{sample_dir}/101101409.1_JPEG.tif", f"{sample_dir}/101200740.27_JPEG.tif"]
    for file in files:
        shutil.copy(file, dir_path)
    return dir_path

@pytest.fixture
def empty_dir(tmp_path):
    dir_path = tmp_path / "empty_dir"
    os.mkdir(dir_path)
    return dir_path

# mixed dir

def test_file_ext_w_mixed_dir(mixed_dir):
    my_files = loader(mixed_dir, file_ext="tif")
    assert len(my_files) == 2
    my_files = loader(f"{mixed_dir}/", file_ext="tif")
    assert len(my_files) == 2

def test_file_ext_w_mixed_file_paths(mixed_dir):
    my_files = loader(f"{mixed_dir}/*", file_ext="tif")
    assert len(my_files) == 2

def test_mixed_file_path(mixed_dir):
    my_files = loader(f"{mixed_dir}/*tif")
    assert len(my_files) == 2

# unmixed dir

def test_unmixed_dir(tiff_dir):
    my_files = loader(tiff_dir)
    assert len(my_files) == 2
    my_files = loader(f"{tiff_dir}/")
    assert len(my_files) == 2

def test_unmixed_file_path(tiff_dir):
    my_files = loader(f"{tiff_dir}/*")
    assert len(my_files) == 2
    my_files = loader(f"{tiff_dir}/*tif")
    assert len(my_files) == 2

# other test cases?

#errors 

def test_multiple_file_types_errors(mixed_dir):
    with pytest.raises(ValueError, match="multiple file types"):
        loader(mixed_dir)
    with pytest.raises(ValueError, match="multiple file types"):
        loader(f"{mixed_dir}/")
    with pytest.raises(ValueError, match="multiple file types"):
        loader(f"{mixed_dir}/*")

def test_no_files_found_errors(tiff_dir):
    with pytest.raises(ValueError, match="No files found"):
        loader(tiff_dir, file_ext = "png")
    with pytest.raises(ValueError, match="No files found"):
        loader(f"{tiff_dir}/*tif", file_ext = "png")
    with pytest.raises(ValueError, match="No files found"):
        loader(f"{tiff_dir}/*png")
    
def test_empty_dir_errors(empty_dir):
    with pytest.raises(ValueError, match="No files found"):
        loader(empty_dir)
    with pytest.raises(ValueError, match="No files found"):
        loader(f"{empty_dir}/")
    with pytest.raises(ValueError, match="No files found"):
        loader(empty_dir, file_ext="png")
    with pytest.raises(ValueError, match="No files found"):
        loader(f"{empty_dir}/*")
