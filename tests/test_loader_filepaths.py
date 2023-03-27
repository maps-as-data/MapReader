from mapreader import loader
import pytest
import os

@pytest.fixture
def create_test_dir():
    mixed_dir_name="test_dir"
    png_dir_name="png_dir"
    if not os.path.exists(mixed_dir_name):
        os.mkdir(mixed_dir_name)
        files=['file1.png', 'file2.png', 'file3.png', 'file4.tif', 'file5.csv']
        for file in files:
            with open(f"{mixed_dir_name}/{file}", "w") as f:
                pass
        if not os.path.exists(f"{mixed_dir_name}/{png_dir_name}"):
            os.mkdir(f"{mixed_dir_name}/{png_dir_name}")
            files=['file1.png', 'file2.png']
            for file in files:
                with open(f"{mixed_dir_name}/{png_dir_name}/{file}", "w") as f:
                    pass

#mixed dir

def test_mixed_dir_w_file_ext(create_test_dir):
    my_files=loader("test_dir", file_ext="png")
    assert len(my_files)==3
    my_files=loader("test_dir/", file_ext="png")
    assert len(my_files)==3

def test_mixed_dir_w_star_w_file_ext(create_test_dir):
    my_files=loader("test_dir/*", file_ext="png")
    assert len(my_files)==3

def test_mixed_dir_w_file_path(create_test_dir):
    my_files=loader("test_dir/*png")
    assert len(my_files)==3

#unmixed dir

def test_unmixed_dir_no_file_ext(create_test_dir):
    my_files=loader("test_dir/png_dir")
    assert len(my_files)==2
    my_files=loader("test_dir/png_dir/")
    assert len(my_files)==2

def test_unmixed_dir_w_star_no_file_ext(create_test_dir):
    my_files=loader("test_dir/png_dir/*")
    assert len(my_files)==2

def test_unmixed_dir_w_file_path(create_test_dir):
    my_files=loader("test_dir/png_dir/*png")
    assert len(my_files)==2

#other test cases?

def test_file_path_w_file_ext():
    my_files=loader("test_dir/*png", file_ext="tif")
    assert len(my_files)==3