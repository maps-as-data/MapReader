from mapreader import loader, load_patches
import pytest
import os


@pytest.fixture
def create_test_dir():
    parent_dir_name = "test_parent_dir"
    patch_dir_name = "patch_dir"
    if not os.path.exists(parent_dir_name):
        os.mkdir(parent_dir_name)
        files = ["file1.png", "file2.png", "file3.png", "file5.csv"]
        for file in files:
            with open(f"{parent_dir_name}/{file}", "w") as f:
                pass
        if not os.path.exists(f"{parent_dir_name}/{patch_dir_name}"):
            os.mkdir(f"{parent_dir_name}/{patch_dir_name}")
            files = [
                "patch1-0-1-2-3-#file1.png#.png",
                "patch2-4-5-6-7-#file1.png#.png",
                "patch1-0-1-2-3-#file2.png#.png",
            ]
            for file in files:
                with open(f"{parent_dir_name}/{patch_dir_name}/{file}", "w") as f:
                    pass


def test_load_patches_no_parents(create_test_dir):
    my_files = load_patches("test_parent_dir/patch_dir")
    assert len(my_files) == 3
    my_files = load_patches("test_parent_dir/patch_dir/")
    assert len(my_files) == 3
    my_files = load_patches("test_parent_dir/patch_dir/*")
    assert len(my_files) == 3
    my_files = load_patches("test_parent_dir/patch_dir/*png")
    assert len(my_files) == 3


# cases of mixed parents dir - i.e. tests using dir and file_ext or file paths


def test_load_patches_w_parent_dir_w_file_ext(create_test_dir):
    my_files = load_patches(
        "test_parent_dir/patch_dir",
        parent_paths="test_parent_dir",
        parent_file_ext="png",
    )
    assert len(my_files) == 6
    my_files = load_patches(
        "test_parent_dir/patch_dir",
        parent_paths="test_parent_dir",
        parent_file_ext="png",
    )
    assert len(my_files) == 6
    my_files = load_patches(
        "test_parent_dir/patch_dir",
        parent_paths="test_parent_dir/*",
        parent_file_ext="png",
    )
    assert len(my_files) == 6


def test_load_patches_w_parent_file_paths(create_test_dir):
    my_files = load_patches(
        "test_parent_dir/patch_dir", parent_paths="test_parent_dir/*png"
    )
    assert len(my_files) == 6


# unmixed 'parents' dir - i.e. using dir and no file_ext


def test_loadParents_no_file_ext(create_test_dir):
    my_files = loader()
    my_files.loadParents("test_parent_dir/patch_dir")
    assert len(my_files) == 3
    my_files.loadParents("test_parent_dir/patch_dir/")
    assert len(my_files) == 3
    my_files.loadParents("test_parent_dir/patch_dir/*")
    assert len(my_files) == 3
