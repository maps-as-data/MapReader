from mapreader import loader
import pytest
import os
import pandas as pd
import pathlib
from PIL import Image
from random import randint

@pytest.fixture
def keys():
    return ["parent_id", "image_path", "shape", "name", "coord", "other"]

@pytest.fixture
def metadata_df():
    return pd.DataFrame({"name":["file1.png", "file2.png", "file3.png"], "coord":[(1.1,1.5),(2.1,1.0),(3.1,4.5)], "other":[1,2,3]})

@pytest.fixture
def matching_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png", "file3.png"]
    for file in files:
        rand_colour = (randint(0,255), randint(0,255), randint(0,255))
        Image.new("RGB",(9,9), color = rand_colour).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep="|")
    metadata_df.to_excel(f"{test_path}/metadata_df.xlsx")
    return test_path

@pytest.fixture
def extra_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png"]
    for file in files:
        rand_colour = (randint(0,255), randint(0,255), randint(0,255))
        Image.new("RGB",(9,9), color = rand_colour).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep="|")
    return test_path 

@pytest.fixture
def missing_metadata_dir(tmp_path, metadata_df):
    test_path = tmp_path / "test_dir"
    os.mkdir(test_path)
    files = ["file1.png", "file2.png", "file3.png", "file4.png"]
    for file in files:
        rand_colour = (randint(0,255), randint(0,255), randint(0,255))
        Image.new("RGB",(9,9), color = rand_colour).save(f"{test_path}/{file}")
    metadata_df.to_csv(f"{test_path}/metadata_df.csv", sep="|")
    return test_path

#if metadata info matches 

def test_matching_metdata_csv(matching_metadata_dir, keys):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{matching_metadata_dir}/metadata_df.csv")
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys

def test_matching_metdata_xlsx(matching_metadata_dir, keys):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{matching_metadata_dir}/metadata_df.xlsx")
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys

def test_matching_metadata_df(matching_metadata_dir, metadata_df, keys):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(metadata_df)
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys

#if you pass index col - this should pick up if index.name is 'name' or 'image_id'
def test_matching_metadata_csv_w_index_col(matching_metadata_dir):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{matching_metadata_dir}/metadata_df.csv", index_col="name")
    keys = ["parent_id", "image_path", "shape", "Unnamed: 0", "coord", "other", "name"]
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys

#if you pass columns
def test_matching_metadata_csv_w_usecols(matching_metadata_dir):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{matching_metadata_dir}/metadata_df.csv", columns=["name","coord"])
    keys = ["parent_id", "image_path", "shape", "name", "coord"]
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys
        assert isinstance(my_files.images["parent"][parent_id]["coord"], tuple)

#if there is extra info in the metadata

def test_extra_metadata_csv_ignore_mismatch(extra_metadata_dir,keys):  
    my_files=loader(f"{extra_metadata_dir}/*png")
    assert len(my_files)==2
    my_files.add_metadata(f"{extra_metadata_dir}/metadata_df.csv", ignore_mismatch=True)
    for parent_id in my_files.list_parents():
        assert list(my_files.images["parent"][parent_id].keys()) == keys
    
def test_extra_metadata_csv_errors(extra_metadata_dir):  
    my_files=loader(f"{extra_metadata_dir}/*png")
    assert len(my_files)==2
    with pytest.raises(ValueError, match="information about non-existant images"):
        my_files.add_metadata(f"{extra_metadata_dir}/metadata_df.csv")
    
#if there is missing info in metadata

def test_missing_metadata_csv_ignore_mismatch(missing_metadata_dir, keys):  
    my_files=loader(f"{missing_metadata_dir}/*png")
    assert len(my_files)==4
    my_files.add_metadata(f"{missing_metadata_dir}/metadata_df.csv", ignore_mismatch=True)
    for parent_id in ["file1.png", "file2.png", "file3.png"]:
        assert list(my_files.images["parent"][parent_id].keys()) == keys
    assert list(my_files.images["parent"]["file4.png"].keys()) == ["parent_id", "image_path", "shape"]
    
def test_missing_metadata_csv_errors(missing_metadata_dir):  
    my_files=loader(f"{missing_metadata_dir}/*png")
    assert len(my_files)==4
    with pytest.raises(ValueError, match="missing information"):
        my_files.add_metadata(f"{missing_metadata_dir}/metadata_df.csv")

# other errors

#if csv file doesn't exist
def test_metadata_not_found(matching_metadata_dir):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    with pytest.raises(ValueError):
        my_files.add_metadata(f"{matching_metadata_dir}/fakefile.csv")

def test_metadata_missing_name_or_image_id(matching_metadata_dir):
    my_files=loader(f"{matching_metadata_dir}/*png")
    assert len(my_files)==3
    incomplete_metadata_df = pd.DataFrame({"coord":[(1.1,1.5),(2.1,1.0),(3.1,4.5)], "other":[1,2,3]})
    incomplete_metadata_df.to_csv(f"{matching_metadata_dir}/incomplete_metadata_df.csv", sep="|")
    with pytest.raises(ValueError, match = "'name' or 'image_id' should be one of the columns"):
        my_files.add_metadata(incomplete_metadata_df)
    with pytest.raises(ValueError, match = "'name' or 'image_id' should be one of the columns"):
        my_files.add_metadata(f"{matching_metadata_dir}/incomplete_metadata_df.csv")

