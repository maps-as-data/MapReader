from mapreader import loader
import pytest
import os
import pandas as pd

@pytest.fixture
def dir_name():
    return "metadata_test_dir"

#if there is extra info in the metadata
def test_extra_metadata(dir_name):
    if os.path.exists(dir_name):
        os.rmdir(dir_name)    
    os.mkdir(dir_name)
    files = ["file1.png", "file2.png"]
    for file in files:
        with open(f"{dir_name}/{file}", "w") as f:
            pass
    metadata_df = pd.DataFrame({"name":["file1.png", "file2.png", "file3.png"], "coord":[(1.1,1.5),(2.1,1.0),(3.1,4.5)], "other":["one","two","three"]})
    metadata_df.to_csv(f"{dir_name}/metadata_df.csv", sep="|")
    
    my_files=loader(f"{dir_name}/*png")
    assert len(my_files)==2
    my_files.add_metadata(f"{dir_name}/metadata_df.csv")
    assert my_files.images["parent"]["file2.png"]["other"]=="two"
    assert isinstance(my_files.images["parent"]["file2.png"]["coord"], tuple)

#if metadata info matches 
def test_matching_metdata(dir_name):
    with open(f"{dir_name}/file3.png", "w") as f:
        pass
    
    my_files=loader(f"{dir_name}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{dir_name}/metadata_df.csv")
    assert my_files.images["parent"]["file3.png"]["other"]=="three"
    assert isinstance(my_files.images["parent"]["file3.png"]["coord"], tuple)

#if you pass a metadata df
def test_metadata_df(dir_name):
    my_files=loader(f"{dir_name}/*png")
    assert len(my_files)==3
    metadata_df=pd.read_csv(f"{dir_name}/metadata_df.csv", delimiter="|")
    my_files.add_metadata(metadata_df)
    assert my_files.images["parent"]["file3.png"]["other"]=="three"

#if you forget your file_ext
def test_metadata_missing_file_ext(dir_name):
    my_files=loader(f"{dir_name}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{dir_name}/metadata_df")
    assert my_files.images["parent"]["file3.png"]["other"]=="three"

#if you pass index col - this should pick up if index.name is 'name' or 'image_id'
def test_index_col(dir_name):
    my_files=loader(f"{dir_name}/*png")
    assert len(my_files)==3
    my_files.add_metadata(f"{dir_name}/metadata_df.csv",index_col="name")
    assert my_files.images["parent"]["file3.png"]["other"]=="three"

#if you pass columns
def test_usecols(dir_name):
    my_files=loader(f"{dir_name}/*png")
    assert len(my_files)==3
    file3_keys = list(my_files.images["parent"]["file3.png"].keys())
    my_files.add_metadata(f"{dir_name}/metadata_df.csv",columns=["name","coord"])
    file3_keys.append("name")
    file3_keys.append("coord")
    assert isinstance(my_files.images["parent"]["file3.png"]["coord"], tuple)
    assert list(my_files.images["parent"]["file3.png"].keys()) == file3_keys

#if there is missing info in metadata
def test_missing_metadata(dir_name):
    with open(f"{dir_name}/file4.png", "w") as f:
        pass
    
    my_files=loader("./metadata_test_dir/*png")
    assert len(my_files)==4
    file4_keys = list(my_files.images["parent"]["file4.png"].keys())
    my_files.add_metadata(f"{dir_name}/metadata_df.csv")
    assert my_files.images["parent"]["file3.png"]["other"]=="three"
    assert isinstance(my_files.images["parent"]["file3.png"]["coord"], tuple)
    assert list(my_files.images["parent"]["file4.png"].keys()) == file4_keys
    