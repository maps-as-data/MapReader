from mapreader import SheetDownloader
import pytest
from pathlib import Path
from shapely.geometry import Polygon
import os
import shutil

@pytest.fixture
def test_dir():
    return Path(__file__).resolve().parent

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent / "sample_files"

@pytest.fixture
def sheet_downloader(sample_dir):
    test_json = f"{sample_dir}/test_json.json" # contains one-inch metadata in idx 0-9 and six-inch metadata in idx 10-20
    download_url = "https://geo.nls.uk/maps/os/1inch_2nd_ed/{z}/{x}/{y}.png"
    return SheetDownloader(test_json, download_url)

def test_init(sheet_downloader):
    metadata = sheet_downloader
    assert metadata.__len__() == 4

def test_extract_published_dates(sheet_downloader):
    sd = sheet_downloader
    sd.extract_published_dates()
    assert sd.published_dates == True
    assert sd.features[0] ["properties"]["published_date"] == 1896 #a standard one
    assert sd.features[3]["properties"]["published_date"] == 1896 #metadata has "1894 to 1896" - method saves end date

def test_query_by_wfs_ids(sheet_downloader):
    sd = sheet_downloader
    sd.query_map_sheets_by_wfs_ids(1) #test single wfs_id
    assert sd.wfs_id_nos == True
    assert sd.found_queries[0] == sd.features[0]
    sd.query_map_sheets_by_wfs_ids([1,2]) #test list of wfs_ids
    assert len(sd.found_queries) == 2
    assert sd.found_queries[0] == sd.features[0]
    sd.query_map_sheets_by_wfs_ids(131, append=True) #test append
    assert len(sd.found_queries) == 3

def test_query_by_polygon(sheet_downloader):
    sd = sheet_downloader
    polygon = Polygon([[-4.79999994, 54.48000003], [-5.39999994, 54.48000003], [-5.40999994, 54.74000003], [-4.80999994, 54.75000003], [-4.79999994, 54.48000003]]) #should match to features[0]
    sd.query_map_sheets_by_polygon(polygon) #test mode = 'within'
    assert sd.polygons == True
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[0]
    sd.query_map_sheets_by_polygon(polygon, mode = 'intersects') #test mode = 'intersects'
    assert len(sd.found_queries) == 2
    another_polygon = Polygon([[-0.23045502, 51.49344796], [-0.23053988, 51.52237709], [-0.16097999, 51.52243594], [-0.16093917, 51.49350674], [-0.23045502, 51.49344796]]) #should match to features[3]
    sd.query_map_sheets_by_polygon(another_polygon, append=True) # test append
    assert len(sd.found_queries) == 3

def test_query_by_coords(sheet_downloader):
    sd = sheet_downloader
    sd.query_map_sheets_by_coordinates((-4.8, 54.5))
    assert sd.polygons == True
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[1]
    sd.query_map_sheets_by_coordinates((-0.23, 51.5), append = True) # test append
    assert len(sd.found_queries) == 2

def test_download_all(sheet_downloader):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    assert sd.grid_bbs == True
    maps_path="./test_maps/"
    metadata_fname="test_metadata.csv"
    sd.download_all_map_sheets(maps_path, metadata_fname)
    assert os.path.exists(f"{maps_path}map_102352861.png")
    assert os.path.exists(f"{maps_path}{metadata_fname}")
    with open(f"{maps_path}{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 5      
    assert csv[0] == '|name|url|coordinates|published_date|grid_bb\n'
    assert csv[3] == '2|map_102352861.png|https://maps.nls.uk/view/102352861|[[-2.6262171, 54.14935172], [-2.62837527, 54.20716287], [-2.48042189, 54.2089733], [-2.4784698, 54.15115834], [-2.6262171, 54.14935172]]|1896|[(10, 504, 327)x(10, 504, 328)]\n'
    shutil.rmtree(maps_path)

def test_download_by_wfs_ids(sheet_downloader):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path="./test_maps/"
    metadata_fname="test_metadata.csv"
    sd.download_map_sheets_by_wfs_ids(1, maps_path, metadata_fname) #test single wfs_id
    assert sd.wfs_id_nos == True
    assert os.path.exists(f"{maps_path}map_74487492.png")
    assert os.path.exists(f"{maps_path}{metadata_fname}")
    with open(f"{maps_path}{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 2   
    assert csv[0] == '|name|url|coordinates|published_date|grid_bb\n'
    assert csv[1] == '0|map_74487492.png|https://maps.nls.uk/view/74487492|[[-4.79999994, 54.48000003], [-5.39999994, 54.48000003], [-5.40999994, 54.74000003], [-4.80999994, 54.75000003], [-4.79999994, 54.48000003]]|1896|[(10, 496, 325)x(10, 498, 326)]\n'
    sd.download_map_sheets_by_wfs_ids([1,2], maps_path, metadata_fname) #test list of wfs_ids
    assert os.path.exists(f"{maps_path}map_74488550.png")
    with open(f"{maps_path}{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 3 #should have only downloaded/added one extra map   
    shutil.rmtree(maps_path)

def test_download_by_polygon(sheet_downloader):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    polygon = Polygon([[-4.79999994, 54.48000003], [-5.39999994, 54.48000003], [-5.40999994, 54.74000003], [-4.80999994, 54.75000003], [-4.79999994, 54.48000003]]) #should match to features[0]
    maps_path=f"./test_maps/"
    metadata_fname="test_metadata.csv"    
    sd.download_map_sheets_by_polygon(polygon, maps_path, metadata_fname) #test mode = 'within'
    assert sd.polygons == True
    assert os.path.exists(f"{maps_path}map_74487492.png")
    assert os.path.exists(f"{maps_path}{metadata_fname}")
    with open(f"{maps_path}{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 2   
    assert csv[0] == '|name|url|coordinates|published_date|grid_bb\n'
    assert csv[1] == '0|map_74487492.png|https://maps.nls.uk/view/74487492|[[-4.79999994, 54.48000003], [-5.39999994, 54.48000003], [-5.40999994, 54.74000003], [-4.80999994, 54.75000003], [-4.79999994, 54.48000003]]|1896|[(10, 496, 325)x(10, 498, 326)]\n'    
    sd.download_map_sheets_by_polygon(polygon, maps_path, metadata_fname, mode = 'intersects') #test mode = 'intersects', now 2 maps
    assert os.path.exists(f"{maps_path}map_74488550.png")
    with open(f"{maps_path}{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 3 #should have only downloaded/added one extra map   
    shutil.rmtree(maps_path)

def test_download_by_coords(sheet_downloader):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path="./test_maps/"
    metadata_fname="test_metadata.csv"    
    sd.download_map_sheets_by_coordinates((-4.8, 54.5), maps_path, metadata_fname)
    assert sd.polygons == True
    assert os.path.exists(f"{maps_path}map_74488550.png")
    assert os.path.exists(f"{maps_path}{metadata_fname}")
    with open(f"{maps_path}{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 2   
    assert csv[0] == '|name|url|coordinates|published_date|grid_bb\n'
    assert csv[1] == '0|map_74488550.png|https://maps.nls.uk/view/74488550|[[-4.19999994, 54.49000003], [-4.79999994, 54.48000003], [-4.80999994, 54.75000003], [-4.20999994, 54.75000003], [-4.19999994, 54.49000003]]|1896|[(10, 498, 325)x(10, 500, 326)]\n'
    shutil.rmtree(maps_path)

def test_download_by_queries(sheet_downloader):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path="./test_maps/"
    metadata_fname="test_metadata.csv"    
    sd.query_map_sheets_by_wfs_ids(131)
    sd.query_map_sheets_by_coordinates((-4.8, 54.5), append=True)
    assert len(sd.found_queries) == 2
    sd.download_map_sheets_by_queries(maps_path, metadata_fname)
    assert os.path.exists(f"{maps_path}map_102352861.png")
    assert os.path.exists(f"{maps_path}{metadata_fname}")
    with open(f"{maps_path}{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 3   
    assert csv[0] == '|name|url|coordinates|published_date|grid_bb\n'
    assert csv[1] == '0|map_102352861.png|https://maps.nls.uk/view/102352861|[[-2.6262171, 54.14935172], [-2.62837527, 54.20716287], [-2.48042189, 54.2089733], [-2.4784698, 54.15115834], [-2.6262171, 54.14935172]]|1896|[(10, 504, 327)x(10, 504, 328)]\n'
    shutil.rmtree(maps_path)





    



    

    

