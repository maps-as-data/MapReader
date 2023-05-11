from mapreader import SheetDownloader
import pytest
import pathlib
from pathlib import Path
from shapely.geometry import Polygon, LineString
import os
import shutil

@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent / "sample_files"

@pytest.fixture
def sheet_downloader(sample_dir):
    test_json = f"{sample_dir}/test_json.json" # contains 4 features, 2x one-inch metadata in idx 0-1 and 2x six-inch metadata in idx 2-3
    download_url = "https://geo.nls.uk/maps/os/1inch_2nd_ed/{z}/{x}/{y}.png"
    return SheetDownloader(test_json, download_url)

def test_init(sheet_downloader):
    assert sheet_downloader.__len__() == 4

def test_extract_published_dates(sheet_downloader):
    sd = sheet_downloader
    sd.extract_published_dates()
    assert sd.published_dates == True
    assert sd.features[0] ["properties"]["published_date"] == 1896 #a standard one
    assert sd.features[3]["properties"]["published_date"] == 1896 #metadata has "1894 to 1896" - method take end date only (thoughts?)

def test_minmax_latlon(sheet_downloader, capfd):
    sd = sheet_downloader
    sd.get_minmax_latlon()
    out, _ = capfd.readouterr()
    assert out == "[INFO] Min lat: 51.49344796, max lat: 54.75000003 \n[INFO] Min lon: -5.40999994, max lon: -0.16093917\n"

# queries 

def test_query_by_wfs_ids(sheet_downloader):
    sd = sheet_downloader
    sd.query_map_sheets_by_wfs_ids(1) #test single wfs_id
    assert sd.wfs_id_nos == True
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[0]
    sd.query_map_sheets_by_wfs_ids([1,2]) #test list of wfs_ids
    assert len(sd.found_queries) == 2
    assert sd.found_queries == sd.features[:2]
    sd.query_map_sheets_by_wfs_ids(132, append=True) #test append
    assert len(sd.found_queries) == 3
    assert sd.found_queries[2] == sd.features[3]

def test_query_by_polygon(sheet_downloader):
    sd = sheet_downloader
    polygon = Polygon([[-4.79999994, 54.48000003], [-5.39999994, 54.48000003], [-5.40999994, 54.74000003], [-4.80999994, 54.75000003], [-4.79999994, 54.48000003]]) #should match to features[0]
    sd.query_map_sheets_by_polygon(polygon) #test mode = 'within'
    assert sd.polygons == True
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[0]
    sd.query_map_sheets_by_polygon(polygon, mode = 'intersects') #test mode = 'intersects'
    assert len(sd.found_queries) == 2
    assert sd.found_queries == sd.features[:2]
    another_polygon = Polygon([[-0.23045502, 51.49344796], [-0.23053988, 51.52237709], [-0.16097999, 51.52243594], [-0.16093917, 51.49350674], [-0.23045502, 51.49344796]]) #should match to features[3]
    sd.query_map_sheets_by_polygon(another_polygon, append=True) # test append
    assert len(sd.found_queries) == 3
    assert sd.found_queries[2] == sd.features[3]

def test_query_by_coords(sheet_downloader):
    sd = sheet_downloader
    sd.query_map_sheets_by_coordinates((-4.8, 54.5))
    assert sd.polygons == True
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[1]
    sd.query_map_sheets_by_coordinates((-0.23, 51.5), append = True) # test append
    assert len(sd.found_queries) == 2
    assert sd.found_queries[1] == sd.features[3]

def test_query_by_line(sheet_downloader):
    sd = sheet_downloader
    line = LineString([(-5.4, 54.5), (-4.8, 54.5)])
    sd.query_map_sheets_by_line(line)
    assert sd.polygons == True
    assert len(sd.found_queries) == 2
    assert sd.found_queries == sd.features[:2]
    another_line = LineString([(-0.2, 51.5),(-0.21,51.6)])
    sd.query_map_sheets_by_line(another_line, append = True) # test append
    assert len(sd.found_queries) == 3
    assert sd.found_queries[2] == sd.features[3]

def test_query_by_string(sheet_downloader):
    sd = sheet_downloader
    sd.query_map_sheets_by_string("Westminster",["properties","PARISH"])
    assert len(sd.found_queries) == 1
    assert sd.found_queries[0] == sd.features[3]
    sd.query_map_sheets_by_string("one_inch", "id", append = True) #test append + w/ keys as string
    assert len(sd.found_queries) == 3
    assert sd.found_queries[1:3] == sd.features[:2]
    sd.query_map_sheets_by_string("095", append = True) #test w/ no keys
    assert len(sd.found_queries) == 4
    assert sd.found_queries[3] == sd.features[2]

# download

def test_download_all(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    assert sd.grid_bbs == True
    maps_path= tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"
    sd.download_all_map_sheets(maps_path, metadata_fname)
    print(f"{maps_path}map_102352861.png")
    assert os.path.exists(f"{maps_path}/map_102352861.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 5      
    assert csv[0] == '\tname\turl\tcoordinates\tcrs\tpublished_date\tgrid_bb\n'
    assert csv[3].startswith('2\tmap_102352861.png')

def test_download_by_wfs_ids(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"
    sd.download_map_sheets_by_wfs_ids(1, maps_path, metadata_fname) #test single wfs_id
    assert os.path.exists(f"{maps_path}/map_74487492.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 2   
    assert csv[0] == '\tname\turl\tcoordinates\tcrs\tpublished_date\tgrid_bb\n'
    assert csv[1].startswith('0\tmap_74487492.png')
    sd.download_map_sheets_by_wfs_ids([1,2], maps_path, metadata_fname) #test list of wfs_ids
    assert os.path.exists(f"{maps_path}/map_74488550.png")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 3 #should have only downloaded/added one extra map   

def test_download_by_polygon(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    polygon = Polygon([[-4.79999994, 54.48000003], [-5.39999994, 54.48000003], [-5.40999994, 54.74000003], [-4.80999994, 54.75000003], [-4.79999994, 54.48000003]]) #should match to features[0]
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"    
    sd.download_map_sheets_by_polygon(polygon, maps_path, metadata_fname) #test mode = 'within'
    assert os.path.exists(f"{maps_path}/map_74487492.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 2   
    assert csv[0] == '\tname\turl\tcoordinates\tcrs\tpublished_date\tgrid_bb\n'
    assert csv[1].startswith('0\tmap_74487492.png')    
    sd.download_map_sheets_by_polygon(polygon, maps_path, metadata_fname, mode = 'intersects') #test mode = 'intersects', now 2 maps
    assert os.path.exists(f"{maps_path}/map_74488550.png")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 3 #should have only downloaded/added one extra map   

def test_download_by_coords(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"    
    sd.download_map_sheets_by_coordinates((-4.8, 54.5), maps_path, metadata_fname)
    assert os.path.exists(f"{maps_path}/map_74488550.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 2   
    assert csv[0] == '\tname\turl\tcoordinates\tcrs\tpublished_date\tgrid_bb\n'
    assert csv[1].startswith('0\tmap_74488550.png')

def test_download_by_line(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"
    line = LineString([(-5.4, 54.5), (-4.8, 54.5)])
    sd.download_map_sheets_by_line(line, maps_path, metadata_fname)
    assert os.path.exists(f"{maps_path}/map_74488550.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 3   
    assert csv[0] == '\tname\turl\tcoordinates\tcrs\tpublished_date\tgrid_bb\n'
    assert csv[1].startswith('0\tmap_74487492.png')

def test_download_by_string(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"
    sd.download_map_sheets_by_string("Westminster",["properties","PARISH"], maps_path, metadata_fname) #test w/ keys list
    sd.download_map_sheets_by_string("one_inch", "id", maps_path, metadata_fname) #test w/ keys string
    sd.download_map_sheets_by_string("095", path_save=maps_path, metadata_fname=metadata_fname) #test w/ no keys
    assert os.path.exists(f"{maps_path}/map_91617032.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 5
    assert csv[0] == '\tname\turl\tcoordinates\tcrs\tpublished_date\tgrid_bb\n'
    assert csv[1].startswith('0\tmap_91617032.png')

def test_download_by_queries(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"    
    sd.query_map_sheets_by_wfs_ids(131)
    sd.query_map_sheets_by_coordinates((-4.8, 54.5), append=True)
    assert len(sd.found_queries) == 2
    sd.download_map_sheets_by_queries(maps_path, metadata_fname)
    assert os.path.exists(f"{maps_path}/map_102352861.png")
    assert os.path.exists(f"{maps_path}/{metadata_fname}")
    with open(f"{maps_path}/{metadata_fname}") as f:
        csv = f.readlines()
    assert len(csv) == 3   
    assert csv[0] == '\tname\turl\tcoordinates\tcrs\tpublished_date\tgrid_bb\n'
    assert csv[1].startswith('0\tmap_102352861.png\t')

# errors

def test_query_by_wfs_ids_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="as int or list of ints"):
        sd.query_map_sheets_by_wfs_ids("str")
    with pytest.raises(ValueError, match="as int or list of ints"):
        sd.query_map_sheets_by_wfs_ids(21.4)

def test_query_by_polygon_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="pass polygon as shapely.geometry.Polygon"):
        sd.query_map_sheets_by_polygon([1,2])
    polygon = Polygon([[-4.79999994, 54.48000003], [-5.39999994, 54.48000003], [-5.40999994, 54.74000003], [-4.80999994, 54.75000003], [-4.79999994, 54.48000003]]) #should match to features[0]
    with pytest.raises(NotImplementedError, match='``mode="within"`` or ``mode="intersects"``'):
        sd.query_map_sheets_by_polygon(polygon, mode ="fake mode")

def test_query_by_coords_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="pass coords as a tuple"):
        sd.query_map_sheets_by_coordinates("str")

def test_query_by_line_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="pass line as shapely.geometry.LineString"):
        sd.query_map_sheets_by_line("str")

def test_query_by_string_errors(sheet_downloader):
    sd = sheet_downloader
    with pytest.raises(ValueError, match="pass ``string`` as a string"):
        sd.query_map_sheets_by_string(10, "id")
    with pytest.raises(ValueError, match="as string or list of strings"):
        sd.query_map_sheets_by_string("Westminster", 10)

#need to add error for if you pass keys as list of non strings (eg.[10])

def test_download_by_wfs_ids_errors(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"
    with pytest.raises(ValueError, match = "No map sheets"):
        sd.download_map_sheets_by_wfs_ids(12, maps_path, metadata_fname)

def test_download_by_polygon_errors(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    polygon = Polygon([[0,1], [1,2], [2,3], [3,4], [0,1]])
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"    
    with pytest.raises(ValueError, match = "out of map metadata bounds"):
        sd.download_map_sheets_by_polygon(polygon, maps_path, metadata_fname)

def test_download_by_coords_errors(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"    
    with pytest.raises(ValueError, match = "out of map metadata bounds"):
        sd.download_map_sheets_by_coordinates((0,1), maps_path, metadata_fname)

def test_download_by_line(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"
    line = LineString([(0,1), (2,3)])
    with pytest.raises(ValueError, match="out of map metadata bounds"):
        sd.download_map_sheets_by_line(line, maps_path, metadata_fname)

def test_download_by_queries_errors(sheet_downloader, tmp_path):
    sd = sheet_downloader
    sd.get_grid_bb(10)
    maps_path=tmp_path / "test_maps/"
    metadata_fname="test_metadata.csv"    
    with pytest.raises(ValueError, match="No query results"):
        sd.download_map_sheets_by_queries(maps_path, metadata_fname)

    


    

