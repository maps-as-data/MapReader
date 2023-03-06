:py:mod:`mapreader.download.tileserver_access`
==============================================

.. py:module:: mapreader.download.tileserver_access


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_access.TileServer



Functions
~~~~~~~~~

.. autoapisummary::

   mapreader.download.tileserver_access.download_tileserver_parallel



.. py:class:: TileServer(metadata_path, geometry='polygone', download_url='https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png')

   .. py:method:: create_info()

      Extract information from metadata and create metadata_info_list and metadata_coord_arr.

      This is a helper function for other methods in this class


   .. py:method:: modify_metadata(remove_image_ids=[], only_keep_image_ids=[])

      Modify metadata using metadata[...]["properties"]["IMAGE"]

      Parameters
      ----------
      remove_image_ids : list, optional
          Image IDs to be removed from metadata variable


   .. py:method:: query_point(latlon_list, append=False)

      Query maps from a list of lats/lons using metadata file

      Args:
          latlon_list (list): a list that contains lats/lons: [lat, lon] or [[lat1, lon1], [lat2, lon2], ...]
          append (bool, optional): If True, append the new query to the list of queries. Defaults to False.


   .. py:method:: print_found_queries()

      Print found queries


   .. py:method:: detect_rectangle_boundary(coords)

      Detect rectangular boundary given a set of coordinates


   .. py:method:: create_metadata_query()

      Create a metadata type variable out of all queries.
      This will be later used in download_tileserver method


   .. py:method:: minmax_latlon()

      Method to return min/max of lats/lons


   .. py:method:: download_tileserver(mode='queries', num_img2test=-1, zoom_level=14, adjust_mult=0.005, retries=10, scraper_max_connections=4, failed_urls_path='failed_urls.txt', tile_tmp_dir='tiles', output_maps_dirname='maps', output_metadata_filename='metadata.csv', pixel_closest=None, redownload=False, id1=0, id2=-1, error_path='errors.txt', max_num_errors=20)

      Download maps via tileserver

      Args:
          mode (str): specify the set of maps to be downloaded:
                      mode = query or queries: this will download the queried maps
                      mode = all: download all maps in the metadata file
          num_img2test (int, optional): Number of images to download for testing. Defaults to -1 (all maps).
          zoom_level (int, optional): Zoom level for maps to be downloaded. Defaults to 14.
          adjust_mult (float, optional): If some tiles cannot be downloaded, shrink the requested bounding box.
                                         by this factor. Defaults to 0.005.
          retries (int, optional): If a tile cannot be downloaded, retry these many times. Defaults to 1.
          failed_urls_path (str, optional): File that contains info about failed download attempts. Defaults to "failed_urls.txt".
          tile_tmp_dir (str, optional): Save tmp files in this directory. Defaults to "tiles".
          output_maps_dirname (str, optional): Path to save downloaded maps. Defaults to "maps".
          output_metadata_filename (str, optional): Path to save metada for downloaded maps. Defaults to "metadata.csv".
                                                    this file will be saved at output_maps_dirname/output_metadata_filename
          pixel_closest (int): adjust the number of pixels in both directions (width and height) after downloading a map
                               for example, if pixel_closest = 100, number of pixels in both directions will be multiples of 100
                               this helps to create only square tiles in processing step
          redownload (bool): if False, only maps that do not exist in the local directory will be retrieved
          id1, id2: consider metadata[id1:id2]


   .. py:method:: extract_region_dates_metadata(one_item)

      Extract name of the region and surveyed/revised/published dates

      Parameters
      ----------
      one_item : dict
          dictionary which contains at least properties/WFS_TITLE


   .. py:method:: find_and_clean_date(ois, ois_key='surveyed')
      :staticmethod:

      Given a string (ois) and a key (ois_key), extract date

      Parameters
      ----------
      ois : str
          string that contains date info
      ois_key : str, optional
          type of date, e.g., surveyed/revised/published


   .. py:method:: plot_metadata_on_map(list2remove=[], map_extent=None, add_text=False)

      Plot the map boundaries specified in metadata

      Args:
          list2remove (list, optional): List of IDs to be removed. Defaults to [].
          map_extent (list or None, optional): Extent of the main map [min_lon, max_lon, min_lat, max_lat]. Defaults to None.
          add_text (bool, optional): Add image IDs to the figure


   .. py:method:: hist_published_dates(min_date=None, max_date=None)

      Plot a histogram for published dates

      Parameters
      ----------
      min_date : int, None
          min date for histogram
      max_date : int, None
          max date for histogram


   .. py:method:: download_tileserver_rect(mode='queries', num_img2test=-1, zoom_level=14, adjust_mult=0.005, retries=1, failed_urls_path='failed_urls.txt', tile_tmp_dir='tiles', output_maps_dirname='maps', output_metadata_filename='metadata.csv', pixel_closest=None, redownload=False, id1=0, id2=-1, min_lat_len=0.05, min_lon_len=0.05)

      Download maps via tileserver

      Args:
          mode (str): specify the set of maps to be downloaded:
                      mode = query or queries: this will download the queried maps
                      mode = all: download all maps in the metadata file
          num_img2test (int, optional): Number of images to download for testing. Defaults to -1 (all maps).
          zoom_level (int, optional): Zoom level for maps to be downloaded. Defaults to 14.
          adjust_mult (float, optional): If some tiles cannot be downloaded, shrink the requested bounding box.
                                         by this factor. Defaults to 0.005.
          retries (int, optional): If a tile cannot be downloaded, retry these many times. Defaults to 1.
          failed_urls_path (str, optional): File that contains info about failed download attempts. Defaults to "failed_urls.txt".
          tile_tmp_dir (str, optional): Save tmp files in this directory. Defaults to "tiles".
          output_maps_dirname (str, optional): Path to save downloaded maps. Defaults to "maps".
          output_metadata_filename (str, optional): Path to save metada for downloaded maps. Defaults to "metadata.csv".
                                                    this file will be saved at output_maps_dirname/output_metadata_filename
          pixel_closest (int): adjust the number of pixels in both directions (width and height) after downloading a map
                               for example, if pixel_closest = 100, number of pixels in both directions will be multiples of 100
                               this helps to create only square tiles in processing step
          redownload (bool): if False, only maps that do not exist in the local directory will be retrieved
          id1, id2: consider metadata[id1:id2]



.. py:function:: download_tileserver_parallel(metadata, start, end, process_np=8, **kwds)

   Run download_tileserver in parallel using multiprocessing

   Args:
       metadata (dictionary):
           Dictionary that contains info about the maps to be downloaded.
           Each key in this dict should have
           ["features"]
           ["properties"]["IMAGEURL"]
           ["geometry"]["coordinates"]
           ["properties"]["IMAGE"]
       start: start index, i.e., metadata[start:end] will be used
       end: end index, i.e., metadata[start:end] will be used
       process_np (int, optional): Number of CPUs to be used. Defaults to 8.


