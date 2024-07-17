mapreader.download.sheet_downloader
===================================

.. py:module:: mapreader.download.sheet_downloader


Classes
-------

.. autoapisummary::

   mapreader.download.sheet_downloader.SheetDownloader


Module Contents
---------------

.. py:class:: SheetDownloader(metadata_path, download_url)

   A class to download map sheets using metadata.


   .. py:method:: get_polygons()

      For each map in metadata, creates a polygon from map geometry and saves to ``features`` dictionary.



   .. py:method:: get_grid_bb(zoom_level = 14)

      For each map in metadata, creates a grid bounding box from map polygons and saves to ``features`` dictionary.

      :param zoom_level: The zoom level to use when creating the grid bounding box.
                         Later used when downloading maps, by default 14.
      :type zoom_level: int, optional



   .. py:method:: extract_wfs_id_nos()

      For each map in metadata, extracts WFS ID numbers from WFS information and saves to ``features`` dictionary.



   .. py:method:: extract_published_dates(date_col = None)

      For each map in metadata, extracts publication date and saves to ``features`` dictionary.

      :param date_col: A key or list of keys which map to the metadata field containing the publication date.
                       Multilayer keys should be passed as a list. e.g.:

                       - "key1" will extract ``self.features[i]["key1"]``
                       - ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``

                       If  None, ["properties"]["WFS_TITLE"] will be used as keys. Date will then be extracted by regex searching for "Published: XXX".
                       By default None.
      :type date_col: str or list, optional



   .. py:method:: get_merged_polygon()

      Creates a multipolygon representing all maps in metadata.



   .. py:method:: get_minmax_latlon()

      Prints minimum and maximum latitudes and longitudes of all maps in metadata.



   .. py:method:: query_map_sheets_by_wfs_ids(wfs_ids, append = False, print = False)

      Find map sheets by WFS ID numbers.

      :param wfs_ids: The WFS ID numbers of the maps to download.
      :type wfs_ids: Union[list, int]
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional



   .. py:method:: query_map_sheets_by_polygon(polygon, mode = 'within', append = False, print = False)

      Find map sheets which are found within or intersecting with a defined polygon.

      :param polygon: shapely Polygon
      :type polygon: Polygon
      :param mode: The mode to use when finding maps.
                   Options of ``"within"``, which returns all map sheets which are completely within the defined polygon,
                   and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
                   By default "within".
      :type mode: str, optional
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional

      .. rubric:: Notes

      Use ``create_polygon_from_latlons()`` to create polygon.



   .. py:method:: query_map_sheets_by_coordinates(coords, append = False, print = False)

      Find maps sheets which contain a defined set of coordinates.
      Coordinates are (x,y).

      :param coords: Coordinates in ``(x,y)`` format.
      :type coords: tuple
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional



   .. py:method:: query_map_sheets_by_line(line, append = False, print = False)

      Find maps sheets which intersect with a line.

      :param line: shapely LineString
      :type line: LineString
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional

      .. rubric:: Notes

      Use ``create_line_from_latlons()`` to create line.



   .. py:method:: query_map_sheets_by_string(string, keys = None, append = False, print = False)

      Find map sheets by searching for a string in a chosen metadata field.

      :param string: The string to search for.
                     Can be raw string and use regular expressions.
      :type string: str
      :param keys: A key or list of keys used to get the metadata field to search in.

                   Key(s) will be passed to each features dictionary.
                   Multilayer keys should be passed as a list. e.g. ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``.

                   If ``None``, will search in all metadata fields. By default ``None``.
      :type keys: str or list, optional
      :param append: Whether to append to current query results list or, if False, start a new list.
                     By default False
      :type append: bool, optional
      :param print: Whether to print query results or not.
                    By default False
      :type print: bool, optional

      .. rubric:: Notes

      ``string`` is case insensitive.



   .. py:method:: print_found_queries()

      Prints query results.



   .. py:method:: download_all_map_sheets(path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads all map sheets in metadata.

      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional



   .. py:method:: download_map_sheets_by_wfs_ids(wfs_ids, path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads map sheets by WFS ID numbers.

      :param wfs_ids: The WFS ID numbers of the maps to download.
      :type wfs_ids: Union[list, int]
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional



   .. py:method:: download_map_sheets_by_polygon(polygon, path_save = 'maps', metadata_fname = 'metadata.csv', mode = 'within', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads any map sheets which are found within or intersecting with a defined polygon.

      :param polygon: shapely Polygon
      :type polygon: Polygon
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param mode: The mode to use when finding maps.
                   Options of ``"within"``, which returns all map sheets which are completely within the defined polygon,
                   and ``"intersects""``, which returns all map sheets which intersect/overlap with the defined polygon.
                   By default "within".
      :type mode: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional

      .. rubric:: Notes

      Use ``create_polygon_from_latlons()`` to create polygon.



   .. py:method:: download_map_sheets_by_coordinates(coords, path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads any maps sheets which contain a defined set of coordinates.
      Coordinates are (x,y).

      :param coords: Coordinates in ``(x,y)`` format.
      :type coords: tuple
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional



   .. py:method:: download_map_sheets_by_line(line, path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads any maps sheets which intersect with a line.

      :param line: shapely LineString
      :type line: LineString
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional

      .. rubric:: Notes

      Use ``create_line_from_latlons()`` to create line.



   .. py:method:: download_map_sheets_by_string(string, keys = None, path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Download map sheets by searching for a string in a chosen metadata field.

      :param string: The string to search for.
                     Can be raw string and use regular expressions.
      :type string: str
      :param keys: A key or list of keys used to get the metadata field to search in.

                   Key(s) will be passed to each features dictionary.
                   Multilayer keys should be passed as a list. e.g. ["key1","key2"] will search for ``self.features[i]["key1"]["key2"]``.

                   If ``None``, will search in all metadata fields. By default ``None``.
      :type keys: str or list, optional
      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional

      .. rubric:: Notes

      ``string`` is case insensitive.



   .. py:method:: download_map_sheets_by_queries(path_save = 'maps', metadata_fname = 'metadata.csv', overwrite = False, download_in_parallel = True, **kwargs)

      Downloads map sheets saved as query results.

      :param path_save: Path to save map sheets, by default "maps"
      :type path_save: str, optional
      :param metadata_fname: Name to use for metadata file, by default "metadata.csv"
      :type metadata_fname: str, optional
      :param overwrite: Whether to overwrite existing maps, by default ``False``.
      :type overwrite: bool, optional
      :param download_in_parallel: Whether to download tiles in parallel, by default ``True``.
      :type download_in_parallel: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to the ``_download_map_sheets()`` method.
      :type \*\*kwargs: dict, optional



   .. py:method:: hist_published_dates(**kwargs)

      Plots a histogram of the publication dates of maps in metadata.

      :param \*\*kwargs: A dictionary containing keyword arguments to pass to plotting function.
                         See matplotlib.pyplot.hist() for acceptable values.

                         e.g. ``**dict(fc='c', ec='k')``
      :type \*\*kwargs: dict, optional

      .. rubric:: Notes

      bins and range already set when plotting so are invalid kwargs.



   .. py:method:: plot_features_on_map(features, map_extent = None, add_id = True)

      Plots boundaries of map sheets on a map using ``cartopy`` library, (if available).

      :param map_extent: The extent of the underlying map to be plotted.

                         If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
                         If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
                         If None, the map extent will be set automatically.
                         By default None.
      :type map_extent: Union[str, list, tuple, None], optional
      :param add_id: Whether to add an ID (WFS ID number) to each map sheet, by default True.
      :type add_id: bool, optional



   .. py:method:: plot_all_metadata_on_map(map_extent = None, add_id = True)

      Plots boundaries of all map sheets in metadata on a map using ``cartopy`` library (if available).

      :param map_extent: The extent of the underlying map to be plotted.

                         If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
                         If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
                         If None, the map extent will be set automatically.
                         By default None.
      :type map_extent: Union[str, list, tuple, None], optional
      :param add_id: Whether to add an ID (WFS ID number) to each map sheet, by default True.
      :type add_id: bool, optional



   .. py:method:: plot_queries_on_map(map_extent = None, add_id = True)

      Plots boundaries of query results on a map using ``cartopy`` library (if available).

      :param map_extent: The extent of the underlying map to be plotted.

                         If a tuple or list, must be of the format ``[lon_min, lon_max, lat_min, lat_max]``.
                         If a string, only ``"uk"``, ``"UK"`` or ``"United Kingdom"`` are accepted and will limit the map extent to the UK's boundaries.
                         If None, the map extent will be set automatically.
                         By default None.
      :type map_extent: Union[str, list, tuple, None], optional
      :param add_id: Whether to add an ID (WFS ID number) to each map sheet, by default True.
      :type add_id: bool, optional
