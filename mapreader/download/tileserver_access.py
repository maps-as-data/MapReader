#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from glob import glob
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import shutil

from .tileserver_helpers import create_hf
from .tileserver_helpers import collect_coord_info
from .tileserver_helpers import check_par_jobs
from .tileserver_scraper import scraper
from .tileserver_stitcher import stitcher


class TileServer:
    def __init__(
        self,
        metadata_path,
        geometry="polygone",
        download_url="https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png",
    ):
        """Initialize TileServer class and read in metadata

        self.metadata is a dictionary that contains info about the maps to be downloaded.
            Each key in this dict should have
            ["properties"]["IMAGEURL"]
            ["geometry"]["coordinates"]
            ["properties"]["IMAGE"]

        Args:
            metadata_path: path to a metadata file downloaded from a tileserver.
                           This file contains information about one series of maps stored on the tileserver.
                           Some example metadata files can be found in `MapReader/mapreader/persistent_data`
            geometry (str, optional): Geometry that defines the boundaries. Defaults to "polygone".
            download_url (str, optional): Base URL to download the maps. Defaults to "https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png".
        """
        # initialize variables
        self.detected_rect_boundary = False
        self.found_queries = None
        self.geometry = geometry
        self.download_url = download_url

        if isinstance(metadata_path, str) and os.path.isfile(metadata_path):
            # Read a metada file
            json_fio = open(metadata_path, "r")
            json_f = json_fio.readlines()[0]
            # change metadata to a dictionary:
            metadata_all = eval(json_f)
            metadata = metadata_all["features"]

        elif isinstance(metadata_path, dict):
            metadata = metadata_all["features"]

        elif isinstance(metadata_path, list):
            metadata = metadata_path[:]

        else:
            raise ValueError(f"Could not find or load: {metadata_path}")

        # metadata contains the "features" of metadata_all
        self.metadata = metadata
        print(self.__str__())

    def create_info(self):
        """
        Extract information from metadata and create metadata_info_list and metadata_coord_arr.

        This is a helper function for other methods in this class
        """
        metadata_info_list = []
        metadata_coord_arr = []
        for one_item in self.metadata:
            min_lon, max_lon, min_lat, max_lat = self.detect_rectangle_boundary(
                one_item["geometry"]["coordinates"][0][0]
            )
            # XXX hardcoded: the file path is hardcoded to map_<IMAGE>.png
            metadata_info_list.append(
                [
                    one_item["properties"]["IMAGEURL"],
                    "map_" + one_item["properties"]["IMAGE"] + ".png",
                ]
            )
            # Collect boundaries for fast queries
            metadata_coord_arr.append([min_lon, max_lon, min_lat, max_lat])
        self.metadata_info_list = metadata_info_list
        self.metadata_coord_arr = np.array(metadata_coord_arr).astype(np.float)
        self.detected_rect_boundary = True

    def modify_metadata(self, remove_image_ids=[], only_keep_image_ids=[]):
        """Modify metadata using metadata[...]["properties"]["IMAGE"]

        Parameters
        ----------
        remove_image_ids : list, optional
            Image IDs to be removed from metadata variable
        """

        print(f"[INFO] #images (before modification): {len(self.metadata)}")

        one_edit = False
        if len(remove_image_ids) != 0:
            for i in range(len(self.metadata) - 1, -1, -1):
                if self.metadata[i]["properties"]["IMAGE"] in remove_image_ids:
                    print(f'Removing {self.metadata[i]["properties"]["IMAGE"]}')
                    del self.metadata[i]
                    one_edit = True

        if len(only_keep_image_ids) != 0:
            for i in range(len(self.metadata) - 1, -1, -1):
                if self.metadata[i]["properties"]["IMAGE"] not in only_keep_image_ids:
                    print(f'Removing {self.metadata[i]["properties"]["IMAGE"]}')
                    del self.metadata[i]
                    one_edit = True

        if one_edit:
            print("[INFO] run .create_info")
            self.create_info()
        print(f"[INFO] #images (after modification): {len(self.metadata)}")

    def query_point(self, latlon_list, append=False):
        """Query maps from a list of lats/lons using metadata file

        Args:
            latlon_list (list): a list that contains lats/lons: [lat, lon] or [[lat1, lon1], [lat2, lon2], ...]
            append (bool, optional): If True, append the new query to the list of queries. Defaults to False.
        """
        # Detect the boundaries of maps
        if not self.detected_rect_boundary:
            self.create_info()

        if not type(latlon_list[0]) == list:
            latlon_list = [latlon_list]

        if append and (self.found_queries != None):
            found_queries = copy.deepcopy(self.found_queries)
        else:
            found_queries = []

        for one_q in latlon_list:
            lat_q, lon_q = one_q
            indx_q = np.where(
                (self.metadata_coord_arr[:, 2] <= lat_q)
                & (lat_q <= self.metadata_coord_arr[:, 3])
                & (self.metadata_coord_arr[:, 0] <= lon_q)
                & (lon_q <= self.metadata_coord_arr[:, 1])
            )[0]
            if len(indx_q) == 0:
                print(f"Could not find any candidates for {one_q}")
                continue

            indx_q = indx_q[0]

            already_in_candidates = False
            # Check if the query is already in the list
            if len(found_queries) > 0:
                for one_in_candidates in found_queries:
                    if self.metadata_info_list[indx_q][0] == one_in_candidates[0]:
                        already_in_candidates = True
                        print(f"Already in the query list: {one_in_candidates[0]}")
                        break
            if not already_in_candidates:
                found_queries.append(copy.deepcopy(self.metadata_info_list[indx_q]))
                found_queries[-1].extend(
                    [copy.deepcopy(self.metadata_coord_arr[indx_q]), indx_q]
                )
        self.found_queries = found_queries

    def print_found_queries(self):
        """Print found queries"""
        print(12 * "-")
        print(f"Found items:")
        print(12 * "-")
        if not self.found_queries:
            print("[]")
        else:
            for one_item in self.found_queries:
                print(f"URL:     \t{one_item[0]}")
                print(f"filepath:\t{one_item[1]}")
                print(f"coords:  \t{one_item[2]}")
                print(f"index:   \t{one_item[3]}")
                print(20 * "=")

    def detect_rectangle_boundary(self, coords):
        """Detect rectangular boundary given a set of coordinates"""
        if self.geometry == "polygone":
            coord_arr = np.array(coords)
            # if len(coord_arr) != 5:
            #    raise ValueError(f"[ERROR] expected length of coordinate list is 5. coords: {coords}")
            min_lon = np.min(coord_arr[:, 0])
            max_lon = np.max(coord_arr[:, 0])
            min_lat = np.min(coord_arr[:, 1])
            max_lat = np.max(coord_arr[:, 1])

            ### # XXX this method results in smaller rectangles (compared to the original polygone) particularly
            ### #     if the map is strongly tilted
            ### min_lon = np.sort(coord_arr[:-1, 0])[1]
            ### max_lon = np.sort(coord_arr[:-1, 0])[2]
            ### min_lat = np.sort(coord_arr[:-1, 1])[1]
            ### max_lat = np.sort(coord_arr[:-1, 1])[2]
        return min_lon, max_lon, min_lat, max_lat

    def create_metadata_query(self):
        """
        Create a metadata type variable out of all queries.
        This will be later used in download_tileserver method
        """
        self.metadata_query = []
        for one_item in self.found_queries:
            self.metadata_query.append(copy.deepcopy(self.metadata[one_item[3]]))

    def minmax_latlon(self):
        """Method to return min/max of lats/lons"""
        if not self.detected_rect_boundary:
            self.create_info()
        print(
            f"Min/Max Lon: {np.min(self.metadata_coord_arr[:, 0])}, {np.max(self.metadata_coord_arr[:, 1])}"
        )
        print(
            f"Min/Max Lat: {np.min(self.metadata_coord_arr[:, 2])}, {np.max(self.metadata_coord_arr[:, 3])}"
        )

    def download_tileserver(
        self,
        mode="queries",
        num_img2test=-1,
        zoom_level=14,
        adjust_mult=0.005,
        retries=10,
        scraper_max_connections=4,
        failed_urls_path="failed_urls.txt",
        tile_tmp_dir="tiles",
        output_maps_dirname="maps",
        output_metadata_filename="metadata.csv",
        pixel_closest=None,
        redownload=False,
        id1=0,
        id2=-1,
        error_path="errors.txt",
        max_num_errors=20,
    ):
        """Download maps via tileserver

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
        """

        if not os.path.isdir(output_maps_dirname):
            os.makedirs(output_maps_dirname)

        if not os.path.isdir("geojson"):
            os.makedirs("geojson")

        if mode in ["query", "queries"]:
            self.create_metadata_query()
            metadata = self.metadata_query
        else:
            metadata = self.metadata

        # Header and Footer for GeoJson
        header, footer = create_hf(geom="polygone")

        if id2 < 0:
            metadata = metadata[id1:]
        elif id2 < id1:
            raise ValueError(f"id2 should be > id1.")
        else:
            metadata = metadata[id1:id2]

        # --- metadata file
        metadata_output_path = os.path.join(
            output_maps_dirname, output_metadata_filename
        )

        saved_metadata = None
        try_cond1 = try_cond2 = False
        if not os.path.isfile(metadata_output_path):
            with open(metadata_output_path, "w") as fio:
                fio.writelines("|name|url|coord|pub_date|region|polygone\n")
                counter = 0
        else:
            with open(metadata_output_path, "r") as fio:
                counter = len(fio.readlines()) - 1
            saved_metadata = pd.read_csv(metadata_output_path, delimiter="|")

        for one_item in metadata:
            try:
                # only for testing
                if counter == num_img2test:
                    break

                print("-------------------")
                output_stitcher = os.path.join(
                    output_maps_dirname,
                    "map_" + one_item["properties"]["IMAGE"] + ".png",
                )

                if saved_metadata is not None:
                    try_cond1 = (
                        str(one_item["geometry"]["coordinates"][0][0])
                        in saved_metadata["polygone"].to_list()
                    )
                    try_cond2 = (
                        f'map_{one_item["properties"]["IMAGE"]}.png'
                        in saved_metadata["name"].to_list()
                    )

                    if (not try_cond1) and (try_cond2):
                        append_id = int(one_item["id"].split(".")[1])
                        output_stitcher = os.path.join(
                            output_maps_dirname,
                            f'map_{one_item["properties"]["IMAGE"]}@{append_id}.png',
                        )

                if os.path.isfile(output_stitcher) and (not redownload):
                    print(f"File already exists: {output_stitcher}")
                    continue

                print("Retrieving: ", one_item["properties"]["IMAGEURL"])
                # print(one_item["geometry"]["coordinates"])

                # failed_urls_path is created by scraper.py
                # remove the file and check if it will be created by scraper
                if os.path.isfile(failed_urls_path):
                    os.remove(failed_urls_path)

                values = one_item["geometry"]["coordinates"][0][0]
                str2write = header + str(values) + footer

                if (not try_cond1) and (try_cond2):
                    poly_filename = f'{one_item["properties"]["IMAGE"]}@{append_id}_{counter}.geojson'
                else:
                    poly_filename = (
                        f'{one_item["properties"]["IMAGE"]}_{counter}.geojson'
                    )
                fio = open(os.path.join("geojson", poly_filename), "w")
                fio.writelines(str2write)
                fio.close()

                if os.path.isdir(tile_tmp_dir):
                    # remove previously retrieved tiles
                    shutil.rmtree(tile_tmp_dir)

                # Run the scraper
                scraper(
                    os.path.join("geojson", poly_filename),
                    zoom_level,
                    self.download_url,
                    tile_tmp_dir,
                    max_connections=scraper_max_connections,
                    retries=retries,
                )

                # if any failed URLs, log and continue
                if os.path.isfile(failed_urls_path):
                    with open(failed_urls_path, "r") as fp:
                        errors = fp.readlines()
                        if len(errors) > max_num_errors:
                            print(f"[WARNING] Could not retrieve {len(errors)} tiles.")
                            with open(error_path, "a+") as fio:
                                line2write = f"{counter}|"
                                line2write += f"{poly_filename}|"
                                line2write += f"{len(errors)}|"
                                line2write += f"{os.path.basename(output_stitcher)}|"
                                line2write += f"{one_item['properties']['IMAGEURL']}|"
                                line2write += f"{values}\n"
                                fio.writelines(line2write)
                            continue

                # Run stitcher
                stitcher(tile_tmp_dir, output_stitcher, pixel_closest)

                # Collect the new bounding boxes according to the retrieved tiles
                list_files = glob(os.path.join(tile_tmp_dir, "*png"))
                coord_info = collect_coord_info(list_files)

                name_region, _, _, published_date = self.extract_region_dates_metadata(
                    one_item
                )

                with open(metadata_output_path, "a+") as fio:
                    line2write = f"{counter}|"
                    line2write += f"{os.path.basename(output_stitcher)}|"
                    line2write += f"{one_item['properties']['IMAGEURL']}|"
                    line2write += f"{coord_info}|"
                    line2write += f"{published_date}|"
                    line2write += f"{name_region}|"
                    line2write += f"{values}\n"
                    fio.writelines(line2write)
            except Exception as err:
                print(f"[ERROR] {err}")
                with open(error_path, "a+") as fio:
                    line2write = f"{counter}|"
                    line2write += f"{poly_filename}|"
                    line2write += f"EXCEPTION|"
                    line2write += f"{os.path.basename(output_stitcher)}|"
                    line2write += f"{one_item['properties']['IMAGEURL']}|"
                    line2write += f"{values}\n"
                    fio.writelines(line2write)

            counter += 1

        # clean-up
        if os.path.isdir(tile_tmp_dir):
            shutil.rmtree(tile_tmp_dir)

    def extract_region_dates_metadata(self, one_item):
        """Extract name of the region and surveyed/revised/published dates

        Parameters
        ----------
        one_item : dict
            dictionary which contains at least properties/WFS_TITLE
        """
        name_region = "None"
        surveyed_date = -1
        revised_date = -1
        published_date = -1
        try:
            one_item_split = one_item["properties"]["WFS_TITLE"].split(",")
            name_region = one_item_split[0].strip()
            for ois in one_item_split[1:]:
                if "surveyed" in ois.lower():
                    surveyed_date = self.find_and_clean_date(ois, ois_key="surveyed")
                if "revised" in ois.lower():
                    revised_date = self.find_and_clean_date(ois, ois_key="revised")
                if "published" in ois.lower():
                    published_date = self.find_and_clean_date(ois, ois_key="published")
        except:
            pass

        return name_region, surveyed_date, revised_date, published_date

    @staticmethod
    def find_and_clean_date(ois, ois_key="surveyed"):
        """Given a string (ois) and a key (ois_key), extract date

        Parameters
        ----------
        ois : str
            string that contains date info
        ois_key : str, optional
            type of date, e.g., surveyed/revised/published
        """
        found_date = ois.lower().split(ois_key + ": ")[1].strip()
        for strip_token in ["to", "ca."]:
            if strip_token in found_date:
                found_date = found_date.split(strip_token)[1].strip()
        return found_date

    def plot_metadata_on_map(self, list2remove=[], map_extent=None, add_text=False):
        """Plot the map boundaries specified in metadata

        Args:
            list2remove (list, optional): List of IDs to be removed. Defaults to [].
            map_extent (list or None, optional): Extent of the main map [min_lon, max_lon, min_lat, max_lat]. Defaults to None.
            add_text (bool, optional): Add image IDs to the figure
        """

        cartopy_installed = False
        try:
            import cartopy.crs as ccrs

            cartopy_installed = True
        except ImportError:
            print(f"[WARNING] cartopy is not installed!")

        # setup the figure/map
        plt.figure(figsize=(15, 15))
        if cartopy_installed:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", color="black", linewidth=1)
            if map_extent == "uk":
                extent = [-8.08999993, 1.81388127, 49.8338702, 60.95000002]
                ax.set_extent(extent)
            elif type(map_extent) in [list, tuple]:
                ax.set_extent(map_extent)
            else:
                pass
        else:
            ax = plt.axes()

        for i in range(len(self.metadata)):
            text_id = self.metadata[i]["id"].split(".")[1]
            if text_id in list2remove:
                continue

            coords = np.array(self.metadata[i]["geometry"]["coordinates"][0][0])
            if cartopy_installed:
                plt.plot(
                    coords[:, 0],
                    coords[:, 1],
                    c="r",
                    alpha=0.5,
                    transform=ccrs.Geodetic(),
                )
            else:
                plt.plot(coords[:, 0], coords[:, 1], c="r", alpha=0.5)
            if add_text:
                plt.text(
                    np.mean(coords[:, 0]) - 0.15,
                    np.mean(coords[:, 1]) - 0.05,
                    f"{text_id}",
                    color="r",
                )
        if cartopy_installed:
            ax.gridlines()
        else:
            plt.grid()
        plt.show()

    def hist_published_dates(self, min_date=None, max_date=None):
        """Plot a histogram for published dates

        Parameters
        ----------
        min_date : int, None
            min date for histogram
        max_date : int, None
            max date for histogram
        """
        all_published_date = []
        for one_item in self.metadata:
            _, _, _, published_date = self.extract_region_dates_metadata(one_item)
            all_published_date.append(int(published_date))

        plt.figure(figsize=(10, 5))
        print(
            f"Min/Max published dates: {min(all_published_date)}, {max(all_published_date)}"
        )
        if min_date == None:
            min_date = min(all_published_date)
        if max_date == None:
            max_date = max(all_published_date)
        plt.hist(
            all_published_date,
            align="mid",
            bins=np.arange(int(min_date) - 0.5, int(max_date) + 0.5, 1),
            color="k",
        )
        plt.grid()
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.xlabel("Published date", size=18)
        plt.ylabel("Counts", size=18)
        plt.show()

    def __str__(self):
        info = f"Metada file has {self.__len__()} items."
        info += f"\nDownload URL: {self.download_url}"
        info += f"\nGeometry: {self.geometry}"
        return info

    def __len__(self):
        return len(self.metadata)

    def download_tileserver_rect(
        self,
        mode="queries",
        num_img2test=-1,
        zoom_level=14,
        adjust_mult=0.005,
        retries=1,
        failed_urls_path="failed_urls.txt",
        tile_tmp_dir="tiles",
        output_maps_dirname="maps",
        output_metadata_filename="metadata.csv",
        pixel_closest=None,
        redownload=False,
        id1=0,
        id2=-1,
        min_lat_len=0.05,
        min_lon_len=0.05,
    ):
        """Download maps via tileserver

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
        """

        if not os.path.isdir(output_maps_dirname):
            os.makedirs(output_maps_dirname)

        if not os.path.isdir("geojson"):
            os.makedirs("geojson")

        if mode in ["query", "queries"]:
            self.create_metadata_query()
            metadata = self.metadata_query
        else:
            metadata = self.metadata

        # Header and Footer for GeoJson
        header, footer = create_hf(geom="polygone")

        if id2 < 0:
            metadata = metadata[id1:]
        elif id2 < id1:
            raise ValueError(f"id2 should be > id1.")
        else:
            metadata = metadata[id1:id2]

        # Dataframe to collect some information about results
        pd_info = pd.DataFrame(columns=["name", "url", "coord", "pub_date", "region"])
        counter = 0
        for one_item in metadata:
            # only for testing
            if counter == num_img2test:
                break

            print("-------------------")
            output_stitcher = os.path.join(
                output_maps_dirname, "map_" + one_item["properties"]["IMAGE"] + ".png"
            )
            if os.path.isfile(output_stitcher) and (not redownload):
                print(f"File already exists: {output_stitcher}")
                continue

            print("Retrieving: ", one_item["properties"]["IMAGEURL"])
            # print(one_item["geometry"]["coordinates"])

            # Change the request to a rectangular one by finding min/max of lats/lons
            min_lon, max_lon, min_lat, max_lat = self.detect_rectangle_boundary(
                one_item["geometry"]["coordinates"][0][0]
            )

            # only save an image if no error was raised
            not_saved = True
            while not_saved:
                # failed_urls_path is created by scraper.py
                # remove the file and check if it will be created by scraper
                if os.path.isfile(failed_urls_path):
                    os.remove(failed_urls_path)

                # --- Create a geojson file
                values = """[
                        [{0},  {1}],
                        [{2},  {3}],
                        [{4},  {5}],
                        [{6},  {7}],
                        [{8},  {9}]]""".format(
                    min_lon,
                    min_lat,
                    max_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                    min_lon,
                    max_lat,
                    min_lon,
                    min_lat,
                )
                str2write = header + values + footer

                poly_filename = one_item["properties"]["IMAGE"] + "_{}.geojson".format(
                    counter
                )
                fio = open(os.path.join("geojson", poly_filename), "w")
                fio.writelines(str2write)
                fio.close()

                if os.path.isdir(tile_tmp_dir):
                    # remove previously retrieved tiles
                    shutil.rmtree(tile_tmp_dir)

                # Run the scraper
                scraper(
                    os.path.join("geojson", poly_filename),
                    zoom_level,
                    self.download_url,
                    tile_tmp_dir,
                    retries=retries,
                )

                # if any failed URLs, adjust the request
                # the request is adjusted by decreasing the requested bounding box
                if os.path.isfile(failed_urls_path):
                    with open(failed_urls_path, "r") as fp:
                        errors = fp.readlines()
                        if len(errors) > 0:
                            print(
                                f"[WARNING] Could not retrieve {len(errors)} tiles, changing the coordinates"
                            )
                            lat_len = abs(max_lat - min_lat)
                            lon_len = abs(max_lon - min_lon)
                            if (lat_len <= min_lat_len) or (lon_len <= min_lon_len):
                                not_saved = True
                                break
                            min_lat += adjust_mult * lat_len
                            min_lon += adjust_mult * lon_len
                            max_lat -= adjust_mult * lat_len
                            max_lon -= adjust_mult * lon_len
                            print(lat_len, lon_len)
                        else:
                            not_saved = False
                else:
                    not_saved = False

            if not_saved:
                continue

            # Run stitcher
            stitcher(tile_tmp_dir, output_stitcher, pixel_closest)

            # Collect the new bounding boxes according to the retrieved tiles
            list_files = glob(os.path.join(tile_tmp_dir, "*png"))
            coord_info = collect_coord_info(list_files)

            name_region, _, _, published_date = self.extract_region_dates_metadata(
                one_item
            )

            pd_info.loc[counter] = [
                os.path.basename(output_stitcher),
                one_item["properties"]["IMAGEURL"],
                coord_info,
                published_date,
                name_region,
            ]
            counter += 1

        # clean-up
        if os.path.isdir(tile_tmp_dir):
            shutil.rmtree(tile_tmp_dir)

        pd_output_file = os.path.join(output_maps_dirname, output_metadata_filename)
        if (not redownload) and (os.path.isfile(pd_output_file)):
            pd_info.to_csv(pd_output_file, mode="a+", header=False)
        else:
            pd_info.to_csv(pd_output_file, mode="w", header=True)


# ----------------------
def download_tileserver_parallel(metadata, start, end, process_np=8, **kwds):
    """Run download_tileserver in parallel using multiprocessing

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
    """

    if end < 0:
        end = len(metadata)

    req_proc = min(end, process_np)

    step = int((end - start) / req_proc + 1)

    jobs = []
    for index in range(req_proc):
        starti = start + index * step
        endi = min(start + (index + 1) * step, end)
        if starti >= endi:
            break

        metadata_one_node = metadata[starti:endi]
        p = multiprocessing.Process(
            target=download_tileserver,
            args=(
                metadata_one_node,
                kwds["geometry"],
                kwds["num_img2test"],
                kwds["download_url"],
                kwds["zoom_level"],
                kwds["adjust_mult"],
                kwds["retries"],
                f'{index}_{kwds["failed_urls_path"]}',
                f'{index}_{kwds["tile_tmp_dir"]}',
                f'{kwds["output_maps_dirname"]}',
                f'{index}_{kwds["output_metadata_filename"]}',
            ),
        )
        jobs.append(p)
    for i in range(len(jobs)):
        jobs[i].start()

    check_par_jobs(jobs)
