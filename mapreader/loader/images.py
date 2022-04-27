#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from geopy.distance import geodesic, great_circle
except ImportError:
    pass

from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd
from PIL import Image
from pylab import cm as pltcm
import pyproj
import random
import sys

from mapreader.slicers.slicers import sliceByPixel
from ..utils import utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class mapImages:
    """mapImages class"""

    def __init__(
        self, path_images=False, tree_level="parent", parent_path=None, **kwds
    ):
        """Instantiate mapImages class,

        Keyword Arguments:
            path_images {str or None} -- Path to image(s), accepts wildcard (default: {False})
        """
        if path_images:
            # List with all paths
            self.path_images = glob(os.path.abspath(path_images))
        else:
            self.path_images = []

        # Create images variable (MAIN object variable)
        # New methods (e.g., reading/loading) should construct images this way
        self.images = {}
        self.images["parent"] = {}
        self.images["child"] = {}
        for one_image_path in self.path_images:
            self.imagesConstructor(
                image_path=one_image_path,
                parent_path=parent_path,
                tree_level=tree_level,
                **kwds,
            )

    def imagesConstructor(
        self, image_path, parent_path=None, tree_level="child", **kwds
    ):
        """Construct images instance variable,

        Arguments:
            image_path {str or None} -- Path to the image

        Keyword Arguments:
            parent_path {str or None} -- Path to the parent of image (default: {None})
            tree_level {str} -- Tree level, choices between parent and child (default: {"child"})
        """

        if tree_level not in ["parent", "child"]:
            raise ValueError(
                f"[ERROR] tree_level should be set to parent or child, current value: {tree_level}"
            )
        if (parent_path is not None) and (tree_level == "parent"):
            raise ValueError(
                f"[ERROR] if tree_level=parent, parent_path should be None."
            )

        # Convert the image_path to its absolute path
        image_path = os.path.abspath(image_path)
        image_id, _ = self.splitImagePath(image_path)

        if parent_path:
            parent_path = os.path.abspath(parent_path)
            parent_basename, _ = self.splitImagePath(parent_path)
        else:
            parent_basename, _ = None, None

        # --- Add other info to images
        self.images[tree_level][image_id] = {
            "parent_id": parent_basename,
            "image_path": image_path,
        }

        for k, v in kwds.items():
            self.images[tree_level][image_id][k] = v

        # --- Make sure parent exists in images["parent"]
        if tree_level == "child" and parent_basename:
            # three possible scenarios
            # 1. parent_basename is not in the parent dictionary
            if not parent_basename in self.images["parent"].keys():
                self.images["parent"][parent_basename] = {
                    "parent_id": None,
                    "image_path": parent_path,
                }
            # 2. parent_basename exists but parent_id is not defined
            if not "parent_id" in self.images["parent"][parent_basename].keys():
                self.images["parent"][parent_basename]["parent_id"] = None
            # 3. parent_basename exists but image_path is not defined
            if not "image_path" in self.images["parent"][parent_basename].keys():
                self.images["parent"][parent_basename]["image_path"] = parent_path

    @staticmethod
    def splitImagePath(inp_path):
        """split image path into basename and dirname,"""
        inp_path = os.path.abspath(inp_path)
        path_basename = os.path.basename(inp_path)
        path_dirname = os.path.dirname(inp_path)
        return path_basename, path_dirname

    def __len__(self):
        return int(len(self.images["parent"]) + len(self.images["child"]))

    def __str__(self):
        print(f"#images: {self.__len__()}")

        print(f"\n#parents: {len(self.images['parent'])}")
        for i, img in enumerate(self.images["parent"]):
            print(os.path.relpath(img))
            if i >= 10:
                print("...")
                break

        print(f"\n#children: {len(self.images['child'])}")
        for i, img in enumerate(self.images["child"]):
            print(os.path.relpath(img))
            if i >= 10:
                print("...")
                break
        return ""

    def add_metadata(
        self, metadata, columns=None, tree_level="parent", index_col=0, delimiter="|"
    ):
        """Add metadata to images at tree_level,

        Args:
            metadata_path (path): path to a csv file, normally created from a pandas dataframe
            columns (list, optional): list of columns to be used. If None (default), all columns are used.
            tree_level (str, optional): parent/child tree level. Defaults to "parent".
            index_col (int, optional): index column
        """

        if isinstance(metadata, pd.DataFrame):
            metadata_df = metadata.copy()
        elif os.path.isfile(metadata):
            # read the metadata data
            if index_col >= 0:
                metadata_df = pd.read_csv(
                    metadata, index_col=index_col, delimiter=delimiter
                )
            else:
                metadata_df = pd.read_csv(
                    metadata, index_col=False, delimiter=delimiter
                )
        else:
            raise ValueError(
                f"metadata should either path to a csv file or pandas dataframe."
            )

        # remove duplicates using "name" column
        if columns == None:
            columns = list(metadata_df.columns)

        if ("name" in columns) and ("image_id" in columns):
            print(f"Both 'name' and 'image_id' columns exist! Use 'name' to index.")
            image_id_col = "name"
        if "name" in columns:
            image_id_col = "name"
        elif "image_id" in columns:
            image_id_col = "image_id"
        else:
            raise ValueError("'name' or 'image_id' should be one of the columns.")
        metadata_df.drop_duplicates(subset=[image_id_col])

        for i, one_row in metadata_df.iterrows():
            if not one_row[image_id_col] in self.images[tree_level].keys():
                # print(f"[WARNING] {one_row[image_id_col]} does not exist in images, skip!")
                continue
            for one_col in columns:
                if one_col in ["coord", "polygone"]:
                    # Make sure coord is interpreted as a tuple
                    self.images[tree_level][one_row[image_id_col]][one_col] = eval(
                        one_row[one_col]
                    )
                else:
                    self.images[tree_level][one_row[image_id_col]][one_col] = one_row[
                        one_col
                    ]

    def show_sample(self, num_samples, tree_level="parent", random_seed=65, **kwds):
        """Show sample images,

        Arguments:
            num_samples {int} -- Number of samples to be plotted

        Keyword Arguments:
            tree_level {str} -- XXX (default: {"child"})
            random_seed {int} -- Random seed for reproducibility (default: {65})
        """
        # set random seed for reproducibility
        random.seed(random_seed)

        img_keys = list(self.images[tree_level].keys())
        num_samples = min(len(img_keys), num_samples)
        sample_img_keys = random.sample(img_keys, k=num_samples)

        if not kwds.get("figsize"):
            plt.figure(figsize=(15, num_samples * 2))
        else:
            plt.figure(figsize=kwds["figsize"])

        for i, imid in enumerate(sample_img_keys):
            plt.subplot(num_samples // 3 + 1, 3, i + 1)
            myimg = mpimg.imread(self.images[tree_level][imid]["image_path"])
            plt.title(imid, size=8)
            plt.imshow(myimg)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

    def list_parents(self):
        """Return list of all parents"""
        return list(self.images["parent"].keys())

    def list_children(self):
        """Return list of all children"""
        return list(self.images["child"].keys())

    def add_shape(self, tree_level="parent"):
        """Run add_shape_id for all tree_level items"""
        list_items = list(self.images[tree_level].keys())
        print(f"[INFO] Add shape, tree level: {tree_level}")
        for one_item in list_items:
            self.add_shape_id(image_id=one_item, tree_level=tree_level)

    def add_coord_increments(self, tree_level="parent"):
        """Run add_coord_increments_id for all tree_level items"""
        list_items = list(self.images[tree_level].keys())
        print(f"[INFO] Add coord-increments, tree level: {tree_level}")
        for one_item in list_items:
            self.add_coord_increments_id(image_id=one_item, tree_level=tree_level)

    def add_center_coord(self, tree_level="child"):
        """Run add_center_coord_id for all tree_level items"""
        list_items = list(self.images[tree_level].keys())
        print(f"[INFO] Add center coordinates, tree level: {tree_level}")
        for one_item in list_items:
            self.add_center_coord_id(image_id=one_item, tree_level=tree_level)

    def add_shape_id(self, image_id, tree_level="parent"):
        """Add an image/array shape to self.images[tree_level][image_id]

        Parameters
        ----------
        image_id : str
            image ID
        tree_level : str, optional
            Tree level, choices between parent and child (default: {"child"})
        """
        myimg = mpimg.imread(self.images[tree_level][image_id]["image_path"])
        myimg_shape = myimg.shape
        self.images[tree_level][image_id]["shape"] = myimg_shape

    def add_coord_increments_id(self, image_id, tree_level="parent"):
        """Add pixel-wise dlon and dlat to self.images[tree_level][image_id]

        Parameters
        ----------
        image_id : str
            image ID
        tree_level : str, optional
            Tree level, choices between parent and child (default: {"child"})
        """
        if not "shape" in self.images[tree_level][image_id].keys():
            self.add_shape(tree_level=tree_level)

        if not "coord" in self.images[tree_level][image_id].keys():
            raise ValueError(
                f"'coord' could not be found in: self.images[tree_level][image_id].keys(). Suggestion: run add_metadata"
            )

        # Extract height/width/chan from shape
        hwc = self.images[tree_level][image_id]["shape"]
        lon_min, lon_max, lat_min, lat_max = self.images[tree_level][image_id]["coord"]
        self.images[tree_level][image_id]["dlon"] = abs(lon_max - lon_min) / hwc[1]
        self.images[tree_level][image_id]["dlat"] = abs(lat_max - lat_min) / hwc[0]

    def add_center_coord_id(self, image_id, tree_level="child"):
        """Add center_lon and center_lat to self.images[tree_level][image_id]

        Parameters
        ----------
        image_id : str
            image ID
        tree_level : str, optional
            Tree level, choices between parent and child (default: {"child"})
        """

        par_id = self.images[tree_level][image_id]["parent_id"]

        if par_id is not None:
            if (not "dlon" in self.images["parent"][par_id].keys()) or (
                not "dlat" in self.images["parent"][par_id].keys()
            ):
                self.add_coord_increments(tree_level="parent")

            dlon = self.images["parent"][par_id]["dlon"]
            dlat = self.images["parent"][par_id]["dlat"]
            lon_min, lon_max, lat_min, lat_max = self.images["parent"][par_id]["coord"]

            min_abs_x = self.images[tree_level][image_id]["min_x"] * dlon
            max_abs_x = self.images[tree_level][image_id]["max_x"] * dlon
            min_abs_y = self.images[tree_level][image_id]["min_y"] * dlat
            max_abs_y = self.images[tree_level][image_id]["max_y"] * dlat

            self.images[tree_level][image_id]["center_lon"] = (
                lon_min + (min_abs_x + max_abs_x) / 2.0
            )
            self.images[tree_level][image_id]["center_lat"] = (
                lat_max - (min_abs_y + max_abs_y) / 2.0
            )
        else:
            print(
                f"[INFO] parent_id is None. Read 'coord' from the specified tree_level and image_id."
            )
            lon_min, lon_max, lat_min, lat_max = self.images[tree_level][image_id][
                "coord"
            ]

            self.images[tree_level][image_id]["center_lon"] = (lon_min + lon_max) / 2.0
            self.images[tree_level][image_id]["center_lat"] = (lat_min + lat_max) / 2.0

    def calc_pixel_width_height(self, parent_id, calc_size_in_m="great-circle"):
        """Calculate width and height of pixels

        Args:
            parent_id (str): ID of the parent image
            calc_size_in_m (str, optional): How to compute the width/heigh, options: geodesic and great-circle (default).
        """
        myimg = mpimg.imread(self.images["parent"][parent_id]["image_path"])
        myimg_shape = myimg.shape
        if not "coord" in self.images["parent"][parent_id].keys():
            raise ValueError(f"[ERROR] could not find coordinate info for: {parent_id}")
        (xmin, xmax, ymin, ymax) = self.images["parent"][parent_id]["coord"]
        print(f"[INFO] Use the following coordinates to compute width/height:")
        print(f"[INFO] lon min/max: {xmin:.4f}/{xmax:.4f}")
        print(f"[INFO] lat min/max: {ymin:.4f}/{ymax:.4f}")
        print(f"[INFO] shape: {myimg_shape}")

        # Calculate the size of image in meters
        if calc_size_in_m == "geodesic":
            bottom = geodesic((ymin, xmin), (ymin, xmax)).meters
            right = geodesic((ymin, xmax), (ymax, xmax)).meters
            top = geodesic((ymax, xmax), (ymax, xmin)).meters
            left = geodesic((ymax, xmin), (ymin, xmin)).meters
            size_in_m = (bottom, top, left, right)
            print(
                f"[INFO] size (in meters) bottom/top/left/right: {bottom:.2f}/{top:.2f}/{left:.2f}/{right:.2f}"
            )

            mean_width = np.mean(
                [size_in_m[0] / myimg_shape[1], size_in_m[1] / myimg_shape[1]]
            )
            mean_height = np.mean(
                [size_in_m[2] / myimg_shape[0], size_in_m[3] / myimg_shape[0]]
            )
            print(
                f"\nEach pixel is ~{mean_width:.3f} X {mean_height:.3f} meters (width x height)."
            )

        elif calc_size_in_m in ["gc", "great-circle"]:
            bottom = great_circle((ymin, xmin), (ymin, xmax)).meters
            right = great_circle((ymin, xmax), (ymax, xmax)).meters
            top = great_circle((ymax, xmax), (ymax, xmin)).meters
            left = great_circle((ymax, xmin), (ymin, xmin)).meters
            size_in_m = (bottom, top, left, right)
            print(
                f"[INFO] size (in meters) bottom/top/left/right: {bottom:.2f}/{top:.2f}/{left:.2f}/{right:.2f}"
            )

            mean_width = np.mean(
                [size_in_m[0] / myimg_shape[1], size_in_m[1] / myimg_shape[1]]
            )
            mean_height = np.mean(
                [size_in_m[2] / myimg_shape[0], size_in_m[3] / myimg_shape[0]]
            )
            print(
                f"\nEach pixel is ~{mean_width:.3f} x {mean_height:.3f} meters (width x height)."
            )

        return xmin, xmax, ymin, ymax, myimg_shape, size_in_m

    def sliceAll(
        self,
        method="pixel",
        slice_size=100,
        path_save="test",
        square_cuts=False,
        resize_factor=False,
        output_format="PNG",
        rewrite=False,
        verbose=False,
        tree_level="parent",
        add2child=True,
        id1=0,
        id2=-1,
    ):
        """Slice all images in the object (the list can be accessed via .images variable)

        Keyword Arguments:
            method {str} -- method to slice an image (default: {"pixel"})
            slice_size {int} -- Number of pixels in both x and y directions (default: {100})
            path_save {str} -- Directory to save the sliced images (default: {"test"})
            square_cuts {bool} -- All sliced images will have the same number of pixels in x and y (default: {True})
            resize_factor {bool} -- Resize image before slicing (default: {False})
            output_format {str} -- Output format (default: {"PNG"})
            tree_level {str} -- image group to be sliced (default: {"parent"})
            verbose {bool} -- Print the progress (default: {False})
        """

        if id2 < 0:
            img_keys = list(self.images[tree_level].keys())[id1:]
        elif id2 < id1:
            raise ValueError(f"id2 should be > id1.")
        else:
            img_keys = list(self.images[tree_level].keys())[id1:id2]

        for one_image in img_keys:
            sliced_images_info = self._slice(
                image_path=self.images[tree_level][one_image]["image_path"],
                method=method,
                slice_size=slice_size,
                path_save=path_save,
                square_cuts=square_cuts,
                resize_factor=resize_factor,
                output_format=output_format,
                rewrite=rewrite,
                verbose=verbose,
                image_id=one_image,
                tree_level=tree_level,
            )

            if add2child:
                for i in range(len(sliced_images_info)):
                    # Add sliced images to the .images["child"]
                    self.imagesConstructor(
                        image_path=sliced_images_info[i][0],
                        parent_path=self.images[tree_level][one_image]["image_path"],
                        tree_level="child",
                        min_x=sliced_images_info[i][1][0],
                        min_y=sliced_images_info[i][1][1],
                        max_x=sliced_images_info[i][1][2],
                        max_y=sliced_images_info[i][1][3],
                    )
        if add2child:
            # add children to the parent dictionary
            self.addChildren()

    def _slice(
        self,
        image_path=False,
        method="pixel",
        slice_size=100,
        path_save="test",
        square_cuts=False,
        resize_factor=False,
        output_format="PNG",
        rewrite=False,
        verbose=True,
        image_id=None,
        tree_level=None,
    ):
        """Slice one image stored at image_path

        Keyword Arguments:
            image_path {str} -- Path to the image to be sliced (default: {False})
            method {str} -- method to slice an image (default: {"pixel"})
            slice_size {int} -- Number of pixels in both x and y directions (default: {100})
            path_save {str} -- Directory to save the sliced images (default: {"test"})
            square_cuts {bool} -- All sliced images will have the same number of pixels in x and y (default: {True})
            resize_factor {bool} -- Resize image before slicing (default: {False})
            output_format {str} -- Output format (default: {"PNG"})
            verbose {bool} -- Print the progress (default: {True})
            image_id {str} -- image ID
            tree_level {str} -- image group to be sliced (default: {"parent"})
        """
        print(40 * "=")
        print(f"Slicing {os.path.relpath(image_path)}")
        print(40 * "-")

        # make sure the dir exists
        self._makeDir(path_save)

        # which image should be sliced
        image_path = os.path.abspath(image_path)
        sliced_images_info = None
        if method == "pixel":
            sliced_images_info = sliceByPixel(
                image_path=image_path,
                slice_size=slice_size,
                path_save=path_save,
                square_cuts=square_cuts,
                resize_factor=resize_factor,
                output_format=output_format,
                rewrite=rewrite,
                verbose=verbose,
            )

        elif method in ["meters", "meter"]:
            if "tiff" in image_path:
                xmin, xmax, ymin, ymax, tiff_shape, size_in_m = utils.extractGeoInfo(
                    image_path=image_path, calc_size_in_m="great-circle"
                )
                print(f"[DEBUG] tiff_shape: {tiff_shape}")
                pixel_height = size_in_m[2] / tiff_shape[1]
                number_pixels4slice = int(slice_size / pixel_height)

                sliced_images_info = sliceByPixel(
                    image_path=image_path,
                    slice_size=number_pixels4slice,
                    path_save=path_save,
                    square_cuts=square_cuts,
                    resize_factor=resize_factor,
                    output_format=output_format,
                    rewrite=rewrite,
                    verbose=verbose,
                )
            elif "png" in image_path.lower():
                (
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    img_shape,
                    size_in_m,
                ) = self.calc_pixel_width_height(image_id)
                # size in meter contains: (bottom, top, left, right)
                # img_shape = (rows, columns)
                pixel_height = size_in_m[2] / img_shape[0]
                number_pixels4slice = int(slice_size / pixel_height)

                sliced_images_info = sliceByPixel(
                    image_path=image_path,
                    slice_size=number_pixels4slice,
                    path_save=path_save,
                    square_cuts=square_cuts,
                    resize_factor=resize_factor,
                    output_format=output_format,
                    rewrite=rewrite,
                    verbose=verbose,
                )
        return sliced_images_info

    def addChildren(self):
        """Add children to parent"""
        for child in self.images["child"].keys():
            my_parent = self.images["child"][child]["parent_id"]
            if not self.images["parent"][my_parent].get("children", False):
                self.images["parent"][my_parent]["children"] = [child]
            else:
                if not child in self.images["parent"][my_parent]["children"]:
                    self.images["parent"][my_parent]["children"].append(child)

    def _makeDir(self, path_make, exists_ok=True):
        """helper function to make directories"""
        os.makedirs(path_make, exist_ok=exists_ok)

    def calc_pixel_stats(self, parent_id=None, calc_mean=True, calc_std=True):
        """Calculate stats of each child in a parent_id and
           store the results

        Arguments:
            parent_id {str, None} -- ID of the parent image. If None, all parents will be used.
        """
        if parent_id is None:
            parent_id = self.list_parents()
        else:
            parent_id = [parent_id]

        for one_par_id in parent_id:
            print(10 * "-")
            print(f"[INFO] calculate pixel stats for image: {one_par_id}")
            if not "children" in self.images["parent"][one_par_id]:
                print(f"[WARNING] No child found for: {one_par_id}")
                continue
            list_children = self.images["parent"][one_par_id]["children"]
            for one_child in list_children:
                if ("mean_pixel_A" in self.images["child"][one_child].keys()) and (
                    "std_pixel_A" in self.images["child"][one_child].keys()
                ):
                    continue
                child_img = mpimg.imread(self.images["child"][one_child]["image_path"])
                if calc_mean:
                    self.images["child"][one_child]["mean_pixel_R"] = np.mean(
                        child_img[:, :, 0]
                    )
                    self.images["child"][one_child]["mean_pixel_G"] = np.mean(
                        child_img[:, :, 1]
                    )
                    self.images["child"][one_child]["mean_pixel_B"] = np.mean(
                        child_img[:, :, 2]
                    )
                    self.images["child"][one_child]["mean_pixel_RGB"] = np.mean(
                        child_img[:, :, 0:3]
                    )
                    self.images["child"][one_child]["mean_pixel_A"] = np.mean(
                        child_img[:, :, 3]
                    )
                if calc_std:
                    self.images["child"][one_child]["std_pixel_R"] = np.std(
                        child_img[:, :, 0]
                    )
                    self.images["child"][one_child]["std_pixel_G"] = np.std(
                        child_img[:, :, 1]
                    )
                    self.images["child"][one_child]["std_pixel_B"] = np.std(
                        child_img[:, :, 2]
                    )
                    self.images["child"][one_child]["std_pixel_RGB"] = np.std(
                        child_img[:, :, 0:3]
                    )
                    self.images["child"][one_child]["std_pixel_A"] = np.std(
                        child_img[:, :, 3]
                    )

    def convertImages(self, fmt="dataframe"):
        """Convert images to a specified format (fmt)

        Keyword Arguments:
            fmt {str} -- convert images variable to this format (default: {"dataframe"})
        """
        if fmt in ["pandas", "dataframe"]:
            children = pd.DataFrame.from_dict(self.images["child"], orient="index")
            parents = pd.DataFrame.from_dict(self.images["parent"], orient="index")
            return parents, children
        else:
            raise ValueError(f"Format {fmt} is not supported!")

    def show_par(self, parent_id, value=False, **kwds):
        """A wrapper function for show,

        Arguments:
            parent_id {str} -- ID of the parent image to be plotted

        Keyword Arguments:
            value {bool, const, random, ...} -- Values to be plotted on the parent image (default: {False})
        """
        image_ids = self.images["parent"][parent_id]["children"]
        self.show(image_ids, value=value, **kwds)

    def show(
        self,
        image_ids,
        value=False,
        plot_parent=True,
        border=True,
        border_color="r",
        vmin=0.5,
        vmax=2.5,
        colorbar="jet",
        alpha=1.0,
        discrete_colorbar=256,
        tree_level="child",
        grid_plot=(20000, 20000),
        plot_histogram=True,
        save_kml_dir=False,
        image_width_resolution=None,
        kml_dpi_image=None,
        **kwds,
    ):
        """Plot a list of image ids,

        Arguments:
            image_ids {list} -- List of image ids to be plotted

        Keyword Arguments:
            value {False or list} -- Value to be plotted on child images
            plot_parent {bool} -- Plot parent image in the background (default: {True})
            border {bool} -- Plot a border for each image id (default: {True})
            border_color {str} -- color of patch borders (default: {r})
            vmin {float or list} -- min. value for the colorbar (default: {0.5})
            vmax {float or list} -- max. value for the colorbar (default: {2.5})
            colorbar {str or list} -- colorbar to visualize "value" on maps (default: {jet})
            alpha {float or list} -- set transparency level for plotting "value" on maps (default: {1.})
            discrete_colorbar {int or list} -- number of discrete colors to be used (default: {256})
            tree_level {str} -- Tree level for the plot XXX (default: {"child"})
            grid_plot {list or tuple} -- Number of rows and columns in the image.
                                         This will later adjusted to the true min/max of all subplots.
                                         (default: (10000, 10000))
            plot_histogram {bool} -- Plot a histogram of 'value' (default: {True})
            save_kml_dir {False or str} -- Directory to save a KML files out of images or False
                                           (default: {False})
            image_width_resolution {None, int} -- pixel width to be used for plotting, only when tree_level="parent"
                                                  pixel height will be adjusted according to the width/height ratio
            kml_dpi_image {None, int} -- The resolution in dots per inch for images created when save_kml_dir is specified
        """
        # create list, if not already a list
        if not (isinstance(image_ids, list) or isinstance(image_ids, tuple)):
            image_ids = [image_ids]
        values = [value] if not (isinstance(value, list)) else value[:]
        vmins = [vmin] if not (isinstance(vmin, list)) else vmin[:]
        vmaxs = [vmax] if not (isinstance(vmax, list)) else vmax[:]
        colorbars = [colorbar] if not (isinstance(colorbar, list)) else colorbar[:]
        alphas = [alpha] if not (isinstance(alpha, list)) else alpha[:]
        discrete_colorbars = (
            [discrete_colorbar]
            if not (isinstance(discrete_colorbar, list))
            else discrete_colorbar[:]
        )

        if tree_level == "parent":
            for one_image_id in image_ids:
                figsize = self._get_kwds(kwds, "figsize")
                plt.figure(figsize=figsize)

                par_path = self.images["parent"][one_image_id]["image_path"]
                par_image = Image.open(par_path)

                # Change the resolution of the image if image_width_resolution is specified
                if image_width_resolution is not None:
                    basewidth = int(image_width_resolution)
                    wpercent = basewidth / float(par_image.size[0])
                    hsize = int((float(par_image.size[1]) * float(wpercent)))
                    par_image = par_image.resize((basewidth, hsize), Image.ANTIALIAS)

                # remove the borders
                plt.gca().set_axis_off()
                plt.subplots_adjust(
                    top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                )

                plt.imshow(
                    par_image,
                    zorder=1,
                    interpolation="nearest",
                    extent=(0, par_image.size[0], par_image.size[1], 0),
                )

                if save_kml_dir:
                    if not "coord" in self.images["parent"][one_image_id].keys():
                        print(
                            f"[WARNING] 'coord' could not be found. This is needed when save_kml_dir is set...continue"
                        )
                        continue

                    # remove x/y ticks when creating KML out of images
                    plt.xticks([])
                    plt.yticks([])

                    os.makedirs(save_kml_dir, exist_ok=True)
                    path2kml = os.path.join(save_kml_dir, f"{one_image_id}")
                    plt.savefig(
                        f"{path2kml}",
                        bbox_inches="tight",
                        pad_inches=0,
                        dpi=kml_dpi_image,
                    )

                    self._createKML(
                        path2kml=path2kml,
                        value=one_image_id,
                        coords=self.images["parent"][one_image_id]["coord"],
                        counter=-1,
                    )
                else:
                    plt.title(one_image_id)
                    plt.show()

        elif tree_level == "child":
            # Collect parents information
            parents = {}
            for i in range(len(image_ids)):
                try:
                    parent_id = self.images[tree_level][image_ids[i]]["parent_id"]
                except Exception as err:
                    print(err)
                    continue
                if not parent_id in parents:
                    parents[parent_id] = {
                        "path": self.images["parent"][parent_id]["image_path"],
                        "child": [],
                    }
                parents[parent_id]["child"].append(image_ids[i])

            for i in parents.keys():
                figsize = self._get_kwds(kwds, "figsize")
                plt.figure(figsize=figsize)

                for i_value, value in enumerate(values):
                    # initialize image2plot
                    # will be filled with values of 'value'
                    image2plot = np.empty(grid_plot)
                    image2plot[:] = np.nan
                    min_x, min_y, max_x, max_y = grid_plot[1], grid_plot[0], 0, 0
                    for child_id in parents[i]["child"]:
                        one_image = self.images[tree_level][child_id]

                        # Set the values for each child
                        if not value:
                            pass

                        elif value == "const":
                            # assign values to image2plot, update min_x, min_y, ...
                            image2plot[
                                one_image["min_y"] : one_image["max_y"],
                                one_image["min_x"] : one_image["max_x"],
                            ] = 1.0

                        elif value == "random":
                            import random

                            # assign values to image2plot, update min_x, min_y, ...
                            image2plot[
                                one_image["min_y"] : one_image["max_y"],
                                one_image["min_x"] : one_image["max_x"],
                            ] = random.random()

                        elif value:
                            if not value in one_image:
                                assign_value = None
                            else:
                                assign_value = one_image[value]
                            # assign values to image2plot, update min_x, min_y, ...
                            image2plot[
                                one_image["min_y"] : one_image["max_y"],
                                one_image["min_x"] : one_image["max_x"],
                            ] = assign_value

                        # reset min/max of x and y
                        min_x = min(min_x, one_image["min_x"])
                        min_y = min(min_y, one_image["min_y"])
                        max_x = max(max_x, one_image["max_x"])
                        max_y = max(max_y, one_image["max_y"])

                        if border:
                            self._plotBorder(one_image, plt, color=border_color)

                    if value:
                        vmin = vmins[i_value]
                        vmax = vmaxs[i_value]
                        alpha = alphas[i_value]
                        colorbar = colorbars[i_value]
                        discrete_colorbar = discrete_colorbars[i_value]

                        # set discrete colorbar
                        colorbar = pltcm.get_cmap(colorbar, discrete_colorbar)

                        # Adjust image2plot to global min/max in x and y directions
                        image2plot = image2plot[min_y:max_y, min_x:max_x]
                        plt.imshow(
                            image2plot,
                            zorder=10 + i_value,
                            interpolation="nearest",
                            cmap=colorbar,
                            vmin=vmin,
                            vmax=vmax,
                            alpha=alpha,
                            extent=(min_x, max_x, max_y, min_y),
                        )

                        if save_kml_dir:
                            plt.xticks([])
                            plt.yticks([])
                        else:
                            plt.colorbar(fraction=0.03)

                if plot_parent:
                    par_path = os.path.join(parents[i]["path"])
                    par_image = mpimg.imread(par_path)
                    plt.imshow(
                        par_image,
                        zorder=1,
                        interpolation="nearest",
                        extent=(0, par_image.shape[1], par_image.shape[0], 0),
                    )
                    if not save_kml_dir:
                        plt.title(i)

                if not save_kml_dir:
                    plt.show()
                else:
                    os.makedirs(save_kml_dir, exist_ok=True)
                    path2kml = os.path.join(save_kml_dir, f"{value}_{i}")
                    plt.savefig(
                        f"{path2kml}",
                        bbox_inches="tight",
                        pad_inches=0,
                        dpi=kml_dpi_image,
                    )

                    self._createKML(
                        path2kml=path2kml,
                        value=value,
                        coords=self.images["parent"][i]["coord"],
                        counter=i,
                    )

                if value and plot_histogram:
                    histogram_range = self._get_kwds(kwds, "histogram_range")
                    plt.figure(figsize=(7, 5))
                    plt.hist(
                        image2plot.flatten(),
                        color="k",
                        bins=np.arange(
                            histogram_range[0] - histogram_range[2] / 2.0,
                            histogram_range[1] + histogram_range[2],
                            histogram_range[2],
                        ),
                    )
                    plt.xlabel(value, size=20)
                    plt.ylabel("Freq.", size=20)
                    plt.xticks(size=18)
                    plt.yticks(size=18)
                    plt.grid()
                    plt.show()

    def _createKML(self, path2kml, value, coords, counter=-1):
        """Create a KML file,

        Args:
            path2kml (str) -- Path to save a KML file
            value (str) -- Value plotted on the underlying image (refer to "show" function)
            coords (list, tuple) -- coordinates of the bounding box
        """

        try:
            import simplekml
        except:
            raise ImportError(
                "[ERROR] simplekml needs to be installed to create KML outputs!"
            )

        (lon_min, lon_max, lat_min, lat_max) = coords

        # -----> create KML
        kml = simplekml.Kml()
        ground = kml.newgroundoverlay(name=str(counter))
        if counter == -1:
            ground.icon.href = f"./{value}"
        else:
            ground.icon.href = f"./{value}_{counter}"

        ground.latlonbox.north = lat_max
        ground.latlonbox.south = lat_min
        ground.latlonbox.east = lon_max
        ground.latlonbox.west = lon_min
        # ground.latlonbox.rotation = -14

        kml.save(f"{path2kml}.kml")

    def _plotBorder(self, image_dict, plt, linewidth=0.5, zorder=20, color="r"):
        """Plot border for an image

        Arguments:
            image_dict {dict} -- image dictionary, e.g., one item in self.images["child"]
            plt {matplotlib.pyplot object} -- a matplotlib.pyplot object

        Keyword Arguments:
            linewidth {int} -- line-width (default: {2})
            zorder {int} -- z-order for the border (default: {5})
            color {str} -- color of the border (default: {"r"})
        """
        plt.plot(
            [image_dict["min_x"], image_dict["min_x"]],
            [image_dict["min_y"], image_dict["max_y"]],
            lw=linewidth,
            zorder=zorder,
            color=color,
        )
        plt.plot(
            [image_dict["min_x"], image_dict["max_x"]],
            [image_dict["max_y"], image_dict["max_y"]],
            lw=linewidth,
            zorder=zorder,
            color=color,
        )
        plt.plot(
            [image_dict["max_x"], image_dict["max_x"]],
            [image_dict["max_y"], image_dict["min_y"]],
            lw=linewidth,
            zorder=zorder,
            color=color,
        )
        plt.plot(
            [image_dict["max_x"], image_dict["min_x"]],
            [image_dict["min_y"], image_dict["min_y"]],
            lw=linewidth,
            zorder=zorder,
            color=color,
        )

    @staticmethod
    def _get_kwds(kwds, key):
        """If kwds dictionary has the key, return value; otherwise, use default,"""
        if key in kwds:
            return kwds[key]
        else:
            if key == "figsize":
                return (10, 10)
            elif key in ["vmin"]:
                return 0
            elif key in ["vmax", "alpha"]:
                return 1
            elif key in ["colorbar"]:
                return "binary"
            elif key in ["histogram_range"]:
                return [0, 1, 0.01]
            elif key in ["discrete_colorbar"]:
                return 100

    def loadPatches(
        self, patch_paths, parent_paths=False, add_geo_par=False, clear_images=False
    ):
        """load patches from files (patch_paths) and add parents if parent_paths is provided

        Arguments:
            patch_paths {str, wildcard accepted} -- path to patches
            parent_paths {False or str, wildcard accepted} -- path to parents

        Keyword Arguments:
            clear_images {bool} -- clear images variable before loading patches (default: {False})
        """
        patch_paths = glob(os.path.abspath(patch_paths))

        if clear_images:
            self.images = {}
            self.images["parent"] = {}
            self.images["child"] = {}

        for tpath in patch_paths:

            if not os.path.isfile(tpath):
                print(f"[WARNING] file does not exist: {tpath}")
                continue

            # patch ID is set to the basename
            patch_id = os.path.basename(tpath)

            # Parent ID and border can be detected using patch_id
            parent_id = self.detectParIDfromPath(patch_id)
            min_x, min_y, max_x, max_y = self.detectBorderFromPath(patch_id)

            # Add child
            if not self.images["child"].get(patch_id, False):
                self.images["child"][patch_id] = {}
            self.images["child"][patch_id]["parent_id"] = parent_id
            self.images["child"][patch_id]["image_path"] = tpath
            self.images["child"][patch_id]["min_x"] = min_x
            self.images["child"][patch_id]["min_y"] = min_y
            self.images["child"][patch_id]["max_x"] = max_x
            self.images["child"][patch_id]["max_y"] = max_y

        if parent_paths:
            # Add parents
            self.loadParents(
                parent_paths=parent_paths, update=False, add_geo=add_geo_par
            )
            # Add children to the parent
            self.addChildren()

    @staticmethod
    def detectParIDfromPath(image_id, parent_delimiter="#"):
        """
        Detect parent ID from path using parent_delimiter
        NOTE: Currently, only one parent can be detected.
        """
        return image_id.split(parent_delimiter)[1]

    @staticmethod
    def detectBorderFromPath(image_id, border_delimiter="-"):
        """
        Detect borders from the path using border_delimiter.
        Here, the assumption is that the child image is named:
        NOTE: STRING-min_x-min_y-max_x-max_y-STRING
        """
        split_path = image_id.split("-")
        return (
            int(split_path[1]),
            int(split_path[2]),
            int(split_path[3]),
            int(split_path[4]),
        )

    def loadParents(
        self, parent_paths=False, parent_ids=False, update=False, add_geo=False
    ):
        """load parent images from files (parent_paths)
           if only parent_ids is specified, self.images["parent"] will be filled with no image_path.
           NOTE: if parent_paths is given, parent_ids will be omitted as ids will be
                 detected from the basename

        Keyword Arguments:
            parent_paths {False or str, wildcard accepted} -- path to parents (default: {False})
            parent_ids {False or list/tuple} -- list of parent ids (default: {False})
        """

        if parent_paths:
            if not isinstance(parent_paths, list):
                parent_paths = glob(os.path.abspath(parent_paths))
            if update:
                self.images["parent"] = {}

            for ppath in parent_paths:
                parent_id = os.path.basename(ppath)
                self.images["parent"][parent_id] = {"parent_id": None}
                if os.path.isfile(ppath):
                    self.images["parent"][parent_id]["image_path"] = os.path.abspath(
                        ppath
                    )
                else:
                    self.images["parent"][parent_id]["image_path"] = None
                if add_geo and ("tiff" in ppath):
                    (
                        lon_min,
                        lon_max,
                        lat_min,
                        lat_max,
                        tiff_shape,
                        size_in_m,
                    ) = utils.extractGeoInfo(ppath, calc_size_in_m="great-circle")
                    self.images["parent"][parent_id]["coord"] = (
                        lon_min,
                        lon_max,
                        lat_min,
                        lat_max,
                    )
                    self.images["parent"][parent_id]["shape"] = tiff_shape
                    self.images["parent"][parent_id]["size_in_m"] = size_in_m

        elif parent_ids:
            if not isinstance(parent_ids, list):
                parent_ids = [parent_ids]
            for parent_id in parent_ids:
                self.images["parent"][parent_id] = {"parent_id": None}
                self.images["parent"][parent_id]["image_path"] = None

    def loadDataframe(self, parents=None, children_df=None, clear_images=True):
        """Read dataframes and form images variable

        Keyword Arguments:
            parents_df {dataframe or path} -- Parents dataframe or path to parents (default: {None})
            children_df {dataframe} -- Children/slices dataframe (default: {None})
            clear_images {bool} -- clear images before reading dataframes (default: {True})
        """
        if clear_images:
            self.images = {}
            self.images["parent"] = {}
            self.images["child"] = {}
        if not isinstance(children_df, type(None)):
            self.images["child"] = children_df.to_dict(orient="index")
        if not isinstance(parents, type(None)):
            if isinstance(parents, str):
                self.loadParents(parents)
            else:
                self.images["parent"] = parents.to_dict(orient="index")
            for one_par in self.images["parent"].keys():

                # Do we need this?
                # k2change = "children"
                # if k2change in self.images["parent"][one_par]:
                #    try:
                #        self.images["parent"][one_par][k2change] = self.images["parent"][one_par][k2change]
                #    except Exception as err:
                #        print(err)

                k2change = "coord"
                if k2change in self.images["parent"][one_par]:
                    try:
                        self.images["parent"][one_par][k2change] = self.images[
                            "parent"
                        ][one_par][k2change]
                    except Exception as err:
                        print(err)

            self.addChildren()

    def load_csv_file(
        self,
        parent_path=None,
        child_path=None,
        clear_images=False,
        index_col_child=0,
        index_col_parent=0,
    ):
        """Read parent and child from CSV files"""
        if clear_images:
            self.images = {}
            self.images["parent"] = {}
            self.images["child"] = {}
        if isinstance(child_path, str) and os.path.isfile(child_path):
            self.images["child"].update(
                pd.read_csv(child_path, index_col=index_col_child).to_dict(
                    orient="index"
                )
            )
        if isinstance(parent_path, str) and os.path.isfile(parent_path):
            self.images["parent"].update(
                pd.read_csv(parent_path, index_col=index_col_parent).to_dict(
                    orient="index"
                )
            )
            self.addChildren()
            for one_par in self.images["parent"].keys():
                k2change = "children"
                if k2change in self.images["parent"][one_par]:
                    self.images["parent"][one_par][k2change] = eval(
                        self.images["parent"][one_par][k2change]
                    )

                k2change = "coord"
                if k2change in self.images["parent"][one_par]:
                    self.images["parent"][one_par][k2change] = eval(
                        self.images["parent"][one_par][k2change]
                    )

    ### def readPatches(self,
    ###               patch_paths,
    ###               parent_paths,
    ###               metadata=None,
    ###               metadata_fmt="dataframe",
    ###               metadata_cols2add=[],
    ###               metadata_index_column="image_id",
    ###               clear_images=False):
    ###     """read patches from files (patch_paths) and add parents if parent_paths is provided
    ###
    ###     Arguments:
    ###         patch_paths {str, wildcard accepted} -- path to patches
    ###         parent_paths {False or str, wildcard accepted} -- path to parents
    ###
    ###     Keyword Arguments:
    ###         clear_images {bool} -- clear images variable before reading patches (default: {False})
    ###     """
    ###     patch_paths = glob(os.path.abspath(patch_paths))

    ###     if clear_images:
    ###         self.images = {}
    ###         self.images["parent"] = {}
    ###         self.images["child"] = {}
    ###
    ###     # XXX check
    ###     if not isinstance(metadata, type(None)):
    ###         include_metadata = True
    ###         if metadata_fmt in ["dataframe"]:
    ###             metadata_df = metadata
    ###         elif metadata_fmt.lower() in ["csv"]:
    ###             try:
    ###                 metadata_df = pd.read_csv(metadata)
    ###             except:
    ###                 print(f"[WARNING] could not find metadata file: {metadata}")
    ###         else:
    ###             print(f"format cannot be recognized: {metadata_fmt}")
    ###             include_metadata = False
    ###         if include_metadata:
    ###             metadata_df['rd_index_id'] = metadata_df[metadata_index_column].apply(lambda x: os.path.basename(x))
    ###     else:
    ###         include_metadata = False
    ###
    ###     for tpath in patch_paths:
    ###         tpath = os.path.abspath(tpath)
    ###         if not os.path.isfile(tpath):
    ###             raise ValueError(f"patch_paths should point to actual files. Current patch_paths: {patch_paths}")
    ###         # patch ID is set to the basename
    ###         patch_id = os.path.basename(tpath)
    ###         # XXXX
    ###         if include_metadata and (not patch_id in list(metadata['rd_index_id'])):
    ###             continue
    ###         # Parent ID and border can be detected using patch_id
    ###         parent_id = self.detectParIDfromPath(patch_id)
    ###         min_x, min_y, max_x, max_y = self.detectBorderFromPath(patch_id)

    ###         # Add child
    ###         if not self.images["child"].get(patch_id, False):
    ###             self.images["child"][patch_id] = {}
    ###         self.images["child"][patch_id]["parent_id"] = parent_id
    ###         self.images["child"][patch_id]["image_path"] = tpath
    ###         self.images["child"][patch_id]["min_x"] = min_x
    ###         self.images["child"][patch_id]["min_y"] = min_y
    ###         self.images["child"][patch_id]["max_x"] = max_x
    ###         self.images["child"][patch_id]["max_y"] = max_y

    ###     # XXX check
    ###     if include_metadata:
    ###         # metadata_cols = set(metadata_df.columns) - set(['rd_index_id'])
    ###         for one_row in metadata_df.iterrows():
    ###             for one_col in list(metadata_cols2add):
    ###                 self.images["child"][one_row[1]['rd_index_id']][one_col] = one_row[1][one_col]

    ###     if parent_paths:
    ###         # Add parents
    ###         self.readParents(parent_paths=parent_paths)
    ###         # Add children to the parent
    ###         self.addChildren()

    ### def process(self, tree_level="parent", update_paths=True,
    ###             save_preproc_dir="./test_preproc"):
    ###     """Process images using process.py module

    ###     Args:
    ###         tree_level (str, optional): "parent" or "child" paths will be used. Defaults to "parent".
    ###         update_paths (bool, optional): XXX. Defaults to True.
    ###         save_preproc_dir (str, optional): Path to store preprocessed images. Defaults to "./test_preproc".
    ###     """
    ###
    ###     from mapreader import process
    ###     # Collect paths and store them self.process_paths
    ###     self.getProcessPaths(tree_level=tree_level)

    ###     saved_paths = process.preprocess_all(self.process_paths,
    ###                                          save_preproc_dir=save_preproc_dir)
    ###     if update_paths:
    ###         self.readParents(saved_paths, update=True)

    ### def getProcessPaths(self, tree_level="parent"):
    ###     """Create a list of paths to be processed

    ###     Args:
    ###         tree_level (str, optional): "parent" or "child" paths will be used. Defaults to "parent".
    ###     """
    ###     process_paths = []
    ###     for one_img in self.images[tree_level].keys():
    ###         process_paths.append(self.images[tree_level][one_img]["image_path"])
    ###     self.process_paths = process_paths
    ###
    ###

    ### def prepare4inference(self, fmt="dataframe"):
    ###     """Convert images to the specified format (fmt)
    ###
    ###     Keyword Arguments:
    ###         fmt {str} -- convert images variable to this format (default: {"dataframe"})
    ###     """
    ###     if fmt in ["pandas", "dataframe"]:
    ###         children = pd.DataFrame.from_dict(self.images["child"], orient="index")
    ###         children.reset_index(inplace=True)
    ###         if len(children) > 0:
    ###             children.rename(columns={"image_path": "image_id"}, inplace=True)
    ###             children.drop(columns=["index", "parent_id"], inplace=True)
    ###             children["label"] = -1

    ###         parents = pd.DataFrame.from_dict(self.images["parent"], orient="index")
    ###         parents.reset_index(inplace=True)
    ###         if len(parents) > 0:
    ###             parents.rename(columns={"image_path": "image_id"}, inplace=True)
    ###             parents.drop(columns=["index", "parent_id"], inplace=True)
    ###             parents["label"] = -1

    ###         return parents, children
    ###     else:
    ###         raise ValueError(f"Format {fmt} is not supported!")
