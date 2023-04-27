try:
    from geopy.distance import geodesic, great_circle
except ImportError:
    pass

import rasterio
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd
from PIL import Image
from pylab import cm as pltcm
from pyproj import Transformer
import random
from typing import Literal, Optional, Union, Dict, Tuple, List, Any

from .slicers import patchifyByPixel

# from ..utils import geo_utils

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class mapImages:
    """
    Class to manage a collection of image paths and construct image objects.

    Parameters
    ----------
    path_images : str or None, optional
        Path to the directory containing images (accepts wildcards). By
        default, ``False``
    file_ext : str or False, optional
        The file extension of the image files to be loaded, ignored if file types are specified in ``path_images`` (e.g. with ``"./path/to/dir/*png"``).
        By default ``False``.
    tree_level : str, optional
        Level of the image hierarchy to construct. The value can be
        ``"parent"`` (default) and ``"patch"``.
    parent_path : str, optional
        Path to parent images (if applicable), by default ``None``.
    **kwds : dict, optional
        Additional keyword arguments to be passed to the ``imagesConstructor``
        method.

    Attributes
    ----------
    path_images : list
        List of paths to the image files.
    images : dict
        A dictionary containing the constructed image data. It has two levels
        of hierarchy, ``"parent"`` and ``"patch"``, depending on the value of
        the ``tree_level`` parameter.
    """

    def __init__(
        self,
        path_images: Optional[str] = None,
        file_ext: Optional[Union[str, bool]] = False,
        tree_level: Optional[str] = "parent",
        parent_path: Optional[str] = None,
        **kwds: Dict,
    ):
        """Initializes the mapImages class."""

        if path_images:
            self.path_images = self._resolve_file_path(path_images, file_ext)

        else:
            self.path_images = []

        # Create images variable (MAIN object variable)
        # New methods (e.g., reading/loading) should construct images this way
        self.images = {
            "parent": {},
            "patch": {},
        }
        for image_path in self.path_images:
            self.imagesConstructor(
                image_path=image_path,
                parent_path=parent_path,
                tree_level=tree_level,
                **kwds,
            )

    @staticmethod
    def _resolve_file_path(file_path, file_ext):
        if file_ext:
            if os.path.isdir(file_path):
                files = glob(os.path.abspath(f"{file_path}/*.{file_ext}"))
            else: #if not dir
                files = glob(os.path.abspath(file_path))
                files = [file for file in files if file.split(".")[-1] == file_ext] 

        else:
            if os.path.isdir(file_path):
                files = glob(os.path.abspath(f"{file_path}/*.*"))
            else:
                files = glob(os.path.abspath(file_path))

        #check for issues
        if len(files) == 0:
            raise ValueError("[ERROR] No files found!")
        test_ext = files[0].split(".")[-1]
        if not all(file.split(".")[-1] == test_ext for file in files):
            raise ValueError(
                "[ERROR] Directory with multiple file types detected - please specificy file extension (`patch_file_ext`) or, pass path to specific file types (wildcards accepted)."
            )
        
        return files

    def __len__(self) -> int:
        return int(len(self.images["parent"]) + len(self.images["patch"]))

    def __str__(self) -> Literal[""]:
        print(f"#images: {self.__len__()}")

        print(f"\n#parents: {len(self.images['parent'])}")
        for i, img in enumerate(self.images["parent"]):
            print(os.path.relpath(img))
            if i >= 10:
                print("...")
                break

        print(f"\n#patches: {len(self.images['patch'])}")
        for i, img in enumerate(self.images["patch"]):
            print(os.path.relpath(img))
            if i >= 10:
                print("...")
                break
        return ""

    def imagesConstructor(
        self,
        image_path: str,
        parent_path: Optional[str] = None,
        tree_level: Optional[str] = "patch",
        **kwds: Dict,
    ) -> None:
        """
        Constructs image data from the given image path and parent path and
        adds it to the ``mapImages`` instance's ``images`` attribute.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        parent_path : str, optional
            Path to the parent image (if applicable), by default ``None``.
        tree_level : str, optional
            Level of the image hierarchy to construct, either ``"patch"``
            (default) or ``"parent"``.
        **kwds : dict, optional
            Additional keyword arguments to be included in the constructed
            image data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``tree_level`` is not set to ``"parent"`` or ``"patch"``.

            If ``tree_level`` is set to ``"parent"`` and ``parent_path`` is
            not ``None``.

        Notes
        -----
        This method assumes that the ``images`` attribute has been initialized
        on the mapImages instance as a dictionary with two levels of hierarchy,
        ``"parent"`` and ``"patch"``. The image data is added to the
        corresponding level based on the value of ``tree_level``.
        """

        if tree_level not in ["parent", "patch"]:
            raise ValueError(
                f"[ERROR] tree_level can only be set to parent or patch, not: {tree_level}"  # noqa
            )

        if (parent_path is not None) and (tree_level == "parent"):
            raise ValueError(
                "[ERROR] if tree_level=parent, parent_path should be None."
            )

        # Convert the image_path to its absolute path
        image_path = os.path.abspath(image_path)
        image_id, _ = self.splitImagePath(image_path)

        self._check_image_mode(image_path)

        if parent_path:
            parent_path = os.path.abspath(parent_path)
            parent_basename, _ = self.splitImagePath(parent_path)
        else:
            parent_basename = None

        # --- Add other info to images
        self.images[tree_level][image_id] = {
            "parent_id": parent_basename,
            "image_path": image_path,
        }

        for k, v in kwds.items():
            self.images[tree_level][image_id][k] = v

        # --- Make sure parent exists in images["parent"]
        if tree_level == "patch" and parent_basename:
            # three possible scenarios
            # 1. parent_basename is not in the parent dictionary
            if parent_basename not in self.images["parent"].keys():
                self.images["parent"][parent_basename] = {
                    "parent_id": None,
                    "image_path": parent_path,
                }
            # 2. parent_basename exists but parent_id is not defined
            if "parent_id" not in self.images["parent"][parent_basename].keys():
                self.images["parent"][parent_basename]["parent_id"] = None
            # 3. parent_basename exists but image_path is not defined
            if "image_path" not in self.images["parent"][parent_basename].keys():
                self.images["parent"][parent_basename]["image_path"] = parent_path

    @staticmethod
    def _check_image_mode(image_path):
        try:
            img = Image.open(image_path)       
        except:
            raise ValueError(f"[ERROR] {image_path} is not an image file.")
       
        if img.mode not in ["1", "L", "LA", "I", "P", "RGB", "RGBA"]:
                raise NotImplementedError(f"[ERROR] Image mode '{img.mode}' not currently accepted.\n\n\
Please save your image(s) as one the following image modes: 1, L, LA, I, P, RGB or RGBA.\n\
See https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes for more information.")

    @staticmethod
    def splitImagePath(inp_path: str) -> Tuple[str, str]:
        """
        Split the input path into basename and dirname.

        Parameters
        ----------
        inp_path : str
            Input path to split.

        Returns
        -------
        tuple
            A tuple containing the basename and dirname of the input path.
        """
        inp_path = os.path.abspath(inp_path)
        path_basename = os.path.basename(inp_path)
        path_dirname = os.path.dirname(inp_path)
        return path_basename, path_dirname

    def add_metadata(
        self,
        metadata: Union[str, pd.DataFrame],
        index_col: Optional[Union[int, str]] = 0,
        delimiter: Optional[str] = "|",
        columns: Optional[List[str]] = None,
        tree_level: Optional[str] = "parent",
        ignore_mismatch: Optional[bool] = False,
    ) -> None:
        """
        Add metadata information to the images dictionary.

        Parameters
        ----------
        metadata : str or pandas.DataFrame
            A csv file path (normally created from a pandas DataFrame) or a
            pandas DataFrame that contains the metadata information.
        index_col : int or str, optional
            Column to use as the index when reading the csv file into a pandas.DataFrame.
            Accepts column indices or column names.
            By default ``0`` (first column).

            Only used if a csv file path is provided as
            the ``metadata`` parameter.
            Ingored if ``columns`` parameter is passed.
        delimiter : str, optional
            Delimiter to use for reading the csv file into a pandas DataFrame, by default ``"|"``.

            Only used if a csv file path is provided as
            the ``metadata`` parameter.
        columns : list, optional
            List of columns indices or names to add to mapImages. 
            If ``None`` is passed, all columns will be used. 
            By default ``None``.
        tree_level : str, optional
            Determines which images dictionary (``"parent"`` or ``"patch"``)
            to add the metadata to, by default ``"parent"``.
        ignore_mismatch : bool, optional
            Whether to error if metadata with mismatching information is passed. 
            By default ``False``.
        
        Raises
        ------
        ValueError
            If metadata is not a pandas DataFrame or a csv file path.

            If 'name' or 'image_id' is not one of the columns in the metadata.

        Returns
        -------
        None

        Notes
        ------
        Your metadata file must contain an column which contains the Image IDs (filenames) of your images.
        This should have a column name of either ``name`` or ``image_id``.
        """
        
        if isinstance(metadata, pd.DataFrame):
            if columns:
                metadata_df=metadata[columns].copy()
            else:
                metadata_df=metadata.copy()
                columns=list(metadata_df.columns)
        
        else: #if not df
            if not metadata.endswith('.csv') and os.path.isfile(f"{metadata}.csv"):
                metadata=f"{metadata}.csv"
            
            if os.path.isfile(metadata):
                if columns:
                    metadata_df = pd.read_csv(
                        metadata, usecols=columns, delimiter=delimiter
                        )
                else:
                    metadata_df = pd.read_csv(
                        metadata, index_col=index_col, delimiter=delimiter
                        )
                    columns=list(metadata_df.columns)

            else:
                raise ValueError(
                    "[ERROR] ``metadata`` should either be the path to a csv file or a pandas DataFrame."  # noqa
                    )

        #identify image_id column
        #what to do if "name" or "image_id" are index col?
        if index_col in ["name", "image_id"]:
            metadata_df[index_col]=metadata_df.index
            columns=list(metadata_df.columns)
        
        if "name" in columns:
            image_id_col = "name"
            if "image_id" in columns:
                print("[WARNING] Both 'name' and 'image_id' columns exist! Using 'name' as index"  # noqa
                    )
        elif "image_id" in columns:
            image_id_col = "image_id"
        else:
            raise ValueError(
                "[ERROR] 'name' or 'image_id' should be one of the columns."
            )
        
        if any(metadata_df.duplicated(subset=image_id_col)):
            print(
                "[WARNING] Duplicates found in metadata. Keeping only first instance of each duplicated value"
                )
            metadata_df.drop_duplicates(subset=image_id_col, inplace=True, keep="First")

        #look for non-intersecting values
        missing_metadata=set(self.images[tree_level].keys())-set(metadata_df[image_id_col])
        extra_metadata=set(metadata_df[image_id_col])-set(self.images[tree_level].keys())

        if not ignore_mismatch:
            if len(missing_metadata)!=0 and len(extra_metadata)!=0:
                raise ValueError(f"[ERROR] Metadata is missing information for: {[*missing_metadata]}. \n\
[ERROR] Metadata contains information about non-existant images: {[*extra_metadata]}"
                )
            elif len(missing_metadata)!=0: 
                raise ValueError(
                    f"[ERROR] Metadata is missing information for: {[*missing_metadata]}"
                )
            elif len(extra_metadata)!=0:
                raise ValueError(
                    f"[ERROR] Metadata contains information about non-existant images: {[*extra_metadata]}"
                )

        for key in self.images[tree_level].keys():
            if key in missing_metadata:
                continue
            else:
                data_series=metadata_df[metadata_df["name"]==key].squeeze()
                for column, item in data_series.items():
                    try:
                        self.images[tree_level][key][column]=eval(item)
                    except:
                        self.images[tree_level][key][column]=item

    def show_sample(
        self,
        num_samples: int,
        tree_level: Optional[str] = "parent",
        random_seed: Optional[int] = 65,
        **kwds: Dict,
    ) -> None:
        """
        Display a sample of images from a particular level in the image
        hierarchy.

        Parameters
        ----------
        num_samples : int
            The number of images to display.
        tree_level : str, optional
            The level of the hierarchy to display images from, which can be
            ``"patch"`` or ``"parent"`` (default).
        random_seed : int, optional
            The random seed to use for reproducibility. Default is ``65``.
        **kwds : dict, optional
            Additional keyword arguments to pass to
            ``matplotlib.pyplot.figure()``.

        Returns
        -------
        None
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

        for i, image_id in enumerate(sample_img_keys):
            plt.subplot(num_samples // 3 + 1, 3, i + 1)
            myimg = mpimg.imread(self.images[tree_level][image_id]["image_path"])
            plt.title(image_id, size=8)
            plt.imshow(myimg)
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()

    def list_parents(self) -> List[str]:
        """Return list of all parents"""
        return list(self.images["parent"].keys())

    def list_patches(self) -> List[str]:
        """Return list of all patches"""
        return list(self.images["patch"].keys())

    def add_shape(self, tree_level: Optional[str] = "parent") -> None:
        """
        Add a shape to each image in the specified level of the image
        hierarchy.

        Parameters
        ----------
        tree_level : str, optional
            The level of the hierarchy to add shapes to, either ``"parent"``
            (default) or ``"patch"``.

        Returns
        -------
        None

        Notes
        -----
        The method runs :meth:`mapreader.load.images.mapImages._add_shape_id`
        for each image present at the ``tree_level`` provided.
        """
        print(f"[INFO] Add shape, tree level: {tree_level}")

        list_items = list(self.images[tree_level].keys())
        for item in list_items:
            self._add_shape_id(image_id=item, tree_level=tree_level)

    def add_coord_increments(self) -> None:
        """
        Adds coordinate increments to each image at the parent level.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The method runs
        :meth:`mapreader.load.images.mapImages._add_coord_increments_id`
        for each image present at the parent level, which calculates
        pixel-wise delta longitute (``dlon``) and delta latititude (``dlat``)
        for the image and adds the data to it.
        """
        print("[INFO] Add coord-increments, tree level: parent")

        parent_list = self.list_parents()
        for parent_id in parent_list:
            if "coord" not in self.images["parent"][parent_id].keys():
                print(
                    f"[WARNING] No coordinates found for {parent_id}. Suggestion: run add_metadata or addGeoInfo"  # noqa
                )
                continue

            self._add_coord_increments_id(image_id=parent_id)

    def add_center_coord(self, tree_level: Optional[str] = "patch") -> None:
        """
        Adds center coordinates to each image at the specified tree level.

        Parameters
        ----------
        tree_level: str, optional
            The tree level where the center coordinates will be added. It can
            be either ``"parent"`` or ``"patch"`` (default).

        Returns
        -------
        None

        Notes
        -----
        The method runs
        :meth:`mapreader.load.images.mapImages._add_center_coord_id`
        for each image present at the ``tree_level`` provided, which calculates
        central longitude and latitude (``center_lon`` and ``center_lat``) for
        the image and adds the data to it.
        """
        print(f"[INFO] Add center coordinates, tree level: {tree_level}")

        list_items = list(self.images[tree_level].keys())
        par_id_list = []
        for item in list_items:
            if tree_level == "parent":
                if "coord" not in self.images[tree_level][item].keys():
                    print(
                        f"[WARNING] 'coord' could not be found in {item}. Suggestion: run add_metadata or addGeoInfo"  # noqa
                    )
                    continue

            if tree_level == "patch":
                par_id = self.images[tree_level][item]["parent_id"]

                if "coord" not in self.images["parent"][par_id].keys():
                    if par_id not in par_id_list:
                        print(
                            f"[WARNING] 'coord' could not be found in {par_id} so center coordinates cannot be calculated for it's patches. Suggestion: run add_metadata or addGeoInfo"  # noqa
                        )
                        par_id_list.append(par_id)
                    continue

            self._add_center_coord_id(image_id=item, tree_level=tree_level)

    def _add_shape_id(
        self, image_id: Union[int, str], tree_level: Optional[str] = "parent"
    ) -> None:
        """
        Add shape (image_height, image_width, image_channels) of the image
        with specified ``image_id`` in the given ``tree_level`` to the
        metadata.

        Parameters
        ----------
        image_id : int or str
            The ID of the image to add shape metadata to.
        tree_level : str, optional
            The tree level where the image is located, which can be
            ``"parent"`` (default) or ``"patch"``.

        Returns
        -------
        None
            This method does not return anything. It modifies the metadata of
            the ``images`` property in-place.

        Notes
        -----
        The shape of the image is obtained by loading the image from its
        ``image_path`` value and getting its shape.
        """
        myimg = mpimg.imread(self.images[tree_level][image_id]["image_path"])
        # shape = (hwc)
        myimg_shape = myimg.shape
        self.images[tree_level][image_id]["shape"] = myimg_shape

    def _add_coord_increments_id(
        self, image_id: Union[int, str], verbose: Optional[bool] = False
    ) -> None:
        """
        Add pixel-wise delta longitute (``dlon``) and delta latititude
        (``dlat``) to the metadata of the image with the specified ``image_id``
        in the parent tree level.

        Parameters
        ----------
        image_id : int or str
            The ID of the image to add coordinate increments metadata to.
        verbose : bool, optional
            Whether to print warning messages when coordinate or shape
            metadata cannot be found. Default is ``False``.

        Returns
        -------
        None
            This method does not return anything. It modifies the metadata of
            the image in-place.

        Notes
        -----
        Coordinate increments (dlon and dlat) are calculated using the
        following formula:

        .. code-block:: python

            dlon = abs(lon_max - lon_min) / image_width
            dlat = abs(lat_max - lat_min) / image_height

        ``lon_max``, ``lon_min``, ``lat_max``, ``lat_min`` are the coordinate
        bounds of the image, and ``image_width`` and ``image_height`` are the
        width and height of the image in pixels respectively.

        This method assumes that the coordinate and shape metadata of the
        image have already been added to the metadata.

        If the coordinate metadata cannot be found, a warning message will be
        printed if ``verbose=True``.

        If the shape metadata cannot be found, this method will call the
        :meth:`mapreader.load.images.mapImages._add_shape_id` method to add
        it.
        """
        # Check for warnings
        if "coord" not in self.images["parent"][image_id].keys():
            if verbose:
                print(
                    f"[WARNING]'coord' could not be found in {image_id}. Suggestion: run add_metadata or addGeoInfo"  # noqa
                )
            return

        # Add shapes if non-existent
        if "shape" not in self.images["parent"][image_id].keys():
            self._add_shape_id(image_id)

        # Extract height/width/chan from shape
        image_height, image_width, _ = self.images["parent"][image_id]["shape"]

        # Extract coordinates from image
        lon_min, lon_max, lat_min, lat_max = self.images["parent"][image_id]["coord"]

        # Calculate dlon and dlat
        dlon = abs(lon_max - lon_min) / image_width
        dlat = abs(lat_max - lat_min) / image_height

        # Assign values
        self.images["parent"][image_id]["dlon"] = dlon
        self.images["parent"][image_id]["dlat"] = dlat

    def _add_center_coord_id(
        self,
        image_id: Union[int, str],
        tree_level: Optional[str] = "patch",
        verbose: Optional[bool] = False,
    ) -> None:
        """
        Calculates and adds center coordinates (longitude as ``center_lon``
        and latitude as ``center_lat``) to a given patch.

        Parameters
        ----------
        image_id : int or str
            The ID of the patch to add center coordinates to.
        tree_level : str, optional
            The level of the patch in the image hierarchy, either
            ``"parent"`` or ``"patch"`` (default).
        verbose : bool, optional
            Whether to print warning messages or not. Defaults to ``False``.

        Raises
        ------
        NotImplementedError
            If ``tree_level`` is not set to ``"parent"`` or ``"patch"``.

        Returns
        -------
        None
        """
        if tree_level == "patch":
            par_id = self.images[tree_level][image_id]["parent_id"]

            if ("dlon" not in self.images["parent"][par_id].keys()) or (
                "dlat" not in self.images["parent"][par_id].keys()
            ):
                if "coord" not in self.images["parent"][par_id].keys():
                    if verbose:
                        print(
                            f"[WARNING] No coordinates found for {image_id}. Suggestion: run add_metadata or addGeoInfo"  # noqa
                        )
                    return

                else:
                    self._add_coord_increments_id(par_id)

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

        elif tree_level == "parent":
            if "coord" not in self.images[tree_level][image_id].keys():
                if verbose:
                    print(
                        f"[WARNING] No coordinates found for {image_id}. Suggestion: run add_metadata or addGeoInfo"  # noqa
                    )
                return

            print(f"[INFO] Reading 'coord' from {image_id}")
            lon_min, lon_max, lat_min, lat_max = self.images[tree_level][image_id][
                "coord"
            ]
            self.images[tree_level][image_id]["center_lon"] = (lon_min + lon_max) / 2.0
            self.images[tree_level][image_id]["center_lat"] = (lat_min + lat_max) / 2.0
        else:
            raise NotImplementedError("Tree level must be set to 'parent' or 'patch'.")

    def calc_pixel_width_height(
        self,
        parent_id: Union[int, str],
        calc_size_in_m: Optional[str] = "great-circle",
        verbose: Optional[bool] = False,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate the width and height of each pixel in a given image in
        meters.

        Parameters
        ----------
        parent_id : int or str
            The ID of the parent image to calculate pixel size.
        calc_size_in_m : str, optional
            Method to use for calculating image size in meters.
            Possible values: ``"great-circle"`` (default), ``"gc"`` (alias for
            ``"great-circle"``), ``"geodesic"``. ``"great-circle"`` and
            ``"gc"`` compute size using the great-circle distance formula,
            while ``"geodesic"`` computes size using the geodesic distance
            formula.
        verbose : bool, optional
            If ``True``, print additional information during the calculation.
            Default is ``False``.

        Returns
        -------
        tuple of floats
            The size of the image in meters as a tuple of bottom, top, left,
            and right distances (in that order).

        Notes
        -----
        This method requires the parent image to have location metadata added
        with either the :meth:`mapreader.load.images.mapImages.add_metadata`
        or :meth:`mapreader.load.images.mapImages.addGeoInfo` methods.

        The calculations are performed using the ``geopy.distance.geodesic``
        and ``geopy.distance.great_circle`` methods. Thus, the method requires
        the ``geopy`` package to be installed.
        """

        if "coord" not in self.images["parent"][parent_id].keys():
            print(
                f"[WARNING] 'coord' could not be found in {parent_id}. Suggestion: run add_metadata or addGeoInfo"  # noqa
            )
            return

        myimg = mpimg.imread(self.images["parent"][parent_id]["image_path"])
        image_height, image_width, _ = myimg_shape = myimg.shape

        (xmin, xmax, ymin, ymax) = self.images["parent"][parent_id]["coord"]
        if verbose:
            print("[INFO] Using coordinates to compute width/height:")
            print(f"[INFO] lon min/max: {xmin:.4f}/{xmax:.4f}")
            print(f"[INFO] lat min/max: {ymin:.4f}/{ymax:.4f}")
            print(f"[INFO] shape (hwc): {myimg_shape}")

        # Calculate the size of image in meters
        if calc_size_in_m == "geodesic":
            bottom = geodesic((ymin, xmin), (ymin, xmax)).meters
            right = geodesic((ymin, xmax), (ymax, xmax)).meters
            top = geodesic((ymax, xmax), (ymax, xmin)).meters
            left = geodesic((ymax, xmin), (ymin, xmin)).meters
            size_in_m = (bottom, top, left, right)
            if verbose:
                print(
                    f"[INFO] size (in meters) bottom/top/left/right: {bottom:.2f}/{top:.2f}/{left:.2f}/{right:.2f}"  # noqa
                )

            mean_width = np.mean(
                [size_in_m[0] / image_width, size_in_m[1] / image_width]
            )
            mean_height = np.mean(
                [size_in_m[2] / image_height, size_in_m[3] / image_height]
            )
            if verbose:
                print(
                    f"Each pixel is ~{mean_width:.3f} X {mean_height:.3f} meters (width x height)."  # noqa
                )

        elif calc_size_in_m in ["gc", "great-circle"]:
            bottom = great_circle((ymin, xmin), (ymin, xmax)).meters
            right = great_circle((ymin, xmax), (ymax, xmax)).meters
            top = great_circle((ymax, xmax), (ymax, xmin)).meters
            left = great_circle((ymax, xmin), (ymin, xmin)).meters
            size_in_m = (bottom, top, left, right)
            if verbose:
                print(
                    f"[INFO] size (in meters) bottom/top/left/right: {bottom:.2f}/{top:.2f}/{left:.2f}/{right:.2f}"  # noqa
                )

            mean_width = np.mean(
                [size_in_m[0] / image_width, size_in_m[1] / image_width]
            )
            mean_height = np.mean(
                [size_in_m[2] / image_height, size_in_m[3] / image_height]
            )
            if verbose:
                print(
                    f"Each pixel is ~{mean_width:.3f} x {mean_height:.3f} meters (width x height)."  # noqa
                )

        return size_in_m

    def patchifyAll(
        self,
        method: Optional[str] = "pixel",
        patch_size: Optional[int] = 100,
        path_save: Optional[str] = "patches",
        square_cuts: Optional[bool] = False,
        resize_factor: Optional[bool] = False,
        output_format: Optional[str] = "png",
        rewrite: Optional[bool] = False,
        verbose: Optional[bool] = False,
        tree_level: Optional[str] = "parent",
        add_to_parent: Optional[bool] = True,
        id1: Optional[int] = 0,
        id2: Optional[int] = -1,
    ) -> None:
        """
        Patchify images in the specified ``tree_level`` and add the patches to the mapImages instance's ``images`` dictionary.

        Parameters
        ----------
        method : str, optional
            Method used to patchify images, choices between ``"pixel"`` (default)
            and ``"meters"`` or ``"meter"``.
        patch_size : int, optional
            Number of pixels/meters in both x and y to use for slicing, by
            default ``100``.
        path_save : str, optional
            Directory to save the patches, by default
            ``"patches"``.
        square_cuts : bool, optional
            If True, all patches will have the same number of pixels in
            x and y, by default ``False``.
        resize_factor : bool, optional
            If True, resize the images before patchifying, by default ``False``.
        output_format : str, optional
            Format to use when writing image files, by default ``"png"``.
        rewrite : bool, optional
            If True, existing patches will be rewritten, by default ``False``.
        verbose : bool, optional
            If True, progress updates will be printed throughout, by default
            ``False``.
        tree_level : str, optional
            Tree level, choices between ``"parent"`` or ``"patch``, by default
            ``"parent"``.
        add_to_parent : bool, optional
            If True, patches will be added to the mapImages instance's
            ``images`` dictionary, by default ``True``.
        id1 : int, optional
            The start index of the images to patchify. Default is ``0``.
        id2 : int, optional
            The end index of the images to patchify. Default is ``-1`` (i.e., all
            images after index ``id1`` will be patchified).

        Raises
        ------
        ValueError
            If ``id2 < id1``.

        Returns
        -------
        None
        """
        if id2 < 0:
            img_keys = list(self.images[tree_level].keys())[id1:]
        elif id2 < id1:
            raise ValueError("id2 should be > id1.")
        else:
            img_keys = list(self.images[tree_level].keys())[id1:id2]

        for image_id in img_keys:
            image_path = self.images[tree_level][image_id]["image_path"]

            patches_info = self._patchify(
                image_path=image_path,
                method=method,
                patch_size=patch_size,
                path_save=path_save,
                square_cuts=square_cuts,
                resize_factor=resize_factor,
                output_format=output_format,
                rewrite=rewrite,
                verbose=verbose,
                image_id=image_id,
                tree_level=tree_level,
            )

            if add_to_parent:
                for i in range(len(patches_info)):
                    # Add patches to the .images["patch"]
                    self.imagesConstructor(
                        image_path=patches_info[i][0],
                        parent_path=image_path,
                        tree_level="patch",
                        min_x=patches_info[i][1][0],
                        min_y=patches_info[i][1][1],
                        max_x=patches_info[i][1][2],
                        max_y=patches_info[i][1][3],
                    )

        if add_to_parent:
            # add patches to the parent dictionary
            self.addPatches()

    def _patchify(
        self,
        image_path: str,
        method: Optional[str] = "pixel",
        patch_size: Optional[int] = 100,
        path_save: Optional[str] = "patches",
        square_cuts: Optional[bool] = False,
        resize_factor: Optional[bool] = False,
        output_format: Optional[str] = "png",
        rewrite: Optional[bool] = False,
        verbose: Optional[bool] = True,
        image_id: Optional[Union[int, str]] = None,
        tree_level: Optional[str] = None,
    ) -> List:
        """
        Patchify one image

        ..
            Private method.

        Parameters
        ----------
        image_path : str
            Path to image.
        method : str, optional
            Method used to slice images, choices between ``"pixel"`` and
            ``"meters"`` or ``"meter"``, by default ``"pixel"``.
        patch_size : int, optional
            Number of pixels/meters in both x and y to use for patchifying, by
            default ``100``.
        path_save : str, optional
            Directory to save the patches, by default
            ``"patches"``.
        square_cuts : bool, optional
            If ``True``, all patches will have the same number of pixels
            in x and y, by default ``False``.
        resize_factor : bool, optional
            If ``True``, resize the images before patchifying, by
            ``False``.
        output_format : str, optional
            Format to use when writing image files, by default
            ``"png"``.
        rewrite : bool, optional
            If ``True``, existing slices will be rewritten, by default
            ``False``.
        verbose : bool, optional
            If ``True``, progress updates will be printed throughout, by
            default ``False``.
        tree_level : str, optional
            Tree level, choices between ``"parent"`` or ``"patch"``, by
            default ``"parent"``.

        Returns
        -------
        list
            patches_info
        """

        print(40 * "=")
        print(f"Patchifying {os.path.relpath(image_path)}")
        print(40 * "-")

        # make sure the dir exists
        self._makeDir(path_save)

        # which image should be sliced
        image_path = os.path.abspath(image_path)
        patches_info = None
        if method == "pixel":
            patches_info = patchifyByPixel(
                image_path=image_path,
                patch_size=patch_size,
                path_save=path_save,
                square_cuts=square_cuts,
                resize_factor=resize_factor,
                output_format=output_format,
                rewrite=rewrite,
                verbose=verbose,
            )

        elif method in ["meters", "meter"]:
            keys = self.images[tree_level][image_id].keys()

            if "coord" not in keys:
                raise ValueError(
                    "Please add coordinate information first. Suggestion: Run add_metadata or addGeoInfo"  # noqa
                )

            if "shape" not in keys:
                self._add_shape_id(image_id=image_id, tree_level=tree_level)

            image_height, _, _ = self.images[tree_level][image_id]["shape"]

            # size in meter contains: (bottom, top, left, right)
            size_in_m = self.calc_pixel_width_height(image_id)

            # pixel height in m per pixel
            pixel_height = size_in_m[2] / image_height
            number_pixels4slice = int(patch_size / pixel_height)

            patches_info = patchifyByPixel(
                image_path=image_path,
                patch_size=number_pixels4slice,
                path_save=path_save,
                square_cuts=square_cuts,
                resize_factor=resize_factor,
                output_format=output_format,
                rewrite=rewrite,
                verbose=verbose,
            )

        return patches_info

    def addPatches(self) -> None:
        """
        Add patches to parent.

        Returns
        -------
        None

        Notes
        -----
        This method adds patches to their corresponding parent image. It
        checks if the parent image has any patches, and if not, it creates
        a list of patches and assigns it to the parent. If the parent image
        already has a list of patches, the method checks if the current patch
        is already in the list. If not, the patch is added to the list.
        """
        for patch in self.images["patch"].keys():
            my_parent = self.images["patch"][patch]["parent_id"]
            if not self.images["parent"][my_parent].get("patches", False):
                self.images["parent"][my_parent]["patches"] = [patch]
            else:
                if patch not in self.images["parent"][my_parent]["patches"]:
                    self.images["parent"][my_parent]["patches"].append(patch)

    def _makeDir(self, path_make: str, exists_ok: Optional[bool] = True) -> None:
        """
        Helper method to make directories.

        ..
            Private method.
        """
        os.makedirs(path_make, exist_ok=exists_ok)

    def calc_pixel_stats(
        self,
        parent_id: Optional[Union[str, int]] = None,
        calc_mean: Optional[bool] = True,
        calc_std: Optional[bool] = True,
    ) -> None:
        """
        Calculate the mean and standard deviation of pixel values for all
        channels (R, G, B, RGB and, if present, Alpha) of all patches of
        a given parent image. Store the results in the mapImages instance's
        ``images`` dictionary.

        Parameters
        ----------
        parent_id : str, int, or None, optional
            The ID of the parent image to calculate pixel stats for. If
            ``None``, calculate pixel stats for all parent images.
        calc_mean : bool, optional
            Whether to calculate mean pixel values. Default is ``True``.
        calc_std : bool, optional
            Whether to calculate standard deviation of pixel values. Default
            is ``True``.

        Returns
        -------
        None

        Notes
        -----
        - Pixel stats are calculated for patches of the parent image
          specified by ``parent_id``.
        - If ``parent_id`` is ``None``, pixel stats are calculated for all
          parent images in the object.
        - If mean or standard deviation of pixel values has already been
          calculated for a patch, the calculation is skipped.
        - Pixel stats are stored in the ``images`` attribute of the
          ``mapImages`` instance, under the ``patch`` key for each patch.
        - If no patches are found for a parent image, a warning message is
          displayed and the method moves on to the next parent image.
        """
        # Get correct parent ID
        if parent_id is None:
            parent_ids = self.list_parents()
        else:
            parent_ids = [parent_id]

        for parent_id in parent_ids:
            print(10 * "-")
            print(f"[INFO] calculate pixel stats for image: {parent_id}")

            if "patches" not in self.images["parent"][parent_id]:
                print(f"[WARNING] No patches found for: {parent_id}")
                continue

            list_patches = self.images["parent"][parent_id]["patches"]

            for patch in list_patches:
                patch_data = self.images["patch"][patch]

                # Check whether calculation has already been run
                patch_keys = patch_data.keys()
                if all(
                    [
                        "mean_pixel_RGB" in patch_keys,
                        "std_pixel_RGB" in patch_keys,
                    ]
                ):
                    continue

                # Load image
                patch_img = mpimg.imread(patch_data["image_path"])

                if calc_mean:
                    # Calculate mean pixel values
                    self.images["patch"][patch]["mean_pixel_R"] = np.mean(
                        patch_img[:, :, 0]
                    )
                    self.images["patch"][patch]["mean_pixel_G"] = np.mean(
                        patch_img[:, :, 1]
                    )
                    self.images["patch"][patch]["mean_pixel_B"] = np.mean(
                        patch_img[:, :, 2]
                    )
                    self.images["patch"][patch]["mean_pixel_RGB"] = np.mean(
                        patch_img[:, :, 0:3]
                    )
                    # check whether alpha is present
                    if patch_img.shape[2] > 3:
                        self.images["patch"][patch]["mean_pixel_A"] = np.mean(
                            patch_img[:, :, 3]
                        )
                if calc_std:
                    # Calculate standard deviation for pixel values
                    self.images["patch"][patch]["std_pixel_R"] = np.std(
                        patch_img[:, :, 0]
                    )
                    self.images["patch"][patch]["std_pixel_G"] = np.std(
                        patch_img[:, :, 1]
                    )
                    self.images["patch"][patch]["std_pixel_B"] = np.std(
                        patch_img[:, :, 2]
                    )
                    self.images["patch"][patch]["std_pixel_RGB"] = np.std(
                        patch_img[:, :, 0:3]
                    )
                    # check whether alpha is present
                    if patch_img.shape[2] > 3:
                        self.images["patch"][patch]["std_pixel_A"] = np.std(
                            patch_img[:, :, 3]
                        )

    def convertImages(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert the ``mapImages`` instance's ``images`` dictionary into pandas
        DataFrames for easy manipulation.

        Returns
        -------
        tuple of two pandas DataFrames
            The method returns a tuple of two DataFrames: One for the
            ``parent`` images and one for the ``patch`` images.
        """
        parents = pd.DataFrame.from_dict(self.images["parent"], orient="index")
        patches = pd.DataFrame.from_dict(self.images["patch"], orient="index")

        return parents, patches

    def show_par(
        self,
        parent_id: Union[int, str],
        value: Optional[Union[List[str], bool]] = False,
        **kwds,
    ) -> None:
        """
        A wrapper method for `.show()` which plots all patches of a
        specified parent (`parent_id`).

        Parameters
        ----------
        parent_id : int or str
            ID of the parent image to be plotted.
        value : list or bool, optional
            Value to be plotted on each patch, by default False.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the parent_id is not found in the image dictionary.

        Notes
        -----
        This is a wrapper method. See the documentation of the
        :meth:`mapreader.load.images.mapImages.show` method for more detail.
        """
        image_ids = self.images["parent"][parent_id]["patches"]
        self.show(image_ids, value=value, **kwds)

    def show(
        self,
        image_ids: Union[str, List[str]],
        value: Optional[Union[str, List[str], bool]] = False,
        plot_parent: Optional[bool] = True,
        border: Optional[bool] = True,
        border_color: Optional[str] = "r",
        vmin: Optional[Union[float, List[float]]] = 0.5,
        vmax: Optional[Union[float, List[float]]] = 2.5,
        colorbar: Optional[Union[str, List[str]]] = "viridis",
        alpha: Optional[Union[float, List[float]]] = 1.0,
        discrete_colorbar: Optional[Union[int, List[int]]] = 256,
        tree_level: Optional[str] = "patch",
        grid_plot: Optional[Tuple[int, int]] = (20000, 20000),
        plot_histogram: Optional[bool] = True,
        save_kml_dir: Optional[Union[bool, str]] = False,
        image_width_resolution: Optional[int] = None,
        kml_dpi_image: Optional[int] = None,
        **kwds: Dict,
    ) -> None:
        """
        Plot images from a list of `image_ids`.

        Parameters
        ----------
        image_ids : str or list
            Image ID or list of image IDs to be plotted.
        value : str, list or bool, optional
            Value to plot on patches, by default ``False``.
        plot_parent : bool, optional
            If ``True``, parent image will be plotted in background, by
            default ``True``.
        border : bool, optional
            If ``True``, a border will be placed around each patch, by
            default ``True``.
        border_color : str, optional
            The color of the border. Default is ``"r"``.
        vmin : float or list, optional
            The minimum value for the colormap. By default ``0.5``.

            If a list is provided, it must be the same length as ``image_ids``.
        vmax : float or list, optional
            The maximum value for the colormap. By default ``2.5``.

            If a list is provided, it must be the same length as ``image_ids``.
        colorbar : str or list, optional
            Colorbar used to visualise chosen ``value``, by default
            ``"viridis"``.

            If a list is provided, it must be the same length as ``image_ids``.
        alpha : float or list, optional
            Transparency level for plotting ``value`` with floating point
            values ranging from 0.0 (transparent) to 1 (opaque). By default,
            ``1.0``.

            If a list is provided, it must be the same length as ``image_ids``.
        discrete_colorbar : int or list, optional
            Number of discrete colurs to use in colorbar, by default ``256``.

            If a list is provided, it must be the same length as ``image_ids``.
        tree_level : str, optional
            The level of the image tree to be plotted. Must be either
            ``"patch"`` (default) or ``"parent"``.
        grid_plot : tuple, optional
            The size of the grid (number of rows and columns) to be used to
            plot images. Later adjusted to the true min/max of all subplots.
            By default ``(20000, 20000)``.
        plot_histogram : bool, optional
            If ``True``, plot histograms of the ``value`` of images. By
            default ``True``.
        save_kml_dir : str or bool, optional
            If ``True``, save KML files of the images. If a string is provided,
            it is the path to the directory in which to save the KML files. If
            set to ``False``, no files are saved. By default ``False``.
        image_width_resolution : int or None, optional
            The pixel width to be used for plotting. If ``None``, the
            resolution is not changed. Default is ``None``.

            Note: Only relevant when ``tree_level="parent"``.
        kml_dpi_image : int or None, optional
            The resolution, in dots per inch, to create KML images when
            ``save_kml_dir`` is specified (as either ``True`` or with path).
            By default ``None``.

        Returns
        -------
        None
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

                # Change the resolution of the image if image_width_resolution
                # is specified
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
                    if "coord" not in self.images["parent"][one_image_id].keys():
                        print(
                            "[WARNING] 'coord' could not be found. This is needed when save_kml_dir is set...continue"  # noqa
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

        elif tree_level == "patch":
            # Collect parents information
            parents = {}
            for i in range(len(image_ids)):
                try:
                    parent_id = self.images[tree_level][image_ids[i]]["parent_id"]
                except Exception as err:
                    print(err)
                    continue

                if parent_id not in parents:
                    parents[parent_id] = {
                        "path": self.images["parent"][parent_id]["image_path"],
                        "patch": [],
                    }

                parents[parent_id]["patch"].append(image_ids[i])

            for i in parents.keys():
                figsize = self._get_kwds(kwds, "figsize")
                plt.figure(figsize=figsize)

                for i_value, value in enumerate(values):
                    # initialize image2plot
                    # will be filled with values of 'value'
                    image2plot = np.empty(grid_plot)
                    image2plot[:] = np.nan
                    min_x, min_y, max_x, max_y = (
                        grid_plot[1],
                        grid_plot[0],
                        0,
                        0,
                    )
                    for patch_id in parents[i]["patch"]:
                        one_image = self.images[tree_level][patch_id]

                        # Set the values for each patch
                        if not value:
                            pass

                        elif value == "const":
                            # Assign values to image2plot, update min_x,
                            # min_y, ...
                            image2plot[
                                one_image["min_y"] : one_image["max_y"],
                                one_image["min_x"] : one_image["max_x"],
                            ] = 1.0

                        elif value == "random":
                            import random

                            # assign values to image2plot, update min_x,
                            # min_y, ...
                            image2plot[
                                one_image["min_y"] : one_image["max_y"],
                                one_image["min_x"] : one_image["max_x"],
                            ] = random.random()

                        elif value:
                            if value not in one_image:
                                assign_value = None
                            else:
                                assign_value = one_image[value]
                            # assign values to image2plot, update min_x,
                            # min_y, ...
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

                        # Adjust image2plot to global min/max in x and y
                        # directions
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

    def _createKML(
        self,
        path2kml: str,
        value: str,
        coords: Union[List, Tuple],
        counter: Optional[int] = -1,
    ) -> None:
        """Create a KML file.

        ..
            Private method.

        Parameters
        ----------
        path2kml : str
            Directory to save KML file.
        value : _type_
            Value to be plotted on the underlying image.
            See `.show()` for detail.
        coords : list or tuple
            Coordinates of the bounding box.
        counter : int, optional
            Counter to be used for HREF, by default `-1`.
        """

        try:
            import simplekml
        except ImportError:
            raise ImportError("[ERROR] simplekml is needed to create KML outputs.")

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

    def _plotBorder(
        self,
        image_dict: Dict,
        plt: plt,
        linewidth: Optional[int] = 0.5,
        zorder: Optional[int] = 20,
        color: Optional[str] = "r",
    ) -> None:
        """Plot border for an image

        ..
            Private method.

        Arguments:
            image_dict : dict
                image dictionary, e.g., one item in ``self.images["patch"]``
            plt : matplotlib.pyplot object
                a matplotlib.pyplot object

        Keyword Arguments:
            linewidth : int
                line-width (default: ``2``)
            zorder : int
                z-order for the border (default: ``5``)
            color : str
                color of the border (default: ``"r"``)
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
    def _get_kwds(
        kwds: Dict, key: str
    ) -> Union[Tuple[int, int], int, List[Union[int, float]], Any]:
        """
        If ``kwds`` dictionary has the ``key``, return value; otherwise, use
        default for ``key`` provided.

        ..
            Private method.
        """
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
        self,
        patch_paths: str,
        patch_file_ext: Optional[Union[str, bool]] = False,
        parent_paths: Optional[Union[str, bool]] = False,
        parent_file_ext: Optional[Union[str, bool]] = False,
        add_geo_par: Optional[bool] = False,
        clear_images: Optional[bool] = False,
    ) -> None:
        """
        Loads patch images from the given paths and adds them to the ``images``
        dictionary in the ``mapImages`` instance.

        Parameters
        ----------
        patch_paths : str
            The file path of the patches to be loaded.

            *Note: The ``patch_paths`` parameter accepts wildcards.*
        patch_file_ext : str or bool, optional
            The file extension of the patches to be loaded, ignored if file extensions are specified in ``patch_paths`` (e.g. with ``"./path/to/dir/*png"``)
            By default ``False``.
        parent_paths : str or bool, optional
            The file path of the parent images to be loaded. If set to
            ``False``, no parents are loaded. Default is ``False``.

            *Note: The ``parent_paths`` parameter accepts wildcards.*
        parent_file_ext : str or bool, optional
            The file extension of the parent images, ignored if file extensions are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
            By default ``False``.
        add_geo_par : bool, optional
            If ``True``, adds geographic information to the parent image.
            Default is ``False``.
        clear_images : bool, optional
            If ``True``, clears the images from the ``images`` dictionary
            before loading. Default is ``False``.

        Returns
        -------
        None
        """
        files = self._resolve_file_path(patch_paths, patch_file_ext)

        if clear_images:
            self.images = {}
            self.images["parent"] = {}
            self.images["patch"] = {}

        for file in files:
            if not os.path.isfile(file):
                print(f"[WARNING] File does not exist: {file}")
                continue

            self._check_image_mode(file)

            # patch ID is set to the basename
            patch_id = os.path.basename(file)

            # Parent ID and border can be detected using patch_id
            parent_id = self.detectParIDfromPath(patch_id)
            min_x, min_y, max_x, max_y = self.detectBorderFromPath(patch_id)

            # Add patch
            if not self.images["patch"].get(patch_id, False):
                self.images["patch"][patch_id] = {}
            self.images["patch"][patch_id]["parent_id"] = parent_id
            self.images["patch"][patch_id]["image_path"] = file
            self.images["patch"][patch_id]["min_x"] = min_x
            self.images["patch"][patch_id]["min_y"] = min_y
            self.images["patch"][patch_id]["max_x"] = max_x
            self.images["patch"][patch_id]["max_y"] = max_y

        if parent_paths:
            # Add parents
            self.loadParents(
                parent_paths=parent_paths,
                parent_file_ext=parent_file_ext,
                update=False,
                add_geo=add_geo_par,
            )
            # Add patches to the parent
            self.addPatches()

    @staticmethod
    def detectParIDfromPath(
        image_id: Union[int, str], parent_delimiter: Optional[str] = "#"
    ) -> str:
        """
        Detect parent IDs from ``image_id``.

        Parameters
        ----------
        image_id : int or str
            ID of patch.
        parent_delimiter : str, optional
            Delimiter used to separate parent ID when naming patch, by
            default ``"#"``.

        Returns
        -------
        str
            Parent ID.
        """
        return image_id.split(parent_delimiter)[1]

    @staticmethod
    def detectBorderFromPath(
        image_id: Union[int, str],
        # border_delimiter="-" # <-- not in use in this method
    ) -> Tuple[int, int, int, int]:
        """
        Detects borders from the path assuming patch is named using the
        following format: ``...-min_x-min_y-max_x-max_y-...``

        Parameters
        ----------
        image_id : int or str
            ID of image

        ..
            border_delimiter : str, optional
                Delimiter used to separate border values when naming patch
                image, by default ``"-"``.

        Returns
        -------
        tuple of min_x, min_y, max_x, max_y
            Border (min_x, min_y, max_x, max_y) of image
        """

        split_path = image_id.split("-")
        return (
            int(split_path[1]),
            int(split_path[2]),
            int(split_path[3]),
            int(split_path[4]),
        )

    def loadParents(
        self,
        parent_paths: Optional[Union[str, bool]] = False,
        parent_ids: Optional[Union[List[str], str, bool]] = False,
        parent_file_ext: Optional[Union[str, bool]] = False,
        update: Optional[bool] = False,
        add_geo: Optional[bool] = False,
    ) -> None:
        """
        Load parent images from file paths (``parent_paths``).

        If ``parent_paths`` is not given, only ``parent_ids``, no image path
        will be added to the images.

        Parameters
        ----------
        parent_paths : str or bool, optional
            Path to parent images, by default ``False``.
        parent_ids : list, str or bool, optional
            ID(s) of parent images. Ignored if ``parent_paths`` are specified.
            By default ``False``.
        parent_file_ext : str or bool, optional
            The file extension of the parent images, ignored if file extensions are specified in ``parent_paths`` (e.g. with ``"./path/to/dir/*png"``)
            By default ``False``.
        update : bool, optional
            If ``True``, current parents will be overwritten, by default
            ``False``.
        add_geo : bool, optional
            If ``True``, geographical info will be added to parents, by
            default ``False``.

        Returns
        -------
        None
        """
        
        if parent_paths:
            files = self._resolve_file_path(parent_paths, parent_file_ext)

            if update:
                self.images["parent"] = {}

            for file in files:

                self._check_image_mode(file)

                parent_id = os.path.basename(file)
                self.images["parent"][parent_id] = {"parent_id": None}
                if os.path.isfile(file):
                    self.images["parent"][parent_id]["image_path"] = os.path.abspath(
                        file
                    )
                else:
                    self.images["parent"][parent_id]["image_path"] = None

                if add_geo:
                    self.addGeoInfo()

        elif parent_ids:
            if not isinstance(parent_ids, list):
                parent_ids = [parent_ids]
            for parent_id in parent_ids:
                self.images["parent"][parent_id] = {"parent_id": None}
                self.images["parent"][parent_id]["image_path"] = None

    def loadDataframe(
        self,
        parents: Optional[Union[pd.DataFrame, str]] = None,
        patch_df: Optional[Union[pd.DataFrame, str]] = None,
        clear_images: Optional[bool] = True,
    ) -> None:
        """
        Form images variable from pandas DataFrame(s).

        Parameters
        ----------
        parents : pandas.DataFrame, str or None, optional
            DataFrame containing parents or path to parents, by default
            ``None``.
        patch_df : pandas.DataFrame or None, optional
            DataFrame containing patches, by default ``None``.
        clear_images : bool, optional
            If ``True``, clear images before reading the dataframes, by
            default ``True``.

        Returns
        -------
        None
        """

        if clear_images:
            self.images = {"parent": {}, "patch": {}}

        if not isinstance(patch_df, type(None)):
            self.images["patch"] = patch_df.to_dict(orient="index")

        if not isinstance(parents, type(None)):
            if isinstance(parents, str):
                self.loadParents(parents)
            else:
                self.images["parent"] = parents.to_dict(orient="index")

            for parent_id in self.images["parent"].keys():
                # Do we need this?
                # k2change = "patches"
                # if k2change in self.images["parent"][parent_id]:
                #    try:
                #        self.images["parent"][parent_id][k2change] = self.images["parent"][parent_id][k2change]  # noqa
                #    except Exception as err:
                #        print(err)

                k2change = "coord"
                if k2change in self.images["parent"][parent_id]:
                    try:
                        self.images["parent"][parent_id][k2change] = self.images[
                            "parent"
                        ][parent_id][k2change]
                    except Exception as err:
                        print(err)

            self.addPatches()

    def load_csv_file(
        self,
        parent_path: Optional[str] = None,
        patch_path: Optional[str] = None,
        clear_images: Optional[bool] = False,
        index_col_patch: Optional[int] = 0,
        index_col_parent: Optional[int] = 0,
    ) -> None:
        """
        Load CSV files containing information about parent and patches,
        and update the ``images`` attribute of the ``mapImages`` instance with
        the loaded data.

        Parameters
        ----------
        parent_path : str, optional
            Path to the CSV file containing parent image information.
        patch_path : str, optional
            Path to the CSV file containing patch information.
        clear_images : bool, optional
            If True, clear all previously loaded image information before
            loading new information. Default is ``False``.
        index_col_patch : int, optional
            Column to set as index for the patch DataFrame, by default ``0``.
        index_col_parent : int, optional
            Column to set as index for the parent DataFrame, by default ``0``.

        Returns
        -------
        None
        """
        if clear_images:
            self.images = {"parent": {}, "patch": {}}

        if isinstance(patch_path, str) and os.path.isfile(patch_path):
            self.images["patch"].update(
                pd.read_csv(patch_path, index_col=index_col_patch).to_dict(
                    orient="index"
                )
            )

        if isinstance(parent_path, str) and os.path.isfile(parent_path):
            self.images["parent"].update(
                pd.read_csv(parent_path, index_col=index_col_parent).to_dict(
                    orient="index"
                )
            )

            self.addPatches()

            for parent_id in self.images["parent"].keys():
                k2change = "patches"
                if k2change in self.images["parent"][parent_id]:
                    self.images["parent"][parent_id][k2change] = eval(
                        self.images["parent"][parent_id][k2change]
                    )

                k2change = "coord"
                if k2change in self.images["parent"][parent_id]:
                    self.images["parent"][parent_id][k2change] = eval(
                        self.images["parent"][parent_id][k2change]
                    )

    def addGeoInfo(
        self,
        proj2convert: Optional[str] = "EPSG:4326",
        calc_method: Optional[str] = "great-circle",
        verbose: Optional[bool] = False,
    ) -> None:
        """
        Add geographic information (shape, coords, reprojected to EPSG:4326,
        and size in meters) to the ``images`` attribute of the ``mapImages``
        instance from image metadata.

        Parameters
        ----------
        proj2convert : str, optional
            Projection to convert coordinates into, by default ``"EPSG:4326"``.
        calc_method : str, optional
            Method to use for calculating image size in meters. Possible
            values: ``"great-circle"`` (default), ``"gc"`` (alias for
            ``"great-circle"``), ``"geodesic"``. ``"great-circle"`` and
            ``"gc"`` compute size using the great-circle distance formula,
            while ``"geodesic"`` computes size using the geodesic distance
            formula.
        verbose : bool, optional
            Whether to print progress messages or not. The default is
            ``False``.

        Returns
        -------
        None

        Notes
        -----
        This method reads the image files specified in the ``image_path`` key
        of each dictionary in the ``parent`` dictionary.

        It then checks if the image has geographic coordinates in its metadata,
        if not it prints a warning message and skips to the next image.

        If coordinates are present, this method converts them to the specified
        projection ``proj2convert`` and calculates the size of each pixel
        based on the method specified in ``calc_method``.

        The resulting information is then added to the dictionary in the
        ``parent`` dictionary corresponding to each image.

        Note that the calculations are performed using the
        ``geopy.distance.geodesic`` and ``geopy.distance.great_circle``
        methods. Thus, the method requires the ``geopy`` package to be
        installed.
        """
        image_ids = list(self.images["parent"].keys())

        for image_id in image_ids:
            image_path = self.images["parent"][image_id]["image_path"]

            # Read the image using rasterio
            tiff_src = rasterio.open(image_path)

            # Get height and width for image
            image_height, image_width = tiff_src.shape

            # Extract channels
            image_channels = tiff_src.count

            # Set shape
            shape = (image_height, image_width, image_channels)
            self.images["parent"][image_id]["shape"] = shape

            # Check whether coordinates are present
            if isinstance(tiff_src.crs, type(None)):
                print(
                    f"No coordinates found in {image_id}. Try add_metadata instead"  # noqa
                )
                continue

            else:
                # Get coordinates as string
                tiff_proj = tiff_src.crs.to_string()

                # Coordinate transformation: proj1 ---> proj2
                transformer = Transformer.from_crs(tiff_proj, proj2convert)
                ymax, xmin = transformer.transform(
                    tiff_src.bounds.left, tiff_src.bounds.top
                )
                ymin, xmax = transformer.transform(
                    tiff_src.bounds.right, tiff_src.bounds.bottom
                )

                # New projected coordinates
                coords = (xmin, xmax, ymin, ymax)
                self.images["parent"][image_id]["coord"] = coords

                # Calculate pixel size in meters
                size_in_m = self.calc_pixel_width_height(
                    parent_id=image_id,
                    calc_size_in_m=calc_method,
                    verbose=verbose,
                )
                self.images["parent"][image_id]["size_in_m"] = size_in_m

    '''
    def readPatches(self,
                  patch_paths,
                  parent_paths,
                  metadata=None,
                  metadata_fmt="dataframe",
                  metadata_cols2add=[],
                  metadata_index_column="image_id",
                  clear_images=False):
        """read patches from files (patch_paths) and add parents if
           parent_paths is provided
            Arguments:
            patch_paths {str, wildcard accepted} -- path to patches
            parent_paths {False or str, wildcard accepted} -- path to parents
            Keyword Arguments:
            clear_images {bool} -- clear images variable before reading
                    patches (default: {False})
        """
        patch_paths = glob(os.path.abspath(patch_paths))

        if clear_images:
            self.images = {}
            self.images["parent"] = {}
            self.images["patch"] = {}
            # XXX check
        if not isinstance(metadata, type(None)):
            include_metadata = True
            if metadata_fmt in ["dataframe"]:
                metadata_df = metadata
            elif metadata_fmt.lower() in ["csv"]:
                try:
                    metadata_df = pd.read_csv(metadata)
                except:
                    print(f"[WARNING] could not find metadata file: {metadata}")  # noqa
            else:
                print(f"format cannot be recognized: {metadata_fmt}")
                include_metadata = False
            if include_metadata:
                metadata_df['rd_index_id'] = metadata_df[metadata_index_column].apply(lambda x: os.path.basename(x))
        else:
            include_metadata = False
            for tpath in patch_paths:
            tpath = os.path.abspath(tpath)
            if not os.path.isfile(tpath):
                raise ValueError(f"patch_paths should point to actual files. Current patch_paths: {patch_paths}")
            # patch ID is set to the basename
            patch_id = os.path.basename(tpath)
            # XXXX
            if include_metadata and (not patch_id in list(metadata['rd_index_id'])):
                continue
            # Parent ID and border can be detected using patch_id
            parent_id = self.detectParIDfromPath(patch_id)
            min_x, min_y, max_x, max_y = self.detectBorderFromPath(patch_id)

            # Add patch
            if not self.images["patch"].get(patch_id, False):
                self.images["patch"][patch_id] = {}
            self.images["patch"][patch_id]["parent_id"] = parent_id
            self.images["patch"][patch_id]["image_path"] = tpath
            self.images["patch"][patch_id]["min_x"] = min_x
            self.images["patch"][patch_id]["min_y"] = min_y
            self.images["patch"][patch_id]["max_x"] = max_x
            self.images["patch"][patch_id]["max_y"] = max_y

        # XXX check
        if include_metadata:
            # metadata_cols = set(metadata_df.columns) - set(['rd_index_id'])
            for one_row in metadata_df.iterrows():
                for one_col in list(metadata_cols2add):
                    self.images["patch"][one_row[1]['rd_index_id']][one_col] = one_row[1][one_col]

        if parent_paths:
            # Add parents
            self.readParents(parent_paths=parent_paths)
            # Add patches to the parent
            self.addPatches()

    def process(self, tree_level="parent", update_paths=True,
                save_preproc_dir="./test_preproc"):
        """Process images using process.py module

        Args:
            tree_level (str, optional): "parent" or "patch" paths will be used. Defaults to "parent".
            update_paths (bool, optional): XXX. Defaults to True.
            save_preproc_dir (str, optional): Path to store preprocessed images. Defaults to "./test_preproc".
        """
            from mapreader import process
        # Collect paths and store them self.process_paths
        self.getProcessPaths(tree_level=tree_level)

        saved_paths = process.preprocess_all(self.process_paths,
                                             save_preproc_dir=save_preproc_dir)
        if update_paths:
            self.readParents(saved_paths, update=True)

    def getProcessPaths(self, tree_level="parent"):
        """Create a list of paths to be processed

        Args:
            tree_level (str, optional): "parent" or "patch" paths will be used. Defaults to "parent".
        """
        process_paths = []
        for one_img in self.images[tree_level].keys():
            process_paths.append(self.images[tree_level][one_img]["image_path"])
        self.process_paths = process_paths
        
    def prepare4inference(self, fmt="dataframe"):
        """Convert images to the specified format (fmt)
            Keyword Arguments:
            fmt {str} -- convert images variable to this format (default: {"dataframe"})
        """
        if fmt in ["pandas", "dataframe"]:
            patches = pd.DataFrame.from_dict(self.images["patch"], orient="index")
            patches.reset_index(inplace=True)
            if len(patches) > 0:
                patches.rename(columns={"image_path": "image_id"}, inplace=True)
                patches.drop(columns=["index", "parent_id"], inplace=True)
                patches["label"] = -1

            parents = pd.DataFrame.from_dict(self.images["parent"], orient="index")
            parents.reset_index(inplace=True)
            if len(parents) > 0:
                parents.rename(columns={"image_path": "image_id"}, inplace=True)
                parents.drop(columns=["index", "parent_id"], inplace=True)
                parents["label"] = -1

            return parents, patches
        else:
            raise ValueError(f"Format {fmt} is not supported!")
    '''