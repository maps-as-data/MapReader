#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import hashlib
import json
import os
import random
import string
import sys
from typing import Dict, List, Literal, Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml
from ipyannotate.annotation import Annotation
from ipyannotate.buttons import BackButton as Back
from ipyannotate.buttons import NextButton as Next
from ipyannotate.buttons import ValueButton as Button
from ipyannotate.canvas import OutputCanvas
from ipyannotate.tasks import Task, Tasks
from ipyannotate.toolbar import Toolbar
from IPython.display import clear_output, display
from PIL import Image

from mapreader import load_patches, loader

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
# warnings.filterwarnings(
#     "ignore", message="Pandas doesn't allow columns to be created via a new attribute name")


def prepare_data(
    df: pd.DataFrame,
    col_names: Optional[List[str]] = ["image_path", "parent_id"],
    annotation_set: Optional[str] = "001",
    label_col_name: Optional[str] = "label",
    redo: Optional[bool] = False,
    random_state: Optional[Union[int, str]] = "random",
    num_samples: Optional[int] = 100,
) -> List[List[Union[str, int]]]:
    """
    Prepare data for image annotation by selecting a subset of images from a
    DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the image data to be annotated.
    col_names : list of str, optional
        List of column names to include in the output. Default columns are
        ``["image_path", "parent_id"]``.
    annotation_set : str, optional
        String specifying the annotation set. Default is ``"001"``.
    label_col_name : str, optional
        Column name containing the label information for each image. Default
        is ``"label"``.
    redo : bool, optional
        If ``True``, all images will be annotated even if they already have a
        label. If ``False`` (default), only images without a label will be
        annotated.
    random_state : int or str, optional
        Seed for the random number generator used when selecting images to
        annotate. If set to ``"random"`` (default), a random seed will be used.
    num_samples : int, optional
        Maximum number of images to annotate. Default is ``100``.

    Returns
    -------
    list of list of str/int
        A list of lists containing the selected image data, with each sublist
        containing the specified columns plus the annotation set and a row
        counter.
    """

    if (label_col_name in list(df.columns)) and (not redo):
        already_annotated = len(df[~df[label_col_name].isnull()])
        print(f"Number of already annotated images: {already_annotated}")
        # only annotate those patches that have not been already annotated
        df = df[df[label_col_name].isnull()]
        print(f"Number of images to be annotated (total): {len(df)}")
    else:
        # if redo = True or "label" column does not exist
        # annotate all patches in the pandas dataframe
        pass

    tar_param = "mean_pixel_RGB"
    if tar_param in df.columns:
        try:
            pd.options.mode.chained_assignment = None
            df["pixel_groups"] = pd.qcut(
                df[tar_param], q=10, precision=2, labels=False
            ).values
            if random_state in ["random"]:
                df = df.groupby("pixel_groups").sample(
                    n=10, random_state=random.randint(0, 1e6)
                )
            else:
                df = df.groupby("pixel_groups").sample(n=10, random_state=random_state)
        except Exception:
            print(f"[INFO] len(df) = {len(df)}, .sample method is deactivated.")
            df = df.iloc[:num_samples]
    else:
        print(f"[WARNING] could not find {tar_param} in columns.")
        df = df.iloc[:num_samples]

    data = []
    row_counter = 0
    for one_row in df.iterrows():
        cols2add = [one_row[0]]
        for i in col_names:
            cols2add.append(one_row[1][i])
        cols2add.append(annotation_set)
        cols2add.append(row_counter)
        data.append(cols2add)
        row_counter += 1

    print(f"Number of images to annotate (current batch): {len(data)}")
    return data


def annotation_interface(
    data: List,
    list_labels: List,
    list_colors: Optional[List[str]] = ["red", "green", "blue", "green"],
    annotation_set: Optional[str] = "001",
    method: Optional[Literal["ipyannotate", "pigeonxt"]] = "ipyannotate",
    list_shortcuts: Optional[List[str]] = None,
) -> Annotation:
    """
    Create an annotation interface for a list of patches with corresponding
    labels.

    Parameters
    ----------
    data : list
        List of patches to annotate.
    list_labels : list
        List of strings representing the labels for each annotation class.
    list_colors : list, optional
        List of strings representing the colors for each annotation class,
        by default ``["red", "green", "blue", "green"]``.
    annotation_set : str, optional
        String representing the annotation set, specified in the yaml file or
        via function argument, by default ``"001"``.
    method : Literal["ipyannotate", "pigeonxt"], optional
        String representing the method for annotation, by default
        ``"ipyannotate"``.
    list_shortcuts : list, optional
        List of strings representing the keyboard shortcuts for each
        annotation class, by default ``None``.

    Returns
    -------
    annotation : Annotation
        The annotation object containing the toolbar, tasks and canvas for the
        interface.

    Raises
    ------
    SystemExit
        If ``method`` parameter is not ``"ipyannotate"`` or ``pigeonxt``.

    Notes
    -----
    This function creates an annotation interface using the ``ipyannotate``
    library, which is a browser-based tool for annotating data.
    """

    if method.lower() == "ipyannotate":

        def display_record(record: Tuple[str, str, str, int, int]) -> None:
            """
            Displays an image and optionally, a context image with a patch
            border.

            Parameters
            ----------
            record : tuple
                A tuple containing the following elements:
                    - str : The name of the patch.
                    - str : The path to the image to be displayed.
                    - str : The path to the parent image, if any.
                    - int : The index of the task, if any.
                    - int : The number of times this patch has been displayed.

            Returns
            -------
            None

            Notes
            -----
            This function should be called from ``prepare_annotation``, there
            are several global variables that are being set in the function.

            This function uses ``matplotlib`` to display images. If the
            context image is displayed, the border of the patch is highlighted
            in red.

            Refer to ``ipyannotate`` and ``matplotlib`` for more info.
            """

            # setup the images
            gridsize = (5, 1)
            plt.clf()
            plt.figure(figsize=(12, 12))
            if treelevel == "patch" and contextimage:
                plt.subplot2grid(gridsize, (2, 0))
            else:
                plt.subplot2grid(gridsize, (0, 0), rowspan=2)
            plt.imshow(Image.open(record[1]))
            plt.xticks([])
            plt.yticks([])
            plt.title(f"{record[0]}", size=20)

            if treelevel == "patch" and contextimage:
                parent_path = os.path.dirname(
                    annotation_tasks["paths"][record[3]]["parent_paths"]
                )
                # Here, we assume that min_x, min_y, max_x and max_y are in the patch
                # name
                split_path = record[0].split("-")
                min_x, min_y, max_x, max_y = (
                    int(split_path[1]),
                    int(split_path[2]),
                    int(split_path[3]),
                    int(split_path[4]),
                )

                # context image
                plt.subplot2grid(gridsize, (0, 0), rowspan=2)

                # ---
                path = os.path.join(parent_path, record[2])
                par_img = Image.open(path).convert("RGB")
                min_y_par = max(0, min_y - y_offset)
                min_x_par = max(0, min_x - x_offset)
                max_x_par = min(max_x + x_offset, np.shape(par_img)[1])
                max_y_par = min(max_y + y_offset, np.shape(par_img)[0])

                # par_img = par_img[min_y_par:max_y_par, min_x_par:max_x_par]
                par_img = par_img.crop((min_x_par, min_y_par, max_x_par, max_y_par))

                plt.imshow(par_img, extent=(min_x_par, max_x_par, max_y_par, min_y_par))
                # ---

                plt.xticks([])
                plt.yticks([])

                # plot the patch border on the context image
                plt.plot([min_x, min_x], [min_y, max_y], lw=2, zorder=10, color="r")
                plt.plot([min_x, max_x], [min_y, min_y], lw=2, zorder=10, color="r")
                plt.plot([max_x, max_x], [max_y, min_y], lw=2, zorder=10, color="r")
                plt.plot([max_x, min_x], [max_y, max_y], lw=2, zorder=10, color="r")

                """
                # context image
                plt.subplot2grid(gridsize, (3, 0), rowspan=2)
                min_y_par = 0
                min_x_par = 0
                max_x_par = par_img.shape[1]
                max_y_par = par_img.shape[0]
                plt.imshow(par_img[min_y_par:max_y_par, min_x_par:max_x_par],
                            extent=(min_x_par, max_x_par, max_y_par, min_y_par))
                plt.plot([min_x_par, min_x_par],
                            [min_y_par, max_y_par],
                            lw=2, zorder=10, color="k")
                plt.plot([min_x_par, max_x_par],
                            [min_y_par, min_y_par],
                            lw=2, zorder=10, color="k")
                plt.plot([max_x_par, max_x_par],
                            [max_y_par, min_y_par],
                            lw=2, zorder=10, color="k")
                plt.plot([max_x_par, min_x_par],
                            [max_y_par, max_y_par],
                            lw=2, zorder=10, color="k")

                plt.xticks([])
                plt.yticks([])

                # plot the patch border on the context image
                plt.plot([min_x, min_x],
                            [min_y, max_y],
                            lw=2, zorder=10, color="r")
                plt.plot([min_x, max_x],
                            [min_y, min_y],
                            lw=2, zorder=10, color="r")
                plt.plot([max_x, max_x],
                            [max_y, min_y],
                            lw=2, zorder=10, color="r")
                plt.plot([max_x, min_x],
                            [max_y, max_y],
                            lw=2, zorder=10, color="r")
                """

            plt.tight_layout()
            plt.show()

            print(20 * "-")
            print("Additional info:")
            print(f"Counter: {record[-1]}")
            if url_main:
                try:
                    map_id = record[2].split("_")[-1].split(".")[0]
                    url = f"{url_main}/{map_id}"
                    # stream=True so we don't download the whole page, only check if
                    # the page exists
                    response = requests.get(url, stream=True)
                    assert response.status_code < 400
                    print()
                    print(f"URL: {url}")
                except:
                    url = False
                    pass

        if not list_shortcuts:
            list_shortcuts = [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
            ]
        list_colors *= 10
        canvas = OutputCanvas(display=display_record)
        # Collect all tasks
        tasks = Tasks(Task(_) for _ in data)
        buttons = []
        for i, one_label in enumerate(list_labels):
            buttons.append(
                Button(
                    i + 1,
                    label=one_label,
                    color=list_colors[i],
                    shortcut=list_shortcuts[i],
                )
            )
        controls = [Back(shortcut="j"), Next(shortcut="k")]
        toolbar = Toolbar(buttons + controls)
        annotation = Annotation(toolbar, tasks, canvas=canvas)
        return annotation

    sys.exit(
        f"method: {method} is not implemented. Currently, we support: ipyannotate and pigeonxt"  # noqa
    )


def prepare_annotation(
    userID: str,
    task: str,
    annotation_tasks_file: str,
    custom_labels: List[str] = [],
    annotation_set: Optional[str] = "001",
    redo_annotation: Optional[bool] = False,
    patch_paths: Optional[Union[str, bool]] = False,
    parent_paths: Optional[str] = False,
    tree_level: Optional[str] = "patch",
    sortby: Optional[str] = None,
    min_alpha_channel: Optional[float] = None,
    min_mean_pixel: Optional[float] = None,
    max_mean_pixel: Optional[float] = None,
    min_std_pixel: Optional[float] = None,
    max_std_pixel: Optional[float] = None,
    context_image: Optional[bool] = False,
    xoffset: Optional[int] = 500,
    yoffset: Optional[int] = 500,
    urlmain: Optional[str] = "https://maps.nls.uk/view/",
    random_state: Optional[Union[str, int]] = "random",
    list_shortcuts: Optional[List[tuple]] = None,
    method: Optional[Literal["ipyannotate", "pigeonxt"]] = "ipyannotate",
) -> Dict:
    """Prepare image data for annotation and launch the annotation interface.

    Parameters
    ----------
    userID : str
        The ID of the user annotating the images. Should be unique as it is
        used in the name of the output file.
    task : str
        The task name that the images are associated with. This task should be
        defined in the yaml file (``annotation_tasks_file``), if not,
        ``custom_labels`` will be used instead.
    annotation_tasks_file : str
        The file path to the YAML file containing information about task, image
        paths and annotation metadata.
    custom_labels : list of str, optional
        A list of custom label names to be used instead of the label names in
        the ``annotation_tasks_file``. Default is ``[]``.
    annotation_set : str, optional
        The ID of the annotation set to use in the YAML file
        (``annotation_tasks_file``). Default is ``"001"``.
    redo_annotation : bool, optional
        If ``True``, allows the user to redo annotations on previously
        annotated images. Default is ``False``.
    patch_paths : str or bool, optional
        The path to the directory containing patches, if ``custom_labels`` are provided. Default is ``False`` and the information is read from the yaml file.
    parent_paths : str, optional
        The path to parent images, if ``custom_labels`` are provided. Default
        is ``False`` and the information is read from the yaml file.
    tree_level : str, optional
        The level of annotation to be used, either ``"patch"`` or ``"parent"``.
        Default is ``"patch"``.
    sortby : str, optional
        If ``"mean"``, sort images by mean pixel intensity. Default is
        ``None``.
    min_alpha_channel : float, optional
        The minimum alpha channel value for images to be included in the
        annotation interface. Only applies to patch level annotations.
        Default is ``None``.
    min_mean_pixel : float, optional
        The minimum mean pixel intensity value for images to be included in
        the annotation interface. Only applies to patch level annotations.
        Default is ``None``.
    max_mean_pixel : float, optional
        The maximum mean pixel intensity value for images to be included in
        the annotation interface. Only applies to patch level annotations.
        Default is ``None``.
    min_std_pixel : float, optional
        The minimum standard deviation of pixel intensity value for images to be included in
        the annotation interface. Only applies to patch level annotations.
        Default is ``None``.
    max_std_pixel : float, optional
        The maximum standard deviation of pixel intensity value for images to be included in
        the annotation interface. Only applies to patch level annotations.
        Default is ``None``.
    context_image : bool, optional
        If ``True``, includes a context image with each patch image in the
        annotation interface. Only applies to patch level annotations. Default
        is ``False``.
    xoffset : int, optional
        The x-offset in pixels to be used for displaying context images in the
        annotation interface. Default is ``500``.
    yoffset : int, optional
        The y-offset in pixels to be used for displaying context images in the
        annotation interface. Default is ``500``.
    urlmain : str, optional
        The main URL to be used for displaying images in the annotation
        interface. Default is ``"https://maps.nls.uk/view/"``.
    random_state : int or str, optional
        Seed or state value for the random number generator used for shuffling
        the image order. Default is ``"random"``.
    list_shortcuts : list of tuples, optional
        A list of tuples containing shortcut key assignments for label names.
        Default is ``None``.
    method : Literal["ipyannotate", "pigeonxt"], optional
        String representing the method for annotation, by default
        ``"ipyannotate"``.

    Returns
    -------
    annotation : dict
        A dictionary containing the annotation results.

    Raises
    -------
    ValueError
        If a specified annotation_set is not a key in the paths dictionary
        of the YAML file with the information about the annotation metadata
        (``annotation_tasks_file``).
    """

    # Specify global variables so they can be used in display_record function
    global annotation_tasks
    global x_offset
    global y_offset
    global url_main
    global treelevel
    global contextimage

    # Note: it is not possible to define global variable + args with the same
    # names so here, we read xoffset and yoffset, assign them to two global
    # variables as these global variables will then be used in display_record
    x_offset = xoffset
    y_offset = yoffset
    url_main = urlmain
    treelevel = tree_level
    contextimage = context_image

    with open(annotation_tasks_file) as annot_file_fio:
        annotation_tasks = yaml.load(annot_file_fio, Loader=yaml.FullLoader)

    if annotation_set not in annotation_tasks["paths"].keys():
        raise ValueError(
            f"{annotation_set} could not be found in {annotation_tasks_file}"
        )
    else:
        if tree_level == "patch":
            patch_paths = annotation_tasks["paths"][annotation_set]["patch_paths"]
        parent_paths = os.path.join(
            annotation_tasks["paths"][annotation_set]["parent_paths"]
        )
        annot_file = os.path.join(
            annotation_tasks["paths"][annotation_set]["annot_dir"],
            f"{task}_#{userID}#.csv",
        )

    if task not in annotation_tasks["tasks"].keys():
        if custom_labels == []:
            raise ValueError(
                f"Task: {task} could not be found and custom_labels == []."
            )
        list_labels = custom_labels
    else:
        list_labels = annotation_tasks["tasks"][task]["labels"]

    if tree_level == "patch":
        # specify the path of patches and the parent images
        mymaps = load_patches(patch_paths=patch_paths, parent_paths=parent_paths)
        if os.path.isfile(annot_file):
            mymaps.add_metadata(
                metadata=annot_file,
                index_col=-1,
                delimiter=",",
                tree_level=tree_level,
            )

        calc_mean = calc_std = False
        # Calculate mean before converting to pandas so the dataframe contains information about mean pixel intensity
        if (
            sortby == "mean"
            or isinstance(min_alpha_channel, float)
            or isinstance(min_mean_pixel, float)
            or isinstance(max_mean_pixel, float)
        ):
            calc_mean = True

        if isinstance(min_std_pixel, float) or isinstance(max_std_pixel, float):
            calc_std = True

        if calc_mean or calc_std:
            mymaps.calc_pixel_stats(calc_mean=calc_mean, calc_std=calc_std)

        # convert images to dataframe
        _, patch_df = mymaps.convertImages()

        if sortby == "mean":
            patch_df.sort_values("mean_pixel_RGB", inplace=True)

        if isinstance(min_alpha_channel, float):
            if "mean_pixel_A" in patch_df.columns:
                patch_df = patch_df[patch_df["mean_pixel_A"] >= min_alpha_channel]

        if isinstance(min_mean_pixel, float):
            if "mean_pixel_RGB" in patch_df.columns:
                patch_df = patch_df[patch_df["mean_pixel_RGB"] >= min_mean_pixel]

        if isinstance(max_mean_pixel, float):
            if "mean_pixel_RGB" in patch_df.columns:
                patch_df = patch_df[patch_df["mean_pixel_RGB"] <= max_mean_pixel]

        if isinstance(min_std_pixel, float):
            if "std_pixel_RGB" in patch_df.columns:
                patch_df = patch_df[patch_df["std_pixel_RGB"] >= min_std_pixel]

        if isinstance(max_std_pixel, float):
            if "std_pixel_RGB" in patch_df.columns:
                patch_df = patch_df[patch_df["std_pixel_RGB"] <= max_std_pixel]

        if isinstance(min_std_pixel, float):
            if "std_pixel_RGB" in patch_df.columns:
                patch_df = patch_df[patch_df["std_pixel_RGB"] >= min_std_pixel]

        if isinstance(max_std_pixel, float):
            if "std_pixel_RGB" in patch_df.columns:
                patch_df = patch_df[patch_df["std_pixel_RGB"] <= max_std_pixel]

        col_names = ["image_path", "parent_id"]
    else:
        mymaps = loader(path_images=parent_paths)
        if os.path.isfile(annot_file):
            mymaps.add_metadata(
                metadata=annot_file,
                index_col=-1,
                delimiter=",",
                tree_level=tree_level,
            )
        # convert images to dataframe
        patch_df, _ = mymaps.convertImages()
        col_names = ["image_path"]

    # prepare data for annotation
    data2annotate = prepare_data(
        patch_df,
        col_names=col_names,
        annotation_set=annotation_set,
        redo=redo_annotation,
        random_state=random_state,
    )

    if len(data2annotate) == 0:
        print("No image to annotate!")
    else:
        annotation = annotation_interface(
            data2annotate,
            list_labels=list_labels,
            annotation_set=annotation_set,
            list_shortcuts=list_shortcuts,
            method=method,
        )
        return annotation


def save_annotation(
    annotation: Annotation,
    userID: str,
    task: str,
    annotation_tasks_file: str,
    annotation_set: str,
) -> None:
    """
    Save annotations for a given task and user to a csv file.

    Parameters
    ----------
    annotation : ipyannotate.annotation.Annotation
        Annotation object containing the annotations to be saved (output from
        the annotation tool).
    userID : str
        User ID of the person performing the annotation. This should be unique
        as it is used in the name of the output file.
    task : str
        Name of the task being annotated.
    annotation_tasks_file : str
        Path to the yaml file describing the annotation tasks, paths, etc.
    annotation_set : str
        Name of the annotation set to which the annotations belong, defined in
        the ``annotation_tasks_file``.

    Returns
    -------
    None
    """
    with open(annotation_tasks_file) as f:
        annotation_tasks = yaml.load(f, Loader=yaml.FullLoader)

    if annotation_set not in annotation_tasks["paths"].keys():
        print(f"{annotation_set} could not be found in {annotation_tasks_file}")
    else:
        annot_file = os.path.join(
            annotation_tasks["paths"][annotation_set]["annot_dir"],
            f"{task}_#{userID}#.csv",
        )

    annot_file_par = os.path.dirname(os.path.abspath(annot_file))
    if not os.path.isdir(annot_file_par):
        os.makedirs(annot_file_par)

    # Read an existing annotation file (for the same task and userID)
    try:
        image_df = pd.read_csv(annot_file)
    except:
        image_df = pd.DataFrame(columns=["image_id", "label"])

    new_labels = 0
    newly_annotated = 0
    for i in range(len(annotation.tasks)):
        if annotation.tasks[i].value is not None:
            newly_annotated += 1
            if (
                not annotation.tasks[i].output[0]
                in image_df["image_id"].values.tolist()
            ):
                image_df = image_df.append(
                    {
                        "image_id": annotation.tasks[i].output[0],
                        "label": annotation.tasks[i].value,
                    },
                    ignore_index=True,
                )
                new_labels += 1

    if len(image_df) > 0:
        image_df = image_df.set_index("image_id")
        image_df.to_csv(annot_file, mode="w")
        print(f"[INFO] Save {newly_annotated} new annotations to {annot_file}")
        print(f"[INFO] {new_labels} labels were not already stored")
        print(f"[INFO] Total number of annotations: {len(image_df)}")
    else:
        print("[INFO] No annotations to save!")


class Annotator(pd.DataFrame):
    """
    A subclass of pd.DataFrame that allows the user to annotate images and
    save the annotations as a CSV file.

    Parameters
    ----------
    data : str or dict or list or None, optional
        Data to create the Annotator DataFrame. Default:
        "./patches/patch-*.png"
    patches : str, optional
        Path to patches folder to load the images, if no ``data`` is provided.
        Default: "./patches/patch-*.png"
    parents : str, optional
        Path to parents folder to load the parent images, if no ``data`` is
        provided. Default: "./maps/*.png"
    metadata : str, optional
        Path to metadata file. Default: "./maps/metadata.csv"
    annotations_dir : str, optional
        Directory to save the annotations CSV file. Default: "./annotations"
    username : str, optional
        Username for the annotation session. If no username is provided, the
        class will generate a hash string for the current user. Note: This
        means that annotations cannot be resumed later.
    task_name : str, optional
        Name of the annotation task. Default: "task"
    image_column : str, optional
        Name of the image column in the ``data`` dataframe, describing the
        path to the image. Default: "image_path"
    label_column : str, optional
        Name of the label column in the ``data`` dataframe, where final labels
        are/will be stored. Default: "label"
    labels : list of str
        List of possible labels for the annotations. Must be provided as a
        list.
    scramble_frame : bool, optional
        Whether to randomly shuffle the examples during annotation. Default:
        True.
    buttons_per_row : int, optional
        Number of buttons to display per row in the annotation interface.
    auto_save : bool, optional
        Whether to automatically save annotations during annotation. Default:
        True.
    example_process_fn : function, optional
        Function to process each example during annotation. Default: None.
    final_process_fn : function, optional
        Function to process the entire annotation process. Default: None.

    Attributes
    ----------
    buttons : list of widgets.Button
        List of annotation buttons.
    labels : list of str
        List of possible labels for the annotations.
    label_column : str
        Name of the label column in the dataframe.
    image_column : str
        Name of the image column in the dataframe.
    buttons_per_row : int
        Number of buttons to display per row in the annotation interface.
    example_process_fn : function
        Function to process each example during annotation.
    final_process_fn : function
        Function to process the entire annotation process.
    auto_save : bool
        Whether to automatically save annotations during annotation.
    annotations_dir : str
        Directory to save the annotations CSV file.
    task_name : str
        Name of the annotation task.
    id : str
        Unique identifier for the current annotation session.
    _file_name : str
        Filename for the annotations CSV file.
    username : str
        Username for the current annotation session.
    current_index : int
        Current index of the annotation process.
    stop_at_last_example : bool
        Whether the annotation process should stop when it reaches the last
        example in the dataframe.
    out : Output
        Output widget that displays the current image during annotation.
    box : HBox or VBox
        Widget box that contains the annotation buttons.
    filtered
        Returns a new dataframe that only contains rows with non-null labels.

    Methods
    -------
    annotate()
        Renders the annotation interface for the first image.
    render()
        Displays the image at the current index in the annotation interface.
    get_labelled_data(sort=True)
        Returns a dataframe that contains the index and the label column,
        sorted by (a) the filename of the parent map image and (b) the
        annotated patch's filename.
    get_current_index()
        Returns the current index of the annotation process.

    Example
    -------
    >>> annotator = Annotator(
            task_name="railspace",
            labels=["no_rail_space", "rail_space"],
            username="james",
            patches="./patches/patch-*.png"
            parents="./maps/*.png"
            annotations_dir="./annotations"
        )
    >>> annotator.annotate()
    """

    def __init__(self, *args, **kwargs):
        # Untangle args and kwargs
        data = None
        if len(args) == 1:
            data = args.pop(0)
        elif len(args) == 2:
            data = args.pop(0)
            metadata = args.pop(1)
        elif isinstance(
            kwargs.get("data"), (str, dict, list, type(None), pd.DataFrame)
        ):
            data = kwargs.get("data")
        elif kwargs.get("data") is not None:
            raise SyntaxError(f"Cannot interpret data of type {type(data)}")

        if isinstance(
            kwargs.get("metadata"), (str, dict, list, type(None), pd.DataFrame)
        ):
            metadata = kwargs.get("metadata")

        kwargs["patches"] = (
            kwargs["patches"] if kwargs.get("patches") else "./patches/patch-*.png"
        )
        kwargs["parents"] = (
            kwargs["parents"] if kwargs.get("parents") else "./maps/*.png"
        )
        kwargs["annotations_dir"] = (
            kwargs["annotations_dir"]
            if kwargs.get("annotations_dir")
            else "./annotations"
        )
        kwargs["metadata"] = (
            kwargs["metadata"] if kwargs.get("metadata") else "./maps/metadata.csv"
        )
        kwargs["username"] = kwargs["username"] if kwargs.get("username") else None
        kwargs["task_name"] = kwargs["task_name"] if kwargs.get("task_name") else "task"
        kwargs["image_column"] = (
            kwargs["image_column"] if kwargs.get("image_column") else "image_path"
        )
        kwargs["label_column"] = (
            kwargs["label_column"] if kwargs.get("label_column") else "label"
        )
        kwargs["labels"] = kwargs["labels"] if kwargs.get("labels") else []
        kwargs["scramble_frame"] = (
            kwargs["scramble_frame"] if kwargs.get("scramble_frame") else True
        )
        kwargs["buttons_per_row"] = (
            kwargs["buttons_per_row"] if kwargs.get("buttons_per_row") else None
        )
        kwargs["auto_save"] = kwargs["auto_save"] if kwargs.get("auto_save") else True
        kwargs["example_process_fn"] = (
            kwargs["example_process_fn"] if kwargs.get("example_process_fn") else None
        )
        kwargs["final_process_fn"] = (
            kwargs["final_process_fn"] if kwargs.get("final_process_fn") else None
        )
        kwargs["username"] = (
            kwargs["username"]
            if kwargs.get("username")
            else "".join(
                [random.choice(string.ascii_letters + string.digits) for n in range(30)]
            )
        )

        # Check metadata
        if isinstance(metadata, str):
            # we have data as string = assume it's a path to a
            data = pd.read_csv(data)

        if isinstance(metadata, (dict, list)):
            # we have data as string = assume it's a path to a
            data = pd.DataFrame(data)

        # Check data
        if isinstance(data, str):
            # we have data as string = assume it's a path to a
            data = pd.read_csv(data)

        if isinstance(data, (dict, list)):
            # we have data as string = assume it's a path to a
            data = pd.DataFrame(data)

        if isinstance(data, type(None)):
            # If we don't get data provided, we'll use the patches and parents to
            # load up the patches
            try:
                metadata, data = self._load_frames(**kwargs)
            except NameError:
                raise SyntaxError(
                    "Data must be provided or class must have a _load_frames method."
                )

        # Last check for metadata + data
        if not len(data):
            raise RuntimeError("No data available.")

        if not len(metadata):
            raise RuntimeError("No metadata available.")

        # Test for columns
        if kwargs["label_column"] not in data.columns:
            raise SyntaxError(
                f"Your DataFrame does not have the label column ({kwargs['label_column']})"
            )

        if kwargs["image_column"] not in data.columns:
            raise SyntaxError(
                f"Your DataFrame does not have the image column ({kwargs['image_column']})"
            )

        image_list = json.dumps(
            sorted(data[kwargs["image_column"]].to_list()), sort_keys=True
        )
        kwargs["id"] = hashlib.md5(image_list.encode("utf-8")).hexdigest()

        _file_name = (
            kwargs["task_name"].replace(" ", "_")
            + f"_#{kwargs['username']}#-{kwargs['id']}.csv"
        )
        kwargs["_file_name"] = os.path.join(kwargs["annotations_dir"], _file_name)

        # Test for existing file
        if os.path.exists(kwargs["_file_name"]):
            print(
                f"[INFO] Existing annotations for {kwargs['username']} being loaded..."
            )
            existing_annotations = pd.read_csv(kwargs["_file_name"], index_col=0)
            existing_annotations[kwargs["label_column"]] = existing_annotations[
                kwargs["label_column"]
            ].apply(lambda x: kwargs["labels"][x])

            data = data.join(
                existing_annotations, how="left", lsuffix="_x", rsuffix="_y"
            )
            data[kwargs["label_column"]] = data["label_y"].fillna(
                data[f"{kwargs['label_column']}_x"]
            )
            data = data.drop(
                columns=[f"{kwargs['label_column']}_x", f"{kwargs['label_column']}_y"]
            )
            data["changed"] = data[kwargs["label_column"]].apply(
                lambda x: True if x else False
            )

        # initiate as a DataFrame
        super().__init__(data)

        self.buttons = []
        self.labels = kwargs["labels"]
        self.label_column = kwargs["label_column"]
        self.image_column = kwargs["image_column"]
        self.buttons_per_row = kwargs["buttons_per_row"]
        self.example_process_fn = kwargs["example_process_fn"]
        self.final_process_fn = kwargs["final_process_fn"]
        self.auto_save = kwargs["auto_save"]
        self.annotations_dir = kwargs["annotations_dir"]
        self.task_name = kwargs["task_name"]
        self.id = kwargs["id"]
        self._file_name = kwargs["_file_name"]
        self.username = kwargs["username"]
        self.metadata = metadata

        # Set current index
        self.current_index = -1

        # Set max buttons
        if not self.buttons_per_row:
            self.buttons_per_row = len(self.labels)

        # Setup buttons
        self._setup_buttons()

        # Setup box for buttons
        self._setup_box()

    def _load_frames(self, *args, **kwargs):
        patches = load_patches(
            patch_paths=kwargs["patches"], parent_paths=kwargs["parents"]
        )
        patches.calc_pixel_stats()

        patches.add_metadata(kwargs["metadata"])

        parents, patches = patches.convertImages()

        if kwargs["label_column"] not in patches.columns:
            patches[kwargs["label_column"]] = None
        patches["changed"] = False

        if kwargs["scramble_frame"]:
            # Scramble them!
            patches = patches.sample(frac=1)

        return parents, patches

    def _setup_buttons(self):
        for label in self.labels:
            btn = widgets.Button(description=label)

            def on_click(lbl, *_, **__):
                self._add_annotation(lbl)

            btn.on_click(functools.partial(on_click, label))
            self.buttons.append(btn)

        # back button
        btn = widgets.Button(description="prev", button_style="info")
        btn.on_click(self._prev_example)
        self.buttons.append(btn)

        # next button
        btn = widgets.Button(description="next", button_style="info")
        btn.style.button_color = "#B3C8D0"
        btn.on_click(self._next_example)
        self.buttons.append(btn)

    def _setup_box(self):
        if len(self.buttons) > self.buttons_per_row:
            self.box = widgets.VBox(
                [
                    widgets.HBox(self.buttons[x : x + self.buttons_per_row])
                    for x in range(0, len(self.buttons), self.buttons_per_row)
                ]
            )
        else:
            self.box = widgets.HBox(self.buttons)

    def annotate(self) -> None:
        """
        Renders the annotation interface for the first image.

        Returns
        -------
        None
        """
        self.out = widgets.Output()
        display(self.box)
        display(self.out)
        self._next_example()

    def _next_example(self, *args):
        if self.current_index < len(self):
            if len(args) == 1 and isinstance(args[0], int):
                self.current_index = args[0]
            else:
                self.current_index += 1
            self.render()

    def _prev_example(self, *args):
        if self.current_index > 0:
            self.current_index -= 1
            self.render()

    def render(self) -> None:
        """
        Displays the image at the current index in the annotation interface.

        If the current index is greater than or equal to the length of the
        dataframe, the method disables the "next" button and calls the
        ``final_process_fn`` method, if defined.

        Returns
        -------
        None
        """
        # Check whether we have reached the end
        if self.current_index >= len(self):
            if self.stop_at_last_example:
                print("Annotation done.")
                if self.final_process_fn is not None:
                    if self.auto_save:
                        self._auto_save()
                    self.final_process_fn(self)
                for button in self.buttons:
                    button.disabled = True
            else:
                self._prev_example()
            return

        # render buttons
        ix = self.iloc[self.current_index].name
        for button in self.buttons:
            if button.description == "prev":
                # disable previous button when at first example
                button.disabled = self.current_index <= 0
            elif button.description == "next":
                # disable skip button when at last example
                button.disabled = self.current_index >= len(self) - 1
            elif button.description != "submit":
                if self.at[ix, self.label_column] == button.description:
                    button.icon = "check"
                else:
                    button.icon = ""

        # display new example
        with self.out:
            clear_output(wait=True)
            image_path = self.at[ix, self.image_column]
            with open(image_path, "rb") as f:
                image = f.read()
            display(widgets.Image(value=image))

    def _add_annotation(self, annotation):
        """Toggle annotation."""
        ix = self.iloc[self.current_index].name
        self.at[ix, self.label_column] = annotation
        self.at[ix, "changed"] = True
        if self.example_process_fn is not None:
            self.example_process_fn(self.at[ix, self.image_column], annotation)
        if self.auto_save:
            self._auto_save()
        self._next_example(self.get_current_index())

    @property
    def filtered(self) -> pd.DataFrame:
        _filter = ~self[self.label_column].isna()
        return self[_filter]

    def get_labelled_data(self, sort: bool = Optional[True]) -> pd.DataFrame:
        """
        Returns a dataframe containing only the labelled images and their
        associated label index.

        Parameters
        ----------
        sort : bool, optional
            Whether to sort the dataframe by the order of the images in the
            input data, by default True

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the labelled images and their associated
            label index.
        """
        col = self.filtered[self.label_column].apply(lambda x: self.labels.index(x))
        df = pd.DataFrame(col, index=pd.Index(col.index, name="image_id"))
        if not sort:
            return df

        df["sort_value"] = df.index.to_list()
        df["sort_value"] = df["sort_value"].apply(
            lambda x: f"{x.split('#')[1]}-{x.split('#')[0]}"
        )
        return df.sort_values("sort_value").drop(columns=["sort_value"])

    def _auto_save(self):
        self.get_labelled_data(sort=True).to_csv(self._file_name)

    def get_current_index(self) -> int:
        """
        Returns the current index in the dataframe of the image being
        displayed in the annotation interface.

        If the current index is less than 0 or greater than or equal to the
        length of the dataframe, the method returns the
        current index.

        Returns
        -------
        int
            The current index in the dataframe of the image being displayed in
            the annotation interface.
        """
        if self.current_index == -1:
            self.current_index = 0

        while True:
            if self.current_index == len(self):
                return self.current_index

            ix = self.iloc[self.current_index].name

            # If the label column at the index is None, return the index,
            # otherwise add one and continue
            if isinstance(self.at[ix, self.label_column], type(None)):
                return self.current_index
            else:
                self.current_index += 1
