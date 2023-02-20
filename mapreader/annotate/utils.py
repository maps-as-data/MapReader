#!/usr/bin/env python
# -*- coding: utf-8 -*-

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL.Image as PIL_image
from PIL import ImageOps
import random
import requests
import sys
import yaml

from mapreader import loader, load_patches

from ipyannotate.toolbar import Toolbar
from ipyannotate.tasks import Task, Tasks
from ipyannotate.canvas import OutputCanvas
from ipyannotate.annotation import Annotation
from ipyannotate.buttons import (
    ValueButton as Button,
    NextButton as Next,
    BackButton as Back,
)

# -------- display_record
def display_record(record):
    """Display patches for annotation

    NOTE: This function should only be called from prepare_annotation,
          there are several global variables that are being set in the function.

    Refer to ipyannotate for more info.
    """

    # setup the images
    gridsize = (5, 1)
    plt.clf()
    plt.figure(figsize=(12, 12))
    if treelevel == "child" and contextimage:
        plt.subplot2grid(gridsize, (2, 0))
    else:
        plt.subplot2grid(gridsize, (0, 0), rowspan=2)
    plt.imshow(PIL_image.open(record[1]))
    plt.xticks([])
    plt.yticks([])
    plt.title(f"{record[0]}", size=20)

    if treelevel == "child" and contextimage:
        parent_path = os.path.dirname(
            annotation_tasks["paths"][record[3]]["parent_paths"]
        )
        # Here, we assume that min_x, min_y, max_x and max_y are in the patch name
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
        par_img = PIL_image.open(os.path.join(parent_path, record[2])).convert("RGB")
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

        ## # context image
        ## plt.subplot2grid(gridsize, (3, 0), rowspan=2)
        ## min_y_par = 0
        ## min_x_par = 0
        ## max_x_par = par_img.shape[1]
        ## max_y_par = par_img.shape[0]
        ## plt.imshow(par_img[min_y_par:max_y_par, min_x_par:max_x_par],
        ##            extent=(min_x_par, max_x_par, max_y_par, min_y_par))
        ## #plt.plot([min_x_par, min_x_par],
        ## #         [min_y_par, max_y_par],
        ## #         lw=2, zorder=10, color="k")
        ## #plt.plot([min_x_par, max_x_par],
        ## #         [min_y_par, min_y_par],
        ## #         lw=2, zorder=10, color="k")
        ## #plt.plot([max_x_par, max_x_par],
        ## #         [max_y_par, min_y_par],
        ## #         lw=2, zorder=10, color="k")
        ## #plt.plot([max_x_par, min_x_par],
        ## #         [max_y_par, max_y_par],
        ## #         lw=2, zorder=10, color="k")

        ## plt.xticks([])
        ## plt.yticks([])

        ## # plot the patch border on the context image
        ## plt.plot([min_x, min_x],
        ##          [min_y, max_y],
        ##          lw=2, zorder=10, color="r")
        ## plt.plot([min_x, max_x],
        ##          [min_y, min_y],
        ##          lw=2, zorder=10, color="r")
        ## plt.plot([max_x, max_x],
        ##          [max_y, min_y],
        ##          lw=2, zorder=10, color="r")
        ## plt.plot([max_x, min_x],
        ##          [max_y, max_y],
        ##          lw=2, zorder=10, color="r")

    plt.tight_layout()
    plt.show()

    print(20 * "-")
    print("Additional info:")
    print(f"Counter: {record[-1]}")
    if url_main:
        try:
            map_id = record[2].split("_")[-1].split(".")[0]
            url = f"{url_main}/{map_id}"
            # stream=True so we don't download the whole page, only check if the page exists
            response = requests.get(url, stream=True)
            assert response.status_code < 400
            print()
            print(f"URL: {url}")
        except:
            url = False
            pass


# -------- prepare_data
def prepare_data(
    df,
    col_names=["image_path", "parent_id"],
    annotation_set="001",
    label_col_name="label",
    redo=False,
    random_state=None,
    num_samples=100,
):
    """Prepare data for annotations

    Parameters
    ----------
    df : DataFrame
        Dataframe which contains information about the patches to be annotated.
    col_names : list, optional
        Column names to be used in annotations, by default ["image_path", "parent_id"]
    annotation_set : str, optional
        Annotation set to be used, by default "001"
    label_col_name : str, optional
        Column name in which labels are stored, by default "label"
    redo : bool, optional
        If True, redo the annotations, by default False
    random_state : int or None, optional
        Random state to be used for sampling dataframe and selecting images to annotate, by default None
    num_samples : int, optional
        Number of samples to annotate, by default 100

    Returns
    -------
    list
        Data - for each image: [image ID, image path, parent ID, annotation set, row]
    """

    if (label_col_name in list(df.columns)) and (not redo):
        print(
            f"Number of already annotated images: {len(df[~df[label_col_name].isnull()])}"
        )
        # only annotate those patches that have not been already annotated
        df = df[df[label_col_name].isnull()]
        print(f"Number of un-annotated images: {len(df)}")
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
            if random_state is None:
                random_state=random.randint(0, 1e6)

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


# -------- annotation_interface
def annotation_interface(
    data,
    list_labels,
    list_colors=["red", "green", "blue", "green"],
    annotation_set="001",
    method="ipyannotate",
    list_shortcuts=None,
):
    """Setup the annotation interface

    Parameters
    ----------
    data : list
        List of patches to be annotated
    list_labels : list
       List of labels
    list_colors : list, optional
        List of colours, by default ["red", "green", "blue", "green"]
    annotation_set : str, optional
        Annotation set, specified in the yaml file, by default "001"
    method : str, optional
        Method to annotate the patches.
        Currently only ipyannotate is supported, by default "ipyannotate"
    list_shortcuts : list or None, optional
        List of shortcuts, by default None

    Returns
    -------
   ipyannotate.annotation.Annotation
        Annotation, output from annotation tool
    """

    if method == "ipyannotate":
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
    else:
        sys.exit(
            f"method: {method} is not implemented. Currently, we support: ipyannotate"
        )


# -------- prepare_annotation
def prepare_annotation(
    userID,
    annotation_tasks_file,
    task,
    custom_labels=[],
    annotation_set="001",
    redo_annotation=False,
    patch_paths=False,
    parent_paths=False,
    tree_level="child",
    sortby=None,
    min_alpha_channel=None,
    min_mean_pixel=None,
    max_mean_pixel=None,
    context_image=False,
    xoffset=500,
    yoffset=500,
    urlmain="https://maps.nls.uk/view/",
    random_state=None,
    list_shortcuts=None,
):
    """Prepare annotations

    Parameters
    ----------
    userID : str
        Unique user ID, used in naming of output file
    annotation_tasks_file : str
        Path to the yaml file which contains tasks/paths/etc. information
    task : str
        Name of the task as defined in the yaml file (annotation_tasks_file)
    custom_labels : list, optional
        Custom labels to be used, by default []
    annotation_set : str
        Name of the annotation set as defined in the yaml file (annotation_tasks_file), by default "001"
    redo_annotation : bool, optional
        If True, redo annotations, by default False
    patch_paths : str or False, optional
        Path to patches when using custom labels (custom_labels).
        If False, information is read from the yaml file (annotation_tasks_file), by default False
    parent_paths : str or False, optional
        Path to parent images when using custom labels (custom_labels).
        If False, information is read from the yaml file (annotation_tasks_file), by default False
    tree_level : str, optional
        Tree level, choices between "parent" or "child", by default "child"
    sortby : str or None, optional
        Method to sort patches, choices between "mean" or None, by default None
    min_alpha_channel : float or None, optional
        If specified, only annotate images with mean alpha values greater than this value, by default None
    min_mean_pixel : float or None, optional
        If specified, only annotate images with mean RGB values greater than this value, by default None
    max_mean_pixel : float or None, optional
        If specified, only annotate images with mean alpha values less than this value, by default None
    context_image : bool, optional
        If True, show a context image behind patch to be annotated, by default False
    xoffset : int, optional
        x offset for context image, by default 500
    yoffset : int, optional
        y offset for context image, by default 500
    urlmain : str or None, optional
        The URL (in the form: urlmain/{map_id}) of the parent image to be shown when annotating, by default "https://maps.nls.uk/view/"
        If None, no URL will be shown
    random_state : int or None, optional
        Random state to be used for sampling dataframe and selecting images to annotate, by default None
    list_shortcuts : list or None, optional
        List of shortcuts, by default None

    Returns
    -------
    ipyannotate.annotation.Annotation
        Annotation, output from annotation tool
    """

    # Specify global variables so they can be used in display_record function
    global annotation_tasks
    global x_offset
    global y_offset
    global url_main
    global treelevel
    global contextimage

    # Note: it is not possible to define global variable + args with the same names
    #       here, we read xoffset and yoffset, assign them to two global variables
    #       these global variables will then be used in display_record
    x_offset = xoffset
    y_offset = yoffset
    url_main = urlmain
    treelevel = tree_level
    contextimage = context_image

    with open(annotation_tasks_file) as annot_file_fio:
        annotation_tasks = yaml.load(annot_file_fio, Loader=yaml.FullLoader)

    if not annotation_set in annotation_tasks["paths"].keys():
        raise ValueError(
            f"{annotation_set} could not be found in {annotation_tasks_file}"
        )
    else:
        if tree_level == "child":
            patch_paths = annotation_tasks["paths"][annotation_set]["patch_paths"]
        parent_paths = os.path.join(
            annotation_tasks["paths"][annotation_set]["parent_paths"]
        )
        annot_file = os.path.join(
            annotation_tasks["paths"][annotation_set]["annot_dir"],
            f"{task}_#{userID}#.csv",
        )

    if not task in annotation_tasks["tasks"].keys():
        if custom_labels == []:
            raise ValueError(
                f"Task: {task} could not be found and custom_labels == []."
            )
        list_labels = custom_labels
    else:
        list_labels = annotation_tasks["tasks"][task]["labels"]

    if tree_level == "child":
        # specify the path of patches and the parent images
        mymaps = load_patches(patch_paths=patch_paths, parent_paths=parent_paths)
        if os.path.isfile(annot_file):
            mymaps.add_metadata(
                metadata=annot_file, index_col=-1, delimiter=",", tree_level=tree_level
            )

        # Calculate mean before converting to pandas so the dataframe contains information about mean pixel intensity
        if (
            sortby == "mean"
            or isinstance(min_alpha_channel, float)
            or isinstance(min_mean_pixel, float)
            or isinstance(max_mean_pixel, float)
        ):
            mymaps.calc_pixel_stats(calc_std=False)

        # convert images to dataframe
        parents_df, patch_df = mymaps.convertImages(fmt="dataframe")

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

        col_names = ["image_path", "parent_id"]
    else:
        mymaps = loader(path_images=parent_paths)
        if os.path.isfile(annot_file):
            mymaps.add_metadata(
                metadata=annot_file, index_col=-1, delimiter=",", tree_level=tree_level
            )
        # convert images to dataframe
        patch_df, _ = mymaps.convertImages(fmt="dataframe")
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
        )
        return annotation


# -------- save_annotation
def save_annotation(annotation, userID, annotation_tasks_file, task, annotation_set):
    """Save annotation results

    Parameters
    ----------
    annotation : ipyannotate.annotation.Annotation
        Annotation output from annotation tool
    userID : str
        Unique user ID, used in naming of output file
    annotation_tasks_file : str
        Path to the yaml file which contains tasks/paths/etc. information
    task : str
        Name of the task as defined in the yaml file (annotation_tasks_file)
    annotation_set : str
        Name of the annotation set as defined in the yaml file (annotation_tasks_file)
    """

    with open(annotation_tasks_file) as annot_file_fio:
        annotation_tasks = yaml.load(annot_file_fio, Loader=yaml.FullLoader)

    if not annotation_set in annotation_tasks["paths"].keys():
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
        if annotation.tasks[i].value != None:
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
