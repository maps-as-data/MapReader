#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from PIL import Image, ImageOps
from typing import Optional, Dict, Callable, Tuple, Any, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Import parhugin
try:
    from parhugin import multiFunc

    parhugin_installed = True
except ImportError:
    print(
        "[WARNING] parhugin (https://github.com/kasra-hosseini/parhugin) is not installed, continue without it."  # noqa
    )
    parhugin_installed = False


class patchTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patchframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        label_col: Optional[str] = "label",
        convert2: Optional[str] = "RGB",
        input_col: Optional[int] = 0,
    ):
        """
        A PyTorch Dataset class for loading image patches from a DataFrame.

        Parameters
        ----------
        patchframe : pandas.DataFrame
            DataFrame containing the paths to image patches and their labels.
        transform : callable, optional
            A callable object (a torchvision transform) that takes in an image
            and performs image transformations. Default is None.
        label_col : str, optional
            The name of the column containing the image labels. Default is
            "label".
        convert2 : str, optional
            The color format to convert the image to. Default is "RGB".
        input_col : int, optional
            The index of the column in the DataFrame containing the image
            paths. Default is 0.

        Attributes
        ----------
        patchframe : pandas.DataFrame
            DataFrame containing the paths to image patches and their labels.
        label_col : str
            The name of the column containing the image labels.
        convert2 : str
            The color format to convert the image to.
        input_col : int
            The index of the column in the DataFrame containing the image
            paths.
        uniq_labels : list
            The unique labels in the label column of the patchframe DataFrame.
        transform : callable
            A callable object (a torchvision transform) that takes in an image
            and performs image transformations.

        Methods
        -------
        __len__()
            Returns the length of the dataset.
        __getitem__(idx)
            Retrieves the image and label at the given index in the dataset.
        return_orig_image(idx)
            Retrieves the original image at the given index in the dataset.
        _default_transform(t_type, resize2)
            Returns a dictionary containing the default image transformations
            for the train and validation sets.
        """
        self.patchframe = patchframe
        self.label_col = label_col
        self.convert2 = convert2
        self.input_col = input_col

        if self.label_col in self.patchframe.columns.tolist():
            self.uniq_labels = self.patchframe[self.label_col].unique().tolist()
        else:
            self.uniq_labels = "NS"

        if transform in ["train", "val"]:
            self.transform = self._default_transform(transform)
        elif transform is None:
            raise ValueError("transform argument is not set.")
        else:
            self.transform = transform

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.patchframe)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        """
        Return the image and its label at the given index in the dataset.

        Parameters
        ----------
        idx : int or torch.Tensor
            Index or indices of the desired image.

        Returns
        -------
        tuple
            A tuple containing the transformed image and its label (if
            available). The label is -1 if it is not present in the DataFrame.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])
        image = Image.open(img_path).convert(self.convert2)

        image = self.transform(image)

        if self.label_col in self.patchframe.iloc[idx].keys():
            image_label = self.patchframe.iloc[idx][self.label_col]
        else:
            image_label = -1

        return image, image_label

    def return_orig_image(self, idx: Union[int, torch.Tensor]) -> Image:
        """
        Return the original image associated with the given index.

        Parameters
        ----------
        idx : int or Tensor
            The index of the desired image, or a Tensor containing the index.

        Returns
        -------
        PIL.Image.Image
            The original image associated with the given index.

        Notes
        -----
        This method returns the original image associated with the given index
        by loading the image file using the file path stored in the
        ``input_col`` column of the ``patchframe`` DataFrame at the given
        index. The loaded image is then converted to the format specified by
        the ``convert2`` attribute of the object. The resulting
        ``PIL.Image.Image`` object is returned.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])

        image = Image.open(img_path).convert(self.convert2)

        return image

    def _default_transform(
        self, t_type: Optional[str] = "train", resize2: Optional[int] = 224
    ) -> Dict:
        """
        Returns a dictionary containing the default image transformations for
        the train and validation sets.

        Parameters
        ----------
        t_type : str, optional
            The type of transformation to return. Either "train" or "val".
            Default is "train".
        resize2 : int, optional
            The size in pixels to resize the image to. Default is 224.

        Returns
        -------
        dict
            A dictionary containing the default image transformations for the
            specified type.

        Raises
        ------
        None
        """
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize(resize2),
                    transforms.RandomApply(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            # transforms.ColorJitter(brightness=0.3, contrast=0.3), # noqa
                        ],
                        p=0.5,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(normalize_mean, normalize_std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize2),
                    transforms.ToTensor(),
                    transforms.Normalize(normalize_mean, normalize_std),
                ]
            ),
        }
        return data_transforms[t_type]


# --- Dataset that returns an image, its context and its label
class patchContextDataset(Dataset):
    def __init__(
        self,
        patchframe: pd.DataFrame,
        transform1: Optional[str] = None,
        transform2: Optional[str] = None,
        label_col: Optional[str] = "label",
        convert2: Optional[str] = "RGB",
        input_col: Optional[int] = 0,
        context_save_path: Optional[str] = "./maps/maps_context",
        create_context: Optional[bool] = False,
        par_path: Optional[str] = "./maps",
        x_offset: Optional[float] = 1.0,
        y_offset: Optional[float] = 1.0,
        slice_method: Optional[str] = "scale",
    ):
        """
        A PyTorch Dataset class for loading contextual information about image
        patches from a DataFrame.

        Parameters
        ----------
        patchframe : pandas.DataFrame
            A pandas DataFrame with columns representing image paths, labels,
            and object bounding boxes.
        transform1 : str, optional
            Optional Torchvision transform to be applied to input images.
            Either "train" or "val". Default is None.
        transform2 : str, optional
            Optional Torchvision transform to be applied to target images.
            Either "train" or "val". Default is None.
        label_col : str, optional
            The name of the column in ``patchframe`` that contains the label
            information. Default is "label".
        convert2 : str, optional
            The color space of the images. Default is "RGB".
        input_col : int, optional
            The index of the column in ``patchframe`` that contains the input
            image paths. Default is 0.
        context_save_path : str, optional
            The path to save context maps to. Default is "./maps/maps_context".
        create_context : bool, optional
            Whether or not to create context maps. Default is False.
        par_path : str, optional
            The path to the directory containing parent images. Default is
            "./maps".
        x_offset : float, optional
            The size of the horizontal offset around objects, as a fraction of
            the image width. Default is 1.0.
        y_offset : float, optional
            The size of the vertical offset around objects, as a fraction of
            the image height. Default is 1.0.
        slice_method : str, optional
            The method used to slice images. Either "scale" or "absolute".
            Default is "scale".

        Attributes
        ----------
        patchframe : pandas.DataFrame
            A pandas DataFrame with columns representing image paths, labels,
            and object bounding boxes.
        label_col : str
            The name of the column in ``patchframe`` that contains the label
            information.
        convert2 : str
            The color space of the images.
        input_col : int
            The index of the column in ``patchframe`` that contains the input
            image paths.
        par_path : str
            The path to the directory containing parent images.
        x_offset : float
            The size of the horizontal offset around objects, as a fraction of
            the image width.
        y_offset : float
            The size of the vertical offset around objects, as a fraction of
            the image height.
        slice_method : str
            The method used to slice images.
        create_context : bool
            Whether or not to create context maps.
        context_save_path : str
            The path to save context maps to.
        uniq_labels : list or str
            The unique labels in ``label_col``, or "NS" if ``label_col`` not in
            ``patchframe``.

        Methods
        ----------
        __len__()
            Returns the length of the dataset.
        __getitem__(idx)
            Retrieves the patch image, the context image and the label at the
            given index in the dataset.
        save_parents()
            Saves parent images.
        save_parents_idx(idx)
            Saves parent image at index ``idx``.
        return_orig_image(idx)
            Return the original image associated with the given index.
        """
        self.patchframe = patchframe
        self.label_col = label_col
        self.convert2 = convert2
        self.input_col = input_col
        self.par_path = par_path
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.slice_method = slice_method
        self.create_context = create_context

        if not self.create_context:
            self.context_save_path = os.path.abspath(context_save_path)

        if self.label_col in self.patchframe.columns.tolist():
            self.uniq_labels = self.patchframe[self.label_col].unique().tolist()
        else:
            self.uniq_labels = "NS"

        if transform1 in ["train", "val"]:
            self.transform1 = self._default_transform(transform1)
        elif transform1 is None:
            raise ValueError("transform argument is not set.")
        else:
            self.transform1 = transform1

        if transform2 in ["train", "val"]:
            self.transform2 = self._default_transform(transform2)
        elif transform2 is None:
            raise ValueError("transform argument is not set.")
        else:
            self.transform2 = transform2

    def save_parents(
        self,
        num_req_p: Optional[int] = 10,
        sleep_time: Optional[float] = 0.001,
        use_parhugin: Optional[bool] = True,
        par_split: Optional[str] = "#",
        loc_split: Optional[str] = "-",
        overwrite: Optional[bool] = False,
    ) -> None:
        """
        Save parent patches for all patches in the patchframe.

        Parameters
        ----------
        num_req_p : int, optional
            The number of required processors for the job, by default 10.
        sleep_time : float, optional
            The time to wait between jobs, by default 0.001.
        use_parhugin : bool, optional
            Flag indicating whether to use Parhugin to parallelize the job, by
            default True.
        par_split : str, optional
            The string used to separate parent IDs in the patch filename, by
            default "#".
        loc_split : str, optional
            The string used to separate patch location and level in the patch
            filename, by default "-".
        overwrite : bool, optional
            Flag indicating whether to overwrite existing parent files, by
            default False.

        Returns
        -------
        None

        Notes
        -----
        Parhugin is a Python package for parallelizing computations across
        multiple CPU cores. The method uses Parhugin to parallelize the
        computation of saving parent patches to disk. When Parhugin is
        installed and ``use_parhugin`` is set to True, the method parallelizes
        the calling of the ``save_parents_idx`` method and its corresponding
        arguments. If Parhugin is not installed or ``use_parhugin`` is set to
        False, the method executes the loop over patch indices sequentially
        instead.
        """
        if parhugin_installed and use_parhugin:
            myproc = multiFunc(num_req_p=num_req_p, sleep_time=sleep_time)
            list_jobs = []
            for idx in range(len(self.patchframe)):
                list_jobs.append(
                    [
                        self.save_parents_idx,
                        (idx, par_split, loc_split, overwrite),
                    ]
                )

            print(f"Total number of jobs: {len(list_jobs)}")
            # and then adding them to myproc
            myproc.add_list_jobs(list_jobs)
            myproc.run_jobs()
        else:
            for idx in range(len(self.patchframe)):
                self.save_parents_idx(idx)

    def save_parents_idx(
        self,
        idx: int,
        par_split: Optional[str] = "#",
        loc_split: Optional[str] = "-",
        overwrite: Optional[bool] = False,
        return_image: Optional[bool] = False,
    ) -> None:
        """
        Save the parents of a specific patch to the specified location.

        Parameters
        ----------
            idx : int
                Index of the patch in the dataset.
            par_split : str, optional
                Delimiter to split the parent names in the file path. Default
                is "#".
            loc_split : str, optional
                Delimiter to split the location of the patch in the file path.
                Default is "-".
            overwrite : bool, optional
                Whether to overwrite the existing parent files. Default is
                False.

        Raises
        ------
        ValueError
            If the patch is not found in the dataset.

        Returns
        -------
        None
        """
        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])
        img_rd = Image.open(img_path).convert(self.convert2)

        if not return_image:
            os.makedirs(self.context_save_path, exist_ok=True)

            path2save_context = os.path.join(
                self.context_save_path, os.path.basename(img_path)
            )

            if os.path.isfile(path2save_context) and (not overwrite):
                return

        if self.slice_method in ["scale"]:
            # size: (width, height)
            tar_y_offset = int(img_rd.size[1] * self.y_offset)
            tar_x_offset = int(img_rd.size[0] * self.x_offset)
        else:
            tar_y_offset = self.y_offset
            tar_x_offset = self.x_offset

        par_name = os.path.basename(img_path).split(par_split)[1]
        split_path = os.path.basename(img_path).split(loc_split)
        min_x, min_y, max_x, max_y = (
            int(split_path[1]),
            int(split_path[2]),
            int(split_path[3]),
            int(split_path[4]),
        )

        if self.par_path in ["dynamic"]:
            par_path2read = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(img_path))),
                par_name,
            )
        else:
            par_path2read = os.path.join(os.path.abspath(self.par_path), par_name)

        par_img = Image.open(par_path2read).convert(self.convert2)

        min_y_par = max(0, min_y - tar_y_offset)
        min_x_par = max(0, min_x - tar_x_offset)
        max_x_par = min(max_x + tar_x_offset, np.shape(par_img)[1])
        max_y_par = min(max_y + tar_y_offset, np.shape(par_img)[0])

        pad_activate = False
        top_pad = left_pad = right_pad = bottom_pad = 0
        if (min_y - tar_y_offset) < 0:
            top_pad = abs(min_y - tar_y_offset)
            pad_activate = True
        if (min_x - tar_x_offset) < 0:
            left_pad = abs(min_x - tar_x_offset)
            pad_activate = True
        if (max_x + tar_x_offset) > np.shape(par_img)[1]:
            right_pad = max_x + tar_x_offset - np.shape(par_img)[1]
            pad_activate = True
        if (max_y + tar_y_offset) > np.shape(par_img)[0]:
            bottom_pad = max_y + tar_y_offset - np.shape(par_img)[0]
            pad_activate = True

        # par_img = par_img[min_y_par:max_y_par, min_x_par:max_x_par]
        par_img = par_img.crop((min_x_par, min_y_par, max_x_par, max_y_par))

        if pad_activate:
            padding = (left_pad, top_pad, right_pad, bottom_pad)
            par_img = ImageOps.expand(par_img, padding)

        if return_image:
            return par_img
        elif not os.path.isfile(path2save_context):
            par_img.save(path2save_context)

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.patchframe)

    def return_orig_image(self, idx: Union[int, torch.Tensor]) -> Image:
        """
        Return the original image associated with the given index.

        Parameters
        ----------
        idx : int or Tensor
            The index of the desired image, or a Tensor containing the index.

        Returns
        -------
        PIL.Image.Image
            The original image associated with the given index.

        Notes
        -----
        This method returns the original image associated with the given index
        by loading the image file using the file path stored in the
        ``input_col`` column of the ``patchframe`` DataFrame at the given
        index. The loaded image is then converted to the format specified by
        the ``convert2`` attribute of the object. The resulting
        ``PIL.Image.Image`` object is returned.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])

        image = Image.open(img_path).convert(self.convert2)

        return image

    def plot_sample(self, indx: int) -> None:
        """
        Plot a sample patch and its corresponding context from the dataset.

        Parameters
        ----------
        indx : int
            The index of the sample to plot.

        Returns
        -------
        None
            Displays the plot of the sample patch and its corresponding
            context.

        Notes
        -----
        This method plots a sample patch and its corresponding context side-by-
        side in a single figure with two subplots. The figure size is set to
        10in x 5in, and the titles of the subplots are set to "Patch" and
        "Context", respectively. The resulting figure is displayed using
        the ``matplotlib`` library (required).
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(transforms.ToPILImage()(self.__getitem__(indx)[0]))
        plt.title("Patch", size=18)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.imshow(transforms.ToPILImage()(self.__getitem__(indx)[1]))
        plt.title("Context", size=18)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.show()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Retrieves the patch image, the context image and the label at the
        given index in the dataset (``idx``).

        Parameters
        ----------
        idx : int
            The index of the data to retrieve.

        Returns
        -------
        tuple of form (torch.Tensor, torch.Tensor, label)
            A tuple of three elements, where the first element is a tensor
            containing the patch image, the second element is a tensor
            containing the context image, and the third element is an integer
            label.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.patchframe.iloc[idx, self.input_col])
        image = Image.open(img_path).convert(self.convert2)

        if self.create_context:
            context_img = self.save_parents_idx(idx, return_image=True)
        else:
            context_img = Image.open(
                os.path.join(self.context_save_path, os.path.basename(img_path))
            ).convert(self.convert2)

        image = self.transform1(image)
        context_img = self.transform2(context_img)

        if self.label_col in self.patchframe.iloc[idx].keys():
            image_label = self.patchframe.iloc[idx][self.label_col]
        else:
            image_label = -1

        return image, context_img, image_label
