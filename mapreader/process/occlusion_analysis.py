#!/usr/bin/env python
from __future__ import annotations

import os
import pathlib
import re

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm.auto import tqdm

from mapreader.utils.load_frames import load_from_csv, load_from_geojson


class OcclusionAnalyzer:
    """A class for carrying out occlusion analysis on patches.

    Parameters
    ----------
    patch_df : pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path
        The DataFrame containing patches and predictions.
    model : str or nn.Module
            The PyTorch model to add to the object.
            If a string, this should be the path to a model checkpoint.
    transform : str or callable
        The transform to apply to the patches. Options of "default" or a torchvision transform.
        Default transform
    delimiter : str
        The delimiter used in the patch_df csv file. By default, ",".
    device : str
        The device to use. By default, "default".
    """

    def __init__(
        self,
        patch_df: pd.DataFrame | gpd.GeoDataFrame | str | pathlib.Path,
        model: str | nn.Module,
        transform: str | callable = "default",
        delimiter: str = ",",
        device: str = "default",
    ):
        if isinstance(patch_df, pd.DataFrame):
            self.patch_df = patch_df

        elif isinstance(patch_df, (str, pathlib.Path)):
            if re.search(r"\..?sv$", str(patch_df)):
                print(f'[INFO] Reading "{patch_df}"')
                patch_df = load_from_csv(
                    patch_df,
                    delimiter=delimiter,
                )
                self.patch_df = patch_df
            elif re.search(r"\..*?json$", str(patch_df)):
                patch_df = load_from_geojson(patch_df)
                self.patch_df = patch_df
            else:
                raise ValueError(
                    "[ERROR] ``patch_df`` must be a path to a CSV/TSV/etc or geojson file, a pandas DataFrame or a geopandas GeoDataFrame."
                )

        else:
            raise ValueError(
                "[ERROR] Please pass ``patch_df`` as a string (path to csv file) or pd.DataFrame."
            )

        if patch_df.index.name != "image_id" and "image_id" in patch_df.columns:
            patch_df.set_index("image_id", drop=True, inplace=True)

        if any(col not in patch_df.columns for col in ["predicted_label", "pred"]):
            raise ValueError(
                "[ERROR] The patch dataframe should contain predicted labels."
            )

        if device == "default":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if isinstance(model, str):
            self.model = self._load_model(model)
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise ValueError("[ERROR] Please pass ``model`` as a string or nn.Module.")
        # set model to evaluation mode
        self.model.eval()

        if transform == "default":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            self.transform = transform

        self.loss_fn = None

    def __len__(self):
        return len(self.patch_df)

    def _load_model(self, model_path: str) -> nn.Module:
        model = torch.load(model_path, map_location=self.device)
        return model

    def add_loss_fn(
        self, loss_fn: str | nn.modules.loss._Loss | None = "cross entropy"
    ) -> None:
        """
        Add a loss function to the object.

        Parameters
        ----------
        loss_fn : str or torch.nn.modules.loss._Loss
            The loss function to use.
            Can be a string or a torch.nn loss function. Accepted string values are "cross entropy" or "ce" (cross-entropy), "bce" (binary cross-entropy) and "mse" (mean squared error).
            By default, "cross entropy" is used.
        """
        if isinstance(loss_fn, str):
            if loss_fn in ["cross entropy", "ce", "cross_entropy", "cross-entropy"]:
                loss_fn = nn.CrossEntropyLoss()
            elif loss_fn in [
                "bce",
                "binary_cross_entropy",
                "binary cross entropy",
                "binary cross-entropy",
            ]:
                loss_fn = nn.BCELoss()
            elif loss_fn in [
                "mse",
                "mean_square_error",
                "mean_squared_error",
                "mean squared error",
            ]:
                loss_fn = nn.MSELoss()
            else:
                raise NotImplementedError(
                    '[ERROR] At present, if passing ``loss_fn`` as a string, the loss function can only be "cross entropy" or "ce" (cross-entropy), "bce" (binary cross-entropy) or "mse" (mean squared error).'
                )

            print(f'[INFO] Using "{loss_fn}" as loss function.')

        elif not isinstance(loss_fn, nn.modules.loss._Loss):
            raise ValueError(
                '[ERROR] Please pass ``loss_fn`` as a string ("cross entropy", "bce" or "mse") or torch.nn loss function (see https://pytorch.org/docs/stable/nn.html).'
            )

        self.loss_fn = loss_fn

    def run_occlusion(
        self,
        label: str,
        sample_size: int = 10,
        save: bool = False,
        path_save: str = "./occlusion_analysis/",
        block_size: int = 14,
    ):
        """Run occlusion analysis on a sample of patches for a given label.

        Parameters
        ----------
        label : str
            The label to run the analysis on.
        sample_size : int
            The number of patches to run the analysis on.
            By default, 10.
        save : bool
            Whether to save the occlusion analysis images. By default, False.
        path_save : str
            The path to save the occlusion analysis images to. By default, "./occlusion_analysis/".
        block_size : int
            The size of the occlusion block. By default, 14.
        """
        if self.loss_fn is None:
            raise ValueError(
                "[ERROR] Please first run ``add_loss_fn`` to set your loss function."
            )

        patches = self.patch_df[self.patch_df["predicted_label"] == label]

        if len(patches) == 0:
            raise ValueError(f'[ERROR] No patches with label "{label}" found.')

        if len(patches) < sample_size:
            sample_size = len(patches)
            print(
                f"[INFO] Sample size reduced to {sample_size} due to limited patches."
            )

        patches = patches.sample(sample_size)
        patch_ids = patches.index.tolist()

        if save:
            if not os.path.exists(path_save):
                os.makedirs(path_save)

            for patch_id in tqdm(patch_ids):
                self._generate_area_importance_heatmap_with_occlusions(
                    patch_id, save=save, path_save=path_save, block_size=block_size
                )

        else:
            results = []

            for patch_id in tqdm(patch_ids):
                results.append(
                    self._generate_area_importance_heatmap_with_occlusions(
                        patch_id, save=save, path_save=path_save, block_size=block_size
                    )
                )
            return results

    def _preprocess_image(self, image: Image) -> torch.Tensor:
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    def _generate_area_importance_heatmap_with_occlusions(
        self, image_id: str, save: bool, path_save: str, block_size: int = 14
    ):
        """
        Generates an area importance heatmap with occlusions for a given image.
        If save is True, the heatmap will be saved to the path specified in path_save, otherwise the image will be returned.

        Parameters
        ----------
        image_id : str
            The ID of the image to generate the heatmap for.
        save: bool
            Whether to save the heatmap. By default, False.
        path_save : str
            The path to save the heatmap to.
        block_size : int
            The size of the occlusion block.

        Returns
        -------
        combined_image : PIL.Image
            The combined image with the original image, heatmap and overlayed image.
        """

        image_path = self.patch_df.loc[image_id, "image_path"]
        image = Image.open(image_path).convert("RGB")

        # preprocess the image and get the real prediction
        image_tensor = self._preprocess_image(image)
        gt_prediction = self.model(image_tensor)

        image = np.array(image)
        height, width, _ = image.shape

        # calculate the number of blocks that fit in the image
        columns = width // block_size
        rows = height // block_size

        heatmap = np.zeros((columns, rows))

        for row in range(rows):
            for column in range(columns):
                # add occlusion to the image
                x = column * block_size
                y = row * block_size

                top = int(y)
                left = int(x)
                right = left + block_size
                bottom = top + block_size

                occluded_image = np.copy(image)
                noise = np.random.rand(block_size, block_size, 3) * 255
                occluded_image[int(top) : int(bottom), int(left) : int(right)] = noise
                occluded_image = Image.fromarray(occluded_image)

                # preprocess the occluded image and get the prediction
                image_tensor = self._preprocess_image(occluded_image)
                prediction = self.model(image_tensor)
                loss = round(float(self.loss_fn(prediction, gt_prediction)), 4)

                # store the loss in the heatmap
                heatmap[row, column] = loss

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = heatmap * 255
        heatmap = np.uint8(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlayed = cv2.addWeighted(image, 1, heatmap, 0.75, 0)
        combined = np.concatenate((image, heatmap, overlayed), axis=1)
        combined_image = Image.fromarray(combined)

        if save:
            combined_image.save(os.path.join(path_save, image_id))
        else:
            return combined_image
