#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
from typing import Union, Optional, Dict, Callable
from PIL import Image
from decimal import Decimal

from .datasets import PatchDataset

class AnnotationsLoader():
    def __init__(self):
        self.annotations = pd.DataFrame()
        self.reviewed = pd.DataFrame()
        self.id_col = None
        self.patch_paths_col = None
        self.label_col = None
            
    def load(
        self,
        annotations: Union[str, pd.DataFrame],
        delimiter: Optional[str] = "\t",
        id_col: Optional[str] = "image_id",
        patch_paths_col: Optional[str] = "image_path",
        label_col: Optional[str] = "label",
        append: Optional[bool] = True,
        scramble_frame: Optional[bool] = False,
        reset_index: Optional[bool] = False,
        ):
        
        if not isinstance(annotations, (str, pd.DataFrame)):
            raise ValueError("[ERROR] Please pass ``annotations`` as a string (path to csv file) or pd.DataFrame.")
    
        if not self.id_col:
            self.id_col = id_col
        elif self.id_col != id_col:
            print(f'[WARNING] ID column was previously "{self.id_col}, but will now be set to {id_col}.')
        
        if not self.patch_paths_col:
            self.patch_paths_col = patch_paths_col
        elif self.patch_paths_col != patch_paths_col:
            print(f'[WARNING] Patch paths column was previously "{self.patch_paths_col}, but will now be set to {patch_paths_col}.')

        if not self.label_col:
            self.label_col = label_col
        elif self.label_col != label_col:
            print(f'[WARNING] Label column was previously "{self.label_col}, but will now be set to {label_col}.')

        if isinstance(annotations, str):
            annotations = self._load_annotations_csv(annotations, delimiter, scramble_frame, reset_index)
        
        annotations = annotations.astype({self.label_col:str}) # ensure labels are interpreted as strings 

        if append:
            self.annotations = self.annotations.append(annotations)
        else:
            self.annotations = annotations

        print(self)

    def _load_annotations_csv(self, annotations, delimiter, scramble_frame, reset_index):
        if os.path.isfile(annotations):
            print(f'[INFO] Reading "{annotations}"')
            annotations = pd.read_csv(annotations, sep=delimiter)
        else:
            raise ValueError(f'[ERROR] "{annotations}" cannot be found.')
                
        if scramble_frame:
            annotations = annotations.sample(frac=1)
        if reset_index:
            annotations.reset_index(drop=True, inplace=True)
            
        annotations.drop_duplicates(subset=self.id_col, inplace=True, keep="first")
        return annotations
        
    def show_patch(self, patch_id: str) -> None:
        """
        Display a patch and its label.

        Parameters
        ----------
        patch_id : str
            The image ID of the patch to show.

        Returns
        -------
        None
        """

        if len(self.annotations) == 0:
            raise ValueError("[ERROR] No annotations loaded.")

        patch_row = self.annotations[self.annotations[self.id_col]==patch_id]
        patch_path = patch_row[self.patch_paths_col].values[0]
        patch_label = patch_row[self.label_col].values[0]
        img = Image.open(patch_path)
    
        plt.imshow(img)
        plt.axis("off")
        plt.title(patch_label)
        plt.show()

    def review_labels(
        self,
        label_to_review: Optional[str] = None,
        chunks: Optional[int] = 8 * 6,
        num_cols: Optional[int] = 8,
        exclude_df: Optional[pd.DataFrame] = None,
        include_df: Optional[pd.DataFrame] = None,
        deduplicate_col: Optional[str] = "image_id",
    ) -> None:
        """
        Perform image review on annotations and update labels for a given
        label or all labels.

        Parameters
        ----------
        label_to_review : str, optional
            The target label to review. If not provided, all labels will be
            reviewed, by default ``None``.
        chunks : int, optional
            The number of images to display at a time, by default ``8 * 6``.
        num_cols : int, optional
            The number of columns in the display, by default ``8``.
        exclude_df : pandas.DataFrame, optional
            A DataFrame of images to exclude from review, by default ``None``.
        include_df : pandas.DataFrame, optional
            A DataFrame of images to include for review, by default ``None``.
        deduplicate_col : str, optional
            The column to use for deduplicating reviewed images, by default
            ``"image_id"``.

        Returns
        -------
        None

        Notes
        ------
        This method reviews images with their corresponding labels and allows
        the user to change the label for each image. The updated labels are
        saved in both the annotations and reviewed DataFrames. If
        ``exclude_df`` is provided, images with ``image_path`` in
        ``exclude_df["image_path"]`` are skipped in the review process. If
        ``include_df`` is provided, only images with ``image_path`` in
        ``include_df["image_path"]`` are reviewed. The reviewed DataFrame is
        deduplicated based on the ``deduplicate_col``.
        """
        if label_to_review:
            annots2review = self.annotations[
                self.annotations[self.label_col] == label_to_review
            ]
            annots2review.reset_index(inplace=True, drop=True)
        else:
            annots2review = self.annotations
            annots2review.reset_index(inplace=True, drop=True)

        if exclude_df is not None:
            if isinstance(exclude_df, pd.DataFrame):
                merged_df = pd.merge(annots2review, exclude_df, how="left", indicator=True)
                annots2review = merged_df[merged_df["_merge"]=="left_only"].drop(columns="_merge")
                annots2review.reset_index(inplace=True, drop=True)
            else:
                raise ValueError("[ERROR] ``exclude_df`` must be a pandas dataframe.")
    
        if include_df is not None:
            if isinstance(include_df, pd.DataFrame):
                annots2review = pd.merge(annots2review, include_df, how="right")
                annots2review.reset_index(inplace=True, drop=True)
            else:
                raise ValueError("[ERROR] ``include_df`` must be a pandas dataframe.")

        image_idx = 0
        while image_idx < len(annots2review):
            print('[INFO] Type "exit", "end" or "stop" to exit.')
            print(f"[INFO] Showing {image_idx}-{image_idx+chunks} out of {len(annots2review)}."  # noqa
            )
            plt.figure(figsize=(num_cols*3, (chunks // num_cols)*3))
            counter = 1
            iter_ids = []
            while (counter <= chunks) and (image_idx < len(annots2review)):
                # The first term is just a ceiling division, equivalent to:
                # from math import ceil
                # int(ceil(chunks / num_cols))
                plt.subplot((chunks // num_cols), num_cols, counter)
                patch_path = annots2review.iloc[image_idx][self.patch_paths_col]
                img = Image.open(patch_path)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.title(
                    f"{annots2review.iloc[image_idx][self.label_col]} | id: {annots2review.iloc[image_idx].name}"  # noqa
                )
                iter_ids.append(annots2review.iloc[image_idx].name)
                # Add to reviewed
                self.reviewed = self.reviewed.append(annots2review.iloc[image_idx])
                try:
                    self.reviewed.drop_duplicates(subset=[deduplicate_col])
                except Exception:
                    pass
                counter += 1
                image_idx += 1
            plt.show()

            print(f"[INFO] IDs of current patches: {iter_ids}")
            q = "\nEnter IDs, comma separated (or press enter to continue): "
            user_input_ids = input(q)

            while user_input_ids.strip().lower() not in [
                "",
                "exit",
                "end",
                "stop",
            ]:
                list_input_ids = user_input_ids.split(",")
                print(f"[INFO] Options for labels (or create a new label):{list(self.annotations[self.label_col].unique())}")
                input_label = input("Enter new label:  ")

                for input_id in list_input_ids:
                    input_id = int(input_id)
                    # Change both annotations and reviewed
                    self.annotations.loc[input_id, self.label_col] = input_label
                    self.reviewed.loc[input_id, self.label_col] = input_label
                    print(f'[INFO] Image {input_id} has been relabelled as "{input_label}"')

                user_input_ids = input(q)

            if user_input_ids.lower() in ["exit", "end", "stop"]:
                break

        print("[INFO] Exited.")

    def show_sample(
        self, label_to_show: str, num_samples: Optional[int] = 9
    ) -> None:
        """Show a random sample of images with the specified label (tar_label).

        Parameters
        ----------
        label_to_show : str, optional
            The label of the images to show.
        num_sample : int, optional
            The number of images to show. If ``None``, all images with the
            specified label will be shown. Default is ``10``.

        Returns
        -------
        None
        """
        if len(self.annotations) == 0:
            raise ValueError("[ERROR] No annotations loaded.")

        annot2plot = self.annotations[self.annotations[self.label_col] == label_to_show]
        annot2plot = annot2plot.sample(frac=1)
        annot2plot.reset_index(drop=True, inplace=True)

        num_samples = min(len(annot2plot), num_samples)

        plt.figure(figsize=(8, num_samples))
        for i in range(num_samples):
            plt.subplot(int(num_samples / 2.0), 3, i + 1)
            patch_path = annot2plot.iloc[i][self.patch_paths_col]
            img = Image.open(patch_path)
            plt.imshow(img)
            plt.axis("off")
            plt.title(annot2plot.iloc[i][self.label_col])
        plt.show()

    def split_annotations(
        self,
        frac_train: Optional[Union[str, float]] = 0.70,
        frac_val: Optional[Union[str, float]] = 0.15,
        frac_test: Optional[Union[str, float]] = 0.15,
        random_state: Optional[int] = 1364,
        train_transform: Optional[Union[str, Callable]] = "train",
        val_transform: Optional[Union[str, Callable]] = "val",
        test_transform: Optional[Union[str, Callable]] = "test",
    ) -> tuple:
        """
        Splits the dataset into three subsets: training, validation, and test
        sets (DataFrames).

        Parameters
        ----------
        frac_train : float, optional
            Fraction of the dataset to be used for training. The default is
            ``0.70``.
        frac_val : float, optional
            Fraction of the dataset to be used for validation. The default is
            ``0.15``.
        frac_test : float, optional
            Fraction of the dataset to be used for testing. The default is
            ``0.15``.
        random_state : int, optional
            Random seed to ensure reproducibility. The default is ``1364``.

        Raises
        ------
        ValueError
            If the sum of fractions of training, validation and test sets does
            not add up to 1.

        Returns
        -------
        Tuple
            A tuple containing the PatchDatasets. 

        Notes
        -----
        As well as returning the PatchDatasets, this method also saves the dataframes as ``self.train``, ``self.val``, ``self.test``.

        Following fractional ratios provided by the user, where each subset is
        stratified by the values in a specific column (that is, each subset has
        the same relative frequency of the values in the column). It performs
        this splitting by running ``train_test_split()`` twice.
        """

        for frac in [frac_train, frac_val, frac_test]:
            frac = Decimal(str(frac))

        if sum([frac_train + frac_val + frac_test]) != 1:
            raise ValueError(
                f"[ERROR] ``frac_train`` ({frac_train}), ``frac_val`` ({frac_val}) and ``frac_test`` ({frac_test}) do not add up to 1."
            ) # noqa

        labels = self.annotations[self.label_col]

        # Split original dataframe into train and temp (val+test) dataframes.
        df_train, df_temp, _, labels_temp = train_test_split(
            self.annotations,
            labels,
            stratify=labels,
            test_size=(1.0 - frac_train),
            random_state=random_state,
        )

        if frac_test != 0:
            # Split the temp dataframe into val and test dataframes.
            relative_frac_test = frac_test / (frac_val + frac_test)
            df_val, df_test, _, _ = train_test_split(
                df_temp,
                labels_temp,
                stratify=labels_temp,
                test_size=relative_frac_test,
                random_state=random_state,
            )
            assert len(self.annotations) == len(df_train) + len(df_val) + len(df_test)

        else:
            df_val = labels_temp
            assert len(self.annotations) == len(df_train) + len(df_val)
        
        self.train = df_train
        self.val = df_val

        train_dataset = PatchDataset(self.train, train_transform, self.patch_paths_col, self.label_col, )
        val_dataset = PatchDataset(self.val, val_transform, self.patch_paths_col, self.label_col)

        print(f"[INFO] Number of annotations in each set:\n\
    - Train:        {len(train_dataset)}\n\
    - Validate:     {len(val_dataset)}")
        
        if df_test is not None:
            self.test = df_test
            test_dataset = PatchDataset(self.test, test_transform, self.patch_paths_col, self.label_col)     
            print(f"    - Test:         {len(test_dataset)}")
            return train_dataset, val_dataset, test_dataset

        return train_dataset, val_dataset
    
    def __str__(self):
        print(f"[INFO] Number of annotations:   {len(self.annotations)}\n")
        if len(self.annotations) > 0:
            value_counts = self.annotations[self.label_col].value_counts()
            print(f'[INFO] Number of instances of each label (from column "{self.label_col}"):')
            for label, count in value_counts.items():
                print(f"    - {label}:      {count}")
        return ""
