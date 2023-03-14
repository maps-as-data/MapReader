#!/usr/bin/env python
# -*- coding: utf-8 -*-

from glob import glob
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
from typing import Union, Optional


class loadAnnotations:
    def __init__(self):
        self.annotations = pd.DataFrame()
        self.reviewed = pd.DataFrame()
        self.col_path = None

    def load_all(self, csv_paths: str, **kwds) -> None:
        """
        Load multiple CSV files into the class instance using the `load`
        method.

        Parameters
        ----------
        csv_paths : str
            The file path pattern to match CSV files to load.
        **kwds : dict
            Additional keyword arguments to pass to the `load` method.

        Returns
        -------
        None
        """
        for csv_path in glob(csv_paths):
            self.load(csv_path=csv_path, append=True, **kwds)

    def load(
        self,
        csv_path: str,
        path2dir: Optional[str] = None,
        col_path: Optional[str] = "image_id",
        keep_these_cols: Optional[bool] = False,
        append: Optional[bool] = True,
        col_label: Optional[str] = "label",
        shuffle_rows: Optional[bool] = True,
        reset_index: Optional[bool] = True,
        random_state: Optional[int] = 1234,
    ) -> None:
        """
        Read and append an annotation file to the class instance's annotations
        DataFrame.

        Parameters
        ----------
        csv_path : str
            Path to an annotation file in CSV format.
        path2dir : str, optional
            Update the `col_path` column by adding `path2dir/col_path`, by
            default None.
        col_path : str, optional
            Name of the column that contains image paths, by default
            "image_id".
        keep_these_cols : bool, optional
            Only keep these columns. If False (default), all columns will be
            kept.
        append : bool, optional
            Append a newly read CSV file to `self.annotations`. By default,
            True.
        col_label : str, optional
            Name of the column that contains labels.
        shuffle_rows : bool, optional
            Shuffle rows after reading annotations. Default is True.
        reset_index : bool, optional
            Reset the index of the annotation DataFrame at the end of the
            method. Default is True.
        random_state : int, optional
            Random seed for row shuffling.

        Returns
        -------
        None
        """
        if isinstance(csv_path, str):
            print(f"* reading: {csv_path}")
            annots_rd = pd.read_csv(csv_path)
        else:
            print("* reading dataframe")
            annots_rd = csv_path.copy()
        self.col_label = col_label
        print(f"* #rows: {len(annots_rd)}")
        print(
            f"* label column name: {self.col_label} (you can change this later by .set_col_label(new_label) )"  # noqa
        )
        if shuffle_rows:
            annots_rd = annots_rd.sample(frac=1, random_state=random_state)
            print("* shuffle rows: Yes")

        if keep_these_cols:
            annots_rd = annots_rd[keep_these_cols]

        if self.col_path is None:
            self.col_path = col_path
        elif self.col_path != col_path:
            print(
                f"[WARNING] previously, the col_path was set to {self.col_path}. Column '{col_path}' will be renamed."  # noqa
            )
            annots_rd.rename(columns={col_path: self.col_path}, inplace=True)

        if path2dir:
            print(
                f"* update paths in '{self.col_path}' column by inserting '{path2dir}'"  # noqa
            )
            annots_rd[self.col_path] = (
                os.path.abspath(path2dir)
                + os.path.sep
                + annots_rd[self.col_path]
            )

        if (len(self.annotations) == 0) or (append is False):
            self.annotations = annots_rd.copy()
        else:
            self.annotations = pd.concat(
                [self.annotations, annots_rd.copy()], ignore_index=True
            )

        self.annotations.drop_duplicates(subset=[self.col_path], inplace=True)
        if reset_index:
            self.annotations.reset_index(drop=True, inplace=True)
        print()
        print(self)

    def set_col_label(self, new_label: str = "label") -> None:
        """
        Set a new label for the column that contains the labels.

        Parameters
        ----------
        new_label : str, optional
            Name of the new label column, by default "label"

        Returns
        -------
        None
        """
        self.col_label = new_label

    def show_image(self, indx: int, cmap: Optional[str] = "viridis") -> None:
        """
        Display an image specified by its index along with its label.

        Parameters
        ----------
        indx : int
            Index of the image in the annotations DataFrame to display.
        cmap : str, optional
            The colormap to use, by default "viridis".

        Returns
        -------
        None
        """
        if (self.col_path is None) or (len(self.annotations) == 0):
            print(f"[ERROR] length: {len(self.annotations)}")
            return

        plt.imshow(
            io.imread(self.annotations.iloc[indx][self.col_path]), cmap=cmap
        )
        plt.title(self.annotations.iloc[indx][self.col_label])
        plt.xticks([])
        plt.yticks([])
        plt.pause(0.001)
        plt.show()

    def adjust_labels(self, shiftby: int = -1) -> None:
        """
        Shift labels in the self.annotations DataFrame by the specified value
        (`shiftby`).

        Parameters
        ----------
        shiftby : int, optional
            The value to shift labels by. Default is -1.

        Returns
        -------
        None

        Notes
        -----
        This function updates the `self.annotations` DataFrame by adding the
        value of `shiftby` to the values of the `self.col_label` column. It
        also prints the value counts of the `self.col_label` column before and
        after the shift.
        """
        print(20 * "-")
        print("[INFO] value counts before shift:")
        print(self.annotations[self.col_label].value_counts())

        self.annotations[self.col_label] += shiftby

        print(20 * "-")
        print("[INFO] value counts after shift:")
        print(self.annotations[self.col_label].value_counts())
        print(20 * "-")

    def review_labels(
        self,
        tar_label: Optional[int] = None,
        start_indx: Optional[int] = 1,
        chunks: Optional[int] = 8 * 6,
        num_cols: Optional[int] = 8,
        figsize: Union[list, tuple] = (8 * 3, 8 * 2),
        exclude_df: Optional[pd.DataFrame] = None,
        include_df: Optional[pd.DataFrame] = None,
        deduplicate_col: Optional[str] = "image_id",
    ) -> None:
        """
        Perform image review on annotations and update labels for a given
        label or all labels.

        Parameters
        ----------
        tar_label : int, optional
            The target label to review. If not provided, all labels will be
            reviewed, by default None.
        start_indx : int, optional
            The index of the first image to review, by default 1.
        chunks : int, optional
            The number of images to display at a time, by default 8*6.
        num_cols : int, optional
            The number of columns in the display, by default 8.
        figsize : list or tuple, optional
            The size of the display window, by default (8*3, 8*2).
        exclude_df : pandas DataFrame, optional
            A DataFrame of images to exclude from review, by default None.
        include_df : pandas DataFrame, optional
            A DataFrame of images to include for review, by default None.
        deduplicate_col : str, optional
            The column to use for deduplicating reviewed images, by default
            "image_id".

        Returns
        -------
        None

        Notes
        ------
        This method reviews images with their corresponding labels and allows
        the user to change the label for each image. The updated labels are
        saved in both the annotations and reviewed DataFrames. If `exclude_df`
        is provided, images with `image_path` in `exclude_df['image_path']` are
        skipped in the review process. If include_df is provided, only images
        with `image_path` in `include_df['image_path']` are reviewed. The
        reviewed DataFrame is deduplicated based on the `deduplicate_col`.
        """
        if tar_label is not None:
            annot2review = self.annotations[
                self.annotations[self.col_label] == tar_label
            ]
        else:
            annot2review = self.annotations

        annot2review.drop_duplicates(inplace=True)

        indx = start_indx - 1
        while indx < len(annot2review):
            plt.figure(figsize=figsize)
            print("\n" + 30 * "*")
            print(
                f"[INFO] review {indx+1}-{indx+chunks}, total: {len(annot2review)}"  # noqa
            )
            print(30 * "*")

            counter = 1
            iter_ids = []
            while (counter <= chunks) and (indx < len(annot2review)):
                # Skip the image if it is in exclude_df
                if exclude_df is not None:
                    if (
                        annot2review.iloc[indx]["image_path"]
                        in exclude_df["image_path"].to_list()
                    ):
                        indx += 1
                        continue

                # Skip the image if it is NOT in include_df
                if include_df is not None:
                    if (
                        annot2review.iloc[indx]["image_path"]
                        not in exclude_df["image_path"].to_list()
                    ):
                        indx += 1
                        continue

                # The first term is just a ceiling division, equivalent to:
                # from math import ceil
                # int(ceil(chunks / num_cols))
                plt.subplot(-(-chunks // num_cols), num_cols, counter)
                plt.imshow(io.imread(annot2review.iloc[indx][self.col_path]))
                plt.xticks([])
                plt.yticks([])
                plt.title(
                    f"{annot2review.iloc[indx][self.col_label]} | id: {annot2review.iloc[indx].name}"  # noqa
                )
                iter_ids.append(annot2review.iloc[indx].name)
                # Add to reviewed
                self.reviewed = self.reviewed.append(annot2review.iloc[indx])
                try:
                    self.reviewed.drop_duplicates(subset=[deduplicate_col])
                except Exception:
                    pass
                counter += 1
                indx += 1
            plt.show()

            print(f"list of IDs: {iter_ids}")
            q = "Enter 'ids', comma separated (or press enter to continue): "
            user_input_ids = input(q)

            while user_input_ids.strip().lower() not in [
                "",
                "exit",
                "end",
                "stop",
            ]:
                list_input_ids = user_input_ids.split(",")
                input_label = int(input("Enter label  :  "))

                for one_input_id in list_input_ids:
                    input_id = int(one_input_id)
                    # Change both annotations and reviewed
                    self.annotations.loc[
                        input_id, self.col_label
                    ] = input_label
                    self.reviewed.loc[input_id, self.col_label] = input_label
                    print(f"{input_id} ---> new label: {input_label}")

                user_input_ids = input(q)

            if user_input_ids.lower() in ["exit", "end", "stop"]:
                break

        print("[INFO] Exit...")

    def show_image_labels(
        self, tar_label: Optional[int] = 1, num_sample: Optional[int] = 10
    ) -> None:
        """Show a random sample of images with the specified label (tar_label).

        Parameters
        ----------
        tar_label : int, optional
            The label to filter the images by, by default 1.
        num_sample : int, optional
            The number of images to show. If None, all images with the
            specified label will be shown. Default is 10.

        Returns
        -------
        None
        """
        if (self.col_path is None) or (len(self.annotations) == 0):
            print(f"[ERROR] length: {len(self.annotations)}")
            return

        annot2plot = self.annotations[
            self.annotations[self.col_label] == tar_label
        ]

        if num_sample is None:
            num_sample = len(annot2plot)

        plt.figure(figsize=(8, num_sample))
        for indx in range(num_sample):
            plt.subplot(int(num_sample / 2.0), 3, indx + 1)
            plt.imshow(io.imread(annot2plot.iloc[indx][self.col_path]))
            plt.xticks([])
            plt.yticks([])
            plt.title(annot2plot.iloc[indx][self.col_label])
        plt.show()

    def split_annotations(
        self,
        stratify_colname: Optional[str] = "label",
        frac_train: Optional[float] = 0.70,
        frac_val: Optional[float] = 0.15,
        frac_test: Optional[float] = 0.15,
        random_state: Optional[int] = 1364,
    ) -> None:
        """
        Splits the dataset into three subsets: training, validation, and test
        sets (DataFrames).

        Following fractional ratios provided by the user, where each subset is
        stratified by the values in a specific column (that is, each subset has
        the same relative frequency of the values in the column). It performs
        this splitting by running `train_test_split()` twice.

        Parameters
        ----------
        stratify_colname : str, optional
            Name of the column on which to stratify the split. The default is
            "label".
        frac_train : float, optional
            Fraction of the dataset to be used for training. The default is
            0.70.
        frac_val : float, optional
            Fraction of the dataset to be used for validation. The default is
            0.15.
        frac_test : float, optional
            Fraction of the dataset to be used for testing. The default is
            0.15.
        random_state : int, optional
            Random seed to ensure reproducibility. The default is 1364.

        Raises
        ------
        ValueError
            If the sum of fractions of training, validation and test sets does
            not add up to 1.

        ValueError
            If `stratify_colname` is not a column in the dataframe.

        Returns
        -------
        None
            Sets properties `df_train`, `df_val`, `df_test` -- three
            Dataframes containing the three splits on the `loadAnnotations`
            instance.
        """

        if abs(frac_train + frac_val + frac_test - 1.0) > 1e-4:
            raise ValueError(
                f"fractions {frac_train}, {frac_val}, {frac_test} do not add up to 1.0."  # noqa
                f"Their sum: {frac_train+frac_val+frac_test}"
            )

        if stratify_colname not in self.annotations.columns:
            raise ValueError(
                f"{stratify_colname} is not a column in the dataframe"
            )

        X = self.annotations  # Contains all columns.
        y = X[
            [stratify_colname]
        ]  # Dataframe of just the column on which to stratify.

        # Split original dataframe into train and temp dataframes.
        df_train, df_temp, y_train, y_temp = train_test_split(
            X,
            y,
            stratify=y,
            test_size=(1.0 - frac_train),
            random_state=random_state,
        )

        if abs(frac_test) < 1e-3:
            df_val = df_temp
            df_test = None
            assert len(self.annotations) == len(df_train) + len(df_val)
        else:
            # Split the temp dataframe into val and test dataframes.
            relative_frac_test = frac_test / (frac_val + frac_test)
            df_val, df_test, y_val, y_test = train_test_split(
                df_temp,
                y_temp,
                stratify=y_temp,
                test_size=relative_frac_test,
                random_state=random_state,
            )
            assert len(self.annotations) == len(df_train) + len(df_val) + len(
                df_test
            )

        self.train = df_train
        self.val = df_val
        self.test = df_test
        print("---------------------")
        print("* Split dataset into:")
        print(f"    Train: {len(self.train)}")
        print(f"    Valid: {len(self.val)}")
        print(f"    Test : {len(self.test) if self.test is not None else 0}")
        print("---------------------")

    def sample_labels(
        self,
        tar_label: Union[int, str],
        num_samples: int,
        random_state: Optional[int] = 12345,
    ) -> None:
        """
        Randomly sample a given number of annotations with a given target
        label and remove all other annotations from the dataframe.

        Parameters
        ----------
        tar_label : int or str
            The target label for which the annotations will be sampled.
        num_samples : int
            The number of annotations to be sampled.
        random_state : int, optional
            Seed to ensure reproducibility of the random number generator.

        Raises
        ------
        ValueError
            If `tar_label` is not a column in the dataframe.

        Returns
        -------
        None
            The dataframe with remaining annotations is stored in
            `self.annotations`.
        """
        if (self.col_path is None) or (len(self.annotations) == 0):
            print(f"[ERROR] length: {len(self.annotations)}")
            return
        all_annots = self.annotations.copy()
        tar_rows = all_annots[all_annots[self.col_label] == tar_label]
        tar_samples = tar_rows.sample(num_samples, random_state=random_state)
        new_annots = all_annots[
            (all_annots[self.col_label] != tar_label)
            | (all_annots.index.isin(tar_samples.index))
        ]
        self.annotations = new_annots

    def __str__(self):
        print("------------------------")
        print(f"* Number of annotations: {len(self.annotations)}\n")
        if len(self.annotations) > 0:
            print("* First few rows:")
            print(self.annotations.head())
            print("...\n")
            print(f"* Value counts (column: {self.col_label}):")
            print(self.annotations[self.col_label].value_counts())
        print("------------------------")
        return ""
