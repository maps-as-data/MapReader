#!/usr/bin/env python
from __future__ import annotations

from ast import literal_eval
from itertools import product

import pandas as pd
from tqdm import tqdm


class PatchDataFrame(pd.DataFrame):
    """A class for storing patch dataframes.

    Parameters
    ----------
    patch_df : pd.DataFrame
        the DataFrame containing patches and predictions
    labels_map : dict
        the dictionary mapping label indices to their labels.
        e.g. `{0: "no", 1: "railspace"}`.
    """

    def __init__(
        self,
        patch_df: pd.DataFrame,
        labels_map: dict,
    ):
        super().__init__(patch_df)

        required_columns = [
            "parent_id",
            "pixel_bounds",
            "pred",
            "predicted_label",
            "conf",
        ]
        if not all([col in self.columns for col in required_columns]):
            raise ValueError(
                f"[ERROR] Your dataframe must contain the following columns: {required_columns}."
            )

        # ensure lists/tuples are evaluated as such
        for col in self.columns:
            try:
                self[col] = self[col].apply(literal_eval)
            except (ValueError, TypeError, SyntaxError):
                pass

        if self.index.name == "image_id":
            if "image_id" in self.columns:
                self.drop(columns=["image_id"], inplace=True)
            self.reset_index(drop=False, names="image_id", inplace=True)

        if all([col in self.columns for col in ["min_x", "min_y", "max_x", "max_y"]]):
            print(
                "[INFO] Using existing pixel bounds columns (min_x, min_y, max_x, max_y)."
            )
        else:
            self[["min_x", "min_y", "max_x", "max_y"]] = [*self.pixel_bounds]

        self.labels_map = labels_map
        self._label_patches = None
        self.context = {}

    def get_context(
        self,
        labels: str | list,
    ):
        """Get the context of the patches with the specified labels.

        Parameters
        ----------
        labels : str | list
            The label(s) to get context for.
        """
        if isinstance(labels, str):
            labels = [labels]
        self._label_patches = self[self["predicted_label"].isin(labels)]

        for ix in tqdm(self._label_patches.index):
            if ix not in self.context:
                context_list = self._get_context_id(ix)
                # only add context if all surrounding patches are found
                if len(context_list) == 9:
                    self.context[ix] = context_list

    def _get_context_id(
        self,
        ix,
    ):
        """Get the context of the patch with the specified index."""
        parent_id = self.at[ix, "parent_id"]
        min_x = self.at[ix, "min_x"]
        min_y = self.at[ix, "min_y"]
        max_x = self.at[ix, "max_x"]
        max_y = self.at[ix, "max_y"]

        context_grid = [
            *product(
                [(self["min_x"], min_x), (min_x, max_x), (max_x, self["max_x"])],
                [(self["min_y"], min_y), (min_y, max_y), (max_y, self["max_y"])],
            )
        ]
        # reshape to min_x, min_y, max_x, max_y
        context_grid = [(x[0][0], x[1][0], x[0][1], x[1][1]) for x in context_grid]

        context_list = [
            self[
                (self["min_x"] == context_loc[0])
                & (self["min_y"] == context_loc[1])
                & (self["max_x"] == context_loc[2])
                & (self["max_y"] == context_loc[3])
                & (self["parent_id"] == parent_id)
            ]
            for context_loc in context_grid
        ]
        context_list = [x.image_id.values[0] for x in context_list if len(x)]
        return context_list

    def update_preds(self, remap: dict, conf: float = 0.7, inplace: bool = False):
        """Update the predictions of the chosen patches based on their context.

        Parameters
        ----------
        remap : dict
            A dictionary mapping the old labels to the new labels.
        conf : float, optional
            Patches with confidence scores below this value will be relabelled, by default 0.7.
        inplace : bool, optional
            Whether to relabel inplace or create new dataframe columns, by default False
        """
        if self._label_patches is None:
            raise ValueError("[ERROR] You must run `get_context` first.")
        if len(self.context) == 0:
            raise ValueError(
                "[ERROR] No patches to update. Try changing which labels you are updating."
            )

        labels = self._label_patches["predicted_label"].unique()
        if any([label not in remap.keys() for label in labels]):
            raise ValueError(
                f"[ERROR] You must specify a remap for each label in {labels}."
            )

        # add new label to labels_map if not already present (assume label index is next in sequence)
        for new_label in remap.values():
            if new_label not in self.labels_map.values():
                print(
                    [
                        f"[INFO] Adding {new_label} to labels_map at index {len(self.labels_map)}."
                    ]
                )
                self.labels_map[len(self.labels_map)] = new_label

        for ix in tqdm(self.context):
            self._update_preds_id(
                ix, labels=labels, remap=remap, conf=conf, inplace=inplace
            )

    def _update_preds_id(
        self, ix, labels: str | list, remap: dict, conf: float, inplace: bool = False
    ):
        """Update the predictions of the patch with the specified index."""
        context_list = self.context[ix]

        context_df = self[self["image_id"].isin(context_list)]
        # drop central patch from context
        context_df.drop(index=ix, inplace=True)

        # reverse the labels_map dict
        label_index_dict = {v: k for k, v in self.labels_map.items()}

        prefix = "" if inplace else "new_"
        if (not any(context_df["predicted_label"].isin(labels))) & (
            self.at[ix, "conf"] < conf
        ):
            self.at[ix, f"{prefix}predicted_label"] = remap[
                self.at[ix, "predicted_label"]
            ]
            self.at[ix, f"{prefix}pred"] = label_index_dict[
                self.at[ix, f"{prefix}predicted_label"]
            ]
