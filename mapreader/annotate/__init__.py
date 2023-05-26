import functools
import hashlib
import json
import os
import random
import string
from itertools import product
from typing import Optional, Tuple, Union, List

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display
from numpy import array_split
from PIL import Image, ImageOps

from ..load.loader import load_patches

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

MAX_SIZE = 100

_CENTER_LAYOUT = widgets.Layout(
    display="flex", flex_flow="column", align_items="center"
)


class Annotator(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        # Untangle args and kwargs
        patch_df = None
        if len(args) == 1:
            patch_df = args.pop(0)
        elif len(args) == 2:
            patch_df = args.pop(0)
            parent_df = args.pop(1)
        elif isinstance(
            kwargs.get("data"), (str, dict, list, type(None), pd.DataFrame)
        ):
            patch_df = kwargs.get("data")
        elif kwargs.get("data") is not None:
            raise SyntaxError(f"Cannot interpret data of type {type(patch_df)}")

        if isinstance(
            kwargs.get("metadata"), (str, dict, list, type(None), pd.DataFrame)
        ):
            parent_df = kwargs.get("metadata")

        kwargs["patches"] = kwargs.get("patches", "./patches/patch-*.png")
        kwargs["parents"] = kwargs.get("parents", "./maps/*.png")
        kwargs["annotations_dir"] = kwargs.get("annotations_dir", "./annotations")

        kwargs["metadata"] = kwargs.get("metadata", "./maps/metadata.csv")
        kwargs["username"] = kwargs.get(
            "username",
            "".join(
                [random.choice(string.ascii_letters + string.digits) for n in range(30)]
            ),
        )
        kwargs["task_name"] = kwargs.get("task_name", "task")
        kwargs["image_column"] = kwargs.get("image_column", "image_path")
        kwargs["label_column"] = kwargs.get("label_column", "label")
        kwargs["labels"] = kwargs.get("labels", [])
        kwargs["scramble_frame"] = kwargs.get("scramble_frame", True)
        kwargs["buttons_per_row"] = kwargs.get("buttons_per_row", None)
        kwargs["auto_save"] = kwargs.get("auto_save", True)
        kwargs["stop_at_last_example"] = kwargs.get("stop_at_last_example", True)
        kwargs["show_context"] = kwargs.get("show_context", True)
        kwargs["min_values"] = kwargs.get("min_values", {})
        kwargs["max_values"] = kwargs.get("max_values", {})
        kwargs["metadata_delimiter"] = kwargs.get("metadata_delimiter", "|")

        # Check metadata
        if isinstance(parent_df, str):
            # we have data as string = assume it's a path to a
            parent_df = pd.read_csv(parent_df, delimiter=kwargs["metadata_delimiter"])

        if isinstance(parent_df, (dict, list)):
            # we have data as string = assume it's a path to a
            parent_df = pd.DataFrame(parent_df)

        # Check data
        if isinstance(patch_df, str):
            # we have data as string = assume it's a path to a
            patch_df = pd.read_csv(patch_df)

        if isinstance(patch_df, (dict, list)):
            # we have data as string = assume it's a path to a
            patch_df = pd.DataFrame(patch_df)

        if isinstance(patch_df, type(None)):
            # If we don't get data provided, we'll use the patches and parents to
            # load up the patches
            try:
                parent_df, patch_df = self._load_frames(**kwargs)
                try:
                    patch_df = patch_df.join(parent_df["url"], on="parent_id")
                except Exception as e:
                    raise RuntimeError(
                        f"Could not join the URL column from the metadata with the data: {e}"
                    )
            except NameError:
                raise SyntaxError(
                    "Data must be provided or class must have a _load_frames method."
                )

        # Last check for metadata + data
        if not len(patch_df):
            raise RuntimeError("No data available.")

        if not len(parent_df):
            raise RuntimeError("No metadata available.")

        # Test for columns
        if kwargs["label_column"] not in patch_df.columns:
            raise SyntaxError(
                f"Your DataFrame does not have the label column ({kwargs['label_column']})"
            )

        if kwargs["image_column"] not in patch_df.columns:
            raise SyntaxError(
                f"Your DataFrame does not have the image column ({kwargs['image_column']})"
            )

        if kwargs.get("sortby"):
            patch_df = patch_df.sort_values(kwargs["sortby"])

        image_list = json.dumps(
            sorted(patch_df[kwargs["image_column"]].to_list()), sort_keys=True
        )

        kwargs["id"] = hashlib.md5(image_list.encode("utf-8")).hexdigest()

        annotations_file = (
            kwargs["task_name"].replace(" ", "_")
            + f"_#{kwargs['username']}#-{kwargs['id']}.csv"
        )
        kwargs["annotations_file"] = os.path.join(
            kwargs["annotations_dir"], annotations_file
        )

        # Test for existing file
        if os.path.exists(kwargs["annotations_file"]):
            print(
                f"[INFO] Existing annotations for {kwargs['username']} being loaded..."
            )
            existing_annotations = pd.read_csv(kwargs["annotations_file"], index_col=0)
            try:
                existing_annotations[kwargs["label_column"]] = existing_annotations[
                    kwargs["label_column"]
                ].apply(lambda x: kwargs["labels"][x])
            except TypeError:
                # We will assume the label column now contains the label value and not the indices for the labels
                pass

            patch_df = patch_df.join(
                existing_annotations, how="left", lsuffix="_x", rsuffix="_y"
            )
            patch_df[kwargs["label_column"]] = patch_df["label_y"].fillna(
                patch_df[f"{kwargs['label_column']}_x"]
            )
            patch_df = patch_df.drop(
                columns=[
                    f"{kwargs['label_column']}_x",
                    f"{kwargs['label_column']}_y",
                ]
            )
            patch_df["changed"] = patch_df[kwargs["label_column"]].apply(
                lambda x: True if x else False
            )

            try:
                patch_df[kwargs["image_column"]] = patch_df[
                    f"{kwargs['image_column']}_x"
                ]
                patch_df = patch_df.drop(
                    columns=[
                        f"{kwargs['image_column']}_x",
                        f"{kwargs['image_column']}_y",
                    ]
                )
            except:
                pass

        # initiate as a DataFrame
        super().__init__(patch_df)

        # pixel_bounds = x0, y0, x1, y1 # I checked with Rosie
        self["min_x"] = self.pixel_bounds.apply(lambda x: x[0])
        self["min_y"] = self.pixel_bounds.apply(lambda x: x[1])
        self["max_x"] = self.pixel_bounds.apply(lambda x: x[2])
        self["max_y"] = self.pixel_bounds.apply(lambda x: x[3])

        self._buttons = []
        self._labels = kwargs["labels"]
        self.label_column = kwargs["label_column"]
        self.image_column = kwargs["image_column"]
        self.buttons_per_row = kwargs["buttons_per_row"]
        self.auto_save = kwargs["auto_save"]
        self.annotations_dir = kwargs["annotations_dir"]
        self.task_name = kwargs["task_name"]
        self.id = kwargs["id"]
        self.annotations_file = kwargs["annotations_file"]
        self.username = kwargs["username"]
        self.stop_at_last_example = kwargs["stop_at_last_example"]
        self.show_context = kwargs["show_context"]
        self._min_values = kwargs["min_values"]
        self._max_values = kwargs["max_values"]
        self.metadata = parent_df
        self.patch_width, self.patch_height = self.get_patch_size()
        self.metadata_delimiter = kwargs["metadata_delimiter"]

        # Set up standards for context display
        self.surrounding = 1
        self.margin = 1

        # Ensure labels are of type list
        if not isinstance(self._labels, list):
            raise SyntaxError("Labels provided must be as a list")

        # Ensure unique values in list
        self._labels = sorted(list(set(self._labels)))

        # Set max buttons
        if not self.buttons_per_row:
            if (len(self._labels) % 2) == 0:
                if len(self._labels) > 4:
                    self.buttons_per_row = 4
                else:
                    self.buttons_per_row = 2
            else:
                if len(self._labels) == 3:
                    self.buttons_per_row = 3
                else:
                    self.buttons_per_row = 5

        # Set indices
        self.current_index = -1
        self.previous_index = 0

        # Setup buttons
        self._setup_buttons()

        # Setup box for buttons
        self._setup_box()

        # Setup queue
        self._queue = self.get_queue()

    @classmethod
    def _load_frames(cls, *_, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads patches and parents data from given paths and returns the
        corresponding dataframes.

        Parameters
        ----------
        **kwargs :
            Needs to contain "patches" and "parents"

            Needs to contain "metadata" and "metadata_delimiter"

            Needs to contain "label_column"

            Needs to contain "scramble_frame"

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Parents and patches dataframes.
        """
        print(
            f"[INFO] Loading patches and parents data from {kwargs['patches']}\
                and {kwargs['parents']}"
        )
        patches = load_patches(
            patch_paths=kwargs["patches"], parent_paths=kwargs["parents"]
        )

        # Add pixel stats
        patches.calc_pixel_stats()

        print(
            f"[INFO] Adding metadata from {kwargs['metadata']} (delimiter {kwargs['metadata_delimiter']})"
        )
        patches.add_metadata(kwargs["metadata"], delimiter=kwargs["metadata_delimiter"])

        parents, patches = patches.convert_images()

        if kwargs["label_column"] not in patches.columns:
            patches[kwargs["label_column"]] = None
        patches["changed"] = False

        if kwargs["scramble_frame"]:
            # Scramble them!
            patches = patches.sample(frac=1)

        return parents, patches

    def get_patch_size(self):
        """
        Calculate and return the width and height of the patches based on the
        first patch of the DataFrame, assuming the same shape of patches
        across the frame.

        Returns
        -------
        Tuple[int, int]
            Width and height of the patches.
        """
        patch_width = self.max_x[0] - self.min_x[0]
        patch_height = self.max_y[0] - self.min_y[0]

        return patch_width, patch_height

    def _setup_buttons(self) -> None:
        """
        Set up buttons for each label to be annotated.
        """
        for label in self._labels:
            btn = widgets.Button(
                description=label,
                button_style="info",
                layout=widgets.Layout(flex="1 1 0%", width="auto"),
            )
            btn.style.button_color = "#9B6F98"

            def on_click(lbl, *_, **__):
                self._add_annotation(lbl)

            btn.on_click(functools.partial(on_click, label))
            self._buttons.append(btn)

    def _setup_box(self) -> None:
        """
        Set up the box which holds all the buttons.
        """
        if len(self._buttons) > self.buttons_per_row:
            self.box = widgets.VBox(
                [
                    widgets.HBox(self._buttons[x : x + self.buttons_per_row])
                    for x in range(0, len(self._buttons), self.buttons_per_row)
                ]
            )
        else:
            self.box = widgets.HBox(self._buttons)

        # back button
        prev_btn = widgets.Button(
            description="« previous", layout=widgets.Layout(flex="1 1 0%", width="auto")
        )
        prev_btn.on_click(self._prev_example)

        # next button
        next_btn = widgets.Button(
            description="next »", layout=widgets.Layout(flex="1 1 0%", width="auto")
        )
        next_btn.on_click(self._next_example)

        self.navbox = widgets.VBox([widgets.HBox([prev_btn, next_btn])])

    def get_queue(
        self, as_type: Optional[str] = "list"
    ) -> Union[List[int], pd.Index, pd.Series]:
        """
        Gets the indices of rows which are legible for annotation.

        Parameters
        ----------
        as_type : str, optional
            The format in which to return the indices. Options: "list",
            "index". Default is "list". If any other value is provided, it
            returns a pandas.Series.

        Returns
        -------
        List[int] or pandas.Index or pandas.Series
            Depending on "as_type", returns either a list of indices, a
            pd.Index object, or a pd.Series of legible rows.
        """

        def check_legibility(row):
            if row.label is not None:
                return False

            test = [
                row[col] >= min_value for col, min_value in self._min_values.items()
            ] + [row[col] <= max_value for col, max_value in self._max_values.items()]

            if not all(test):
                return False

            return True

        test = self.copy()
        test["eligible"] = test.apply(check_legibility, axis=1)
        test = test[
            ["eligible"] + [col for col in test.columns if not col == "eligible"]
        ]

        indices = test[test.eligible].index
        if as_type == "list":
            return list(indices)
        if as_type == "index":
            return indices
        return test[test.eligible]

    def get_context(self):
        """
        Provides the surrounding context for the patch to be annotated.

        Returns
        -------
        ipywidgets.VBox
            An IPython VBox widget containing the surrounding patches for
            context.

        ..
            TODO: ensure view of the patch is no larger than 150px x 150px
            using ImageOps.contain(image, (150, 150))
        """

        def get_path(image_path, dim=True):
            if dim == True or dim == "True":
                dim = True
                im = Image.open(image_path)
                image_path = ".temp.png"

                # add alpha
                alpha = Image.new("L", im.size, 60)
                im.putalpha(alpha)

                # ensure size
                if max(im.size) > MAX_SIZE:
                    im = ImageOps.contain(im, (MAX_SIZE, MAX_SIZE))

                im.save(image_path)
            else:
                dim = False

                # ensure correct size
                im = Image.open(image_path)
                if max(im.size) > MAX_SIZE:
                    im = ImageOps.contain(im, (MAX_SIZE, MAX_SIZE))

                image_path = ".temp.png"
                im.save(image_path)

            with open(image_path, "rb") as f:
                im = f.read()

            layout = widgets.Layout(margin=f"{self.margin}px")
            return widgets.Image(value=im, layout=layout)

        def get_empty_square():
            im = Image.new(
                size=(self.patch_width, self.patch_height),
                mode="RGB",
                color="white",
            )
            image_path = ".temp.png"
            im.save(image_path)

            with open(image_path, "rb") as f:
                im = f.read()

            return widgets.Image(value=im)

        if self.surrounding > 3:
            display(
                widgets.HTML(
                    """<p style="color:red;"><b>Warning: More than 3 surrounding tiles may crowd the display and not display correctly.</b></p>"""
                )
            )

        ix = self._queue[self.current_index]

        x = self.at[ix, "min_x"]
        y = self.at[ix, "min_y"]
        current_parent = self.at[ix, "parent_id"]

        parent_frame = self.query(f"parent_id=='{current_parent}'")

        deltas = list(range(-self.surrounding, self.surrounding + 1))
        y_and_x = list(
            product(
                [y + y_delta * self.patch_height for y_delta in deltas],
                [x + x_delta * self.patch_width for x_delta in deltas],
            )
        )
        queries = [f"min_x == {x} & min_y == {y}" for y, x in y_and_x]
        items = [parent_frame.query(query) for query in queries]

        # derive ids from items
        ids = [x.index[0] if len(x.index) == 1 else None for x in items]
        ids = [x != ix for x in ids]

        # derive images from items
        images = [
            x.at[x.index[0], "image_path"] if len(x.index) == 1 else None for x in items
        ]

        # zip them
        images = list(zip(images, ids))

        # split them into rows
        per_row = len(deltas)
        image_widgets = [
            [get_path(x[0], dim=x[1]) if x[0] else get_empty_square() for x in lst]
            for lst in array_split(images, per_row)
        ]

        h_boxes = [widgets.HBox(x) for x in image_widgets]

        return widgets.VBox(h_boxes, layout=_CENTER_LAYOUT)

    def annotate(
        self,
        show_context: Optional[bool] = None,
        min_values: Optional[dict] = {},
        max_values: Optional[dict] = {},
        surrounding: Optional[int] = 1,
        margin: Optional[int] = 0,
    ) -> None:
        """
        Renders the annotation interface for the first image.

        Parameters
        ----------
        show_context : bool, optional
            Whether or not to display the surrounding context for each image.
            Default: None.
        min_values : dict, optional
            Minimum values for each property to filter images for annotation.
            It should be provided as a dictionary consisting of column names
            (keys) and minimum values as floating point values (values).
            Default: {}.
        max_values : dict, optional
            Maximum values for each property to filter images for annotation.
            It should be provided as a dictionary consisting of column names
            (keys) and minimum values as floating point values (values).
            Default: {}.
        surrounding : int, optional
            The number of surrounding images to show for context. Default: 1.
        margin : int, optional
            The margin to use for the context images. Default: 0.

        Returns
        -------
        None
        """
        self.current_index = -1
        for button in self._buttons:
            button.disabled = False

        if show_context is not None:
            self.show_context = show_context

        self._min_values = min_values
        self._max_values = max_values
        self.surrounding = surrounding
        self.margin = margin

        # re-set up queue
        self._queue = self.get_queue()

        self.out = widgets.Output()
        display(self.box)
        display(self.navbox)
        display(self.out)

        # self.get_current_index()
        # TODO: Does not pick the correct NEXT...
        self._next_example()

    def _next_example(self, *_) -> Tuple[int, int, str]:
        """
        Advances the annotation interface to the next image.

        Returns
        -------
        Tuple[int, int, str]
            Previous index, current index, and path of the current image.
        """
        if not len(self._queue):
            self.render_complete()
            return

        if isinstance(self.current_index, type(None)) or self.current_index == -1:
            self.current_index = 0
        else:
            current_index = self.current_index + 1

            try:
                self._queue[current_index]
                self.previous_index = self.current_index
                self.current_index = current_index
            except IndexError:
                pass

        ix = self._queue[self.current_index]

        img_path = self.at[ix, self.image_column]

        self.render()
        return self.previous_index, self.current_index, img_path

    def _prev_example(self, *_) -> Tuple[int, int, str]:
        """
        Moves the annotation interface to the previous image.

        Returns
        -------
        Tuple[int, int, str]
            Previous index, current index, and path of the current image.
        """
        if not len(self._queue):
            self.render_complete()
            return

        current_index = self.current_index - 1

        if current_index < 0:
            current_index = 0

        try:
            self._queue[current_index]
            self.previous_index = current_index - 1
            self.current_index = current_index
        except IndexError:
            pass

        ix = self._queue[self.current_index]

        img_path = self.at[ix, self.image_column]

        self.render()
        return self.previous_index, self.current_index, img_path

    def render(self) -> None:
        """
        Displays the image at the current index in the annotation interface.

        If the current index is greater than or equal to the length of the
        dataframe, the method disables the "next" button and saves the data.

        Returns
        -------
        None
        """
        # Check whether we have reached the end
        if self.current_index >= len(self) - 1:
            if self.stop_at_last_example:
                self.render_complete()
            else:
                self._prev_example()
            return

        # ix = self.iloc[self.current_index].name
        ix = self._queue[self.current_index]

        # render buttons
        for button in self._buttons:
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
            image = self.get_patch_image(ix)
            if self.show_context:
                context = self.get_context()
                display(context)
            else:
                display(image)
            add_ins = []
            if self.at[ix, "url"]:
                url = self.at[ix, "url"]
                add_ins += [
                    widgets.HTML(
                        f'<p><a href="{url}" target="_blank">Click to see entire map.</a></p>'
                    )
                ]
            add_ins += [
                widgets.IntProgress(
                    value=self.current_index + 1 if self.current_index else 1,
                    min=0,
                    max=len(self._queue),
                    step=1,
                    description=f"{self.current_index + 1 if self.current_index else 1} / {len(self._queue)}",
                    orientation="horizontal",
                    barstyle="success",
                )
            ]
            display(
                widgets.VBox(
                    add_ins,
                    layout=_CENTER_LAYOUT,
                )
            )

    def get_patch_image(self, ix: int) -> widgets.Image:
        """
        Returns the image at the given index.

        Parameters
        ----------
        ix : int
            The index of the image in the dataframe.

        Returns
        -------
        ipywidgets.Image
            A widget displaying the image at the given index.
        """
        image_path = self.at[ix, self.image_column]
        with open(image_path, "rb") as f:
            image = f.read()

        return widgets.Image(value=image)

    def _add_annotation(self, annotation: str) -> None:
        """
        Adds the provided annotation to the current image.

        Parameters
        ----------
        annotation : str
            The label to add to the current image.

        Returns
        -------
        None
        """
        # ix = self.iloc[self.current_index].name
        ix = self._queue[self.current_index]
        self.at[ix, self.label_column] = annotation
        self.at[ix, "changed"] = True
        if self.auto_save:
            self._auto_save()
        self._next_example()

    def _auto_save(self):
        """
        Automatically saves the annotations made so far.

        Returns
        -------
        None
        """
        self.get_labelled_data(sort=True).to_csv(self.annotations_file)

    def get_labelled_data(
        self,
        sort: Optional[bool] = True,
        index_labels: Optional[bool] = False,
        include_paths: Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        Returns the annotations made so far.

        Parameters
        ----------
        sort : bool, optional
            Whether to sort the dataframe by the order of the images in the
            input data, by default True
        index_labels : bool, optional
            Whether to return the label's index number (in the labels list
            provided in setting up the instance) or the human-readable label
            for each row, by default False
        include_paths : bool, optional
            Whether to return a column containing the full path to the
            annotated image or not, by default True

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the labelled images and their associated
            label index.
        """
        if index_labels:
            col1 = self.filtered[self.label_column].apply(
                lambda x: self._labels.index(x)
            )
        else:
            col1 = self.filtered[self.label_column]

        if include_paths:
            col2 = self.filtered[self.image_column]
            df = pd.DataFrame(
                {self.image_column: col2, self.label_column: col1},
                index=pd.Index(col1.index, name="image_id"),
            )
        else:
            df = pd.DataFrame(col1, index=pd.Index(col1.index, name="image_id"))
        if not sort:
            return df

        df["sort_value"] = df.index.to_list()
        df["sort_value"] = df["sort_value"].apply(
            lambda x: f"{x.split('#')[1]}-{x.split('#')[0]}"
        )
        return df.sort_values("sort_value").drop(columns=["sort_value"])

    @property
    def filtered(self) -> pd.DataFrame:
        _filter = ~self[self.label_column].isna()
        return self[_filter]

    def render_complete(self):
        """
        Renders the completion message once all images have been annotated.

        Returns
        -------
        None
        """
        clear_output()
        display(
            widgets.HTML(f"<p><b>All annotations done with current settings.</b></p>")
        )
        if self.auto_save:
            self._auto_save()
        for button in self._buttons:
            button.disabled = True
