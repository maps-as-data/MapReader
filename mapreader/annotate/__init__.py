import functools
import hashlib
import json
import os
import random
import string
from itertools import product
from typing import Optional, Tuple

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display
from numpy import array_split
from PIL import Image, ImageOps

from ..load.loader import load_patches

_CENTER_LAYOUT = widgets.Layout(
    display="flex", flex_flow="column", align_items="center"
)

MAX_SIZE = 100


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
    metadata_delimiter : str, optional
        Separator in the metadata file. Default: "|".
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
    stop_at_last_example : bool, optional
        Whether the annotation process should stop when it reaches the last
        example in the dataframe. Default: True.
    show_context : bool, optional
        Whether to show the images that appear around the given image that
        is being annotated. Default: False.
    sortby : str, optional
        The name of the column by which to sort the data to be annotated.
    min_values : dict, optional
        A dictionary consisting of column names (keys) and minimum values as
        floating point values (values) which will be applied as a filter to
        the annotation data before annotations commence. Default: ``{}``.
    max_values : dict, optional
        A dictionary consisting of column names (keys) and maximum values as
        floating point values (values) which will be applied as a filter to
        the annotation data before annotations commence. Default: ``{}``.

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
    auto_save : bool
        Whether to automatically save annotations during annotation.
    annotations_dir : str
        Directory to save the annotations CSV file.
    task_name : str
        Name of the annotation task.
    id : str
        Unique identifier for the current annotation session.
    annotations_file : str
        Filename for the resulting annotations CSV file.
    username : str
        Username for the current annotation session.
    current_index : int
        Current index of the annotation process.
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

        self.buttons = []
        self.labels = kwargs["labels"]
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
        self.min_values = kwargs["min_values"]
        self.max_values = kwargs["max_values"]
        self.metadata = parent_df
        self.patch_width, self.patch_height = self.get_patch_size()
        self.metadata_delimiter = kwargs["metadata_delimiter"]

        # Ensure labels are of type list
        if not isinstance(self.labels, list):
            raise SyntaxError("Labels provided must be as a list")

        # Ensure unique values in list
        self.labels = sorted(list(set(self.labels)))

        # Set max buttons
        if not self.buttons_per_row:
            if (len(self.labels) % 2) == 0:
                if len(self.labels) > 4:
                    self.buttons_per_row = 4
                else:
                    self.buttons_per_row = 2
            else:
                if len(self.labels) == 3:
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
        self.queue = self.get_queue()

    def _load_frames(self, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        parents, patches = patches.convertImages()

        if kwargs["label_column"] not in patches.columns:
            patches[kwargs["label_column"]] = None
        patches["changed"] = False

        if kwargs["scramble_frame"]:
            # Scramble them!
            patches = patches.sample(frac=1)

        return parents, patches

    def _setup_buttons(self) -> None:
        for label in self.labels:
            btn = widgets.Button(
                description=label,
                button_style="info",
                layout=widgets.Layout(flex="1 1 0%", width="auto"),
            )
            btn.style.button_color = "#9B6F98"

            def on_click(lbl, *_, **__):
                self._add_annotation(lbl)

            btn.on_click(functools.partial(on_click, label))
            self.buttons.append(btn)

    def _setup_box(self) -> None:
        if len(self.buttons) > self.buttons_per_row:
            self.box = widgets.VBox(
                [
                    widgets.HBox(self.buttons[x : x + self.buttons_per_row])
                    for x in range(0, len(self.buttons), self.buttons_per_row)
                ]
            )
        else:
            self.box = widgets.HBox(self.buttons)

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

    def annotate(
        self, show_context=None, min_values={}, max_values={}, surrounding=1
    ) -> None:
        """
        Renders the annotation interface for the first image.

        Parameters
        ----------
        min_values : dict, optional
            A dictionary consisting of column names (keys) and minimum values as
            floating point values (values) which will be applied as a filter to
            the annotation data before annotations commence. Default: ``{}``.
        max_values : dict, optional
            A dictionary consisting of column names (keys) and maximum values as
            floating point values (values) which will be applied as a filter to
            the annotation data before annotations commence. Default: ``{}``.

        Returns
        -------
        None
        """
        self.current_index = -1
        for button in self.buttons:
            button.disabled = False

        if show_context is not None:
            self.show_context = show_context

        self.min_values = min_values
        self.max_values = max_values
        self.surrounding = surrounding

        # re-set up queue
        self.queue = self.get_queue()

        self.out = widgets.Output()
        display(self.box)
        display(self.navbox)
        display(self.out)

        # self.get_current_index()
        # TODO: Does not pick the correct NEXT...
        self._next_example()

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
        ix = self.queue[self.current_index]

        # render buttons
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
                    max=len(self.queue),
                    step=1,
                    description=f"{self.current_index + 1 if self.current_index else 1} / {len(self.queue)}",
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

    def _add_annotation(self, annotation: str) -> None:
        """Toggle annotation."""
        # ix = self.iloc[self.current_index].name
        ix = self.queue[self.current_index]
        self.at[ix, self.label_column] = annotation
        self.at[ix, "changed"] = True
        if self.auto_save:
            self._auto_save()
        self._next_example()

    @property
    def filtered(self) -> pd.DataFrame:
        _filter = ~self[self.label_column].isna()
        return self[_filter]

    def get_labelled_data(
        self,
        sort: Optional[bool] = True,
        index_labels: Optional[bool] = False,
        include_paths: Optional[bool] = True,
    ) -> pd.DataFrame:
        """
        Returns a dataframe containing only the labelled images and their
        associated label index.

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
                lambda x: self.labels.index(x)
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

    def _auto_save(self):
        self.get_labelled_data(sort=True).to_csv(self.annotations_file)

    def get_patch_size(self):
        patch_width = self.max_x[0] - self.min_x[0]
        patch_height = self.max_y[0] - self.min_y[0]

        return patch_width, patch_height

    def get_patch_image(self, ix):
        image_path = self.at[ix, self.image_column]
        with open(image_path, "rb") as f:
            image = f.read()

        return widgets.Image(value=image)

    def get_context(self):
        """
        TODO:
        - add docstring
        - ensure view of the patch is no larger than 150px x 150px using ImageOps.contain(image, (150, 150))
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

            layout = widgets.Layout(margin="0px")
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

        ix = self.queue[self.current_index]

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

    def get_queue(self, as_type="list"):
        def check_legibility(row):
            if row.label is not None:
                return False

            test = [
                row[col] >= min_value for col, min_value in self.min_values.items()
            ] + [row[col] <= max_value for col, max_value in self.max_values.items()]

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

    def _next_example(self, *args):
        if not len(self.queue):
            self.render_complete()
            return

        if isinstance(self.current_index, type(None)) or self.current_index == -1:
            self.current_index = 0
        else:
            current_index = self.current_index + 1

            try:
                self.queue[current_index]
                self.previous_index = self.current_index
                self.current_index = current_index
            except IndexError:
                pass

        ix = self.queue[self.current_index]

        img_path = self.at[ix, self.image_column]

        self.render()
        return self.previous_index, self.current_index, img_path

    def _prev_example(self, *args):
        if not len(self.queue):
            self.render_complete()
            return

        current_index = self.current_index - 1

        if current_index < 0:
            current_index = 0

        try:
            self.queue[current_index]
            self.previous_index = current_index - 1
            self.current_index = current_index
        except IndexError:
            pass

        ix = self.queue[self.current_index]

        img_path = self.at[ix, self.image_column]

        self.render()
        return self.previous_index, self.current_index, img_path

    def render_complete(self):
        clear_output()
        display(
            widgets.HTML(f"<p><b>All annotations done with current settings.</b></p>")
        )
        if self.auto_save:
            self._auto_save()
        for button in self.buttons:
            button.disabled = True
