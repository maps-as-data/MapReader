import ipywidgets as widgets
from IPython.display import clear_output, display
import pandas as pd
import random
import string
import json
import hashlib
import os

from typing import Tuple, Optional

from ..load.loader import load_patches
import functools
from PIL import Image


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
        kwargs["stop_at_last_example"] = (
            kwargs["stop_at_last_example"]
            if kwargs.get("stop_at_last_example")
            else True
        )
        kwargs["show_context"] = (
            kwargs["show_context"] if kwargs.get("show_context") else False
        )
        kwargs["min_values"] = kwargs["min_values"] if kwargs.get("min_values") else {}
        kwargs["max_values"] = kwargs["max_values"] if kwargs.get("max_values") else {}

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
                try:
                    data = data.join(metadata["url"], on="parent_id")
                except Exception as e:
                    raise RuntimeError(
                        f"Could not join the URL column from the metadata with the data: {e}"
                    )
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

        if kwargs.get("sortby"):
            data = data.sort_values(kwargs["sortby"])

        query1 = " & ".join(
            [
                f"{col} >= {min_value}"
                for col, min_value in kwargs.get("min_values", {}).items()
            ]
        )
        query2 = " & ".join(
            [
                f"{col} <= {max_value}"
                for col, max_value in kwargs.get("max_values", {}).items()
            ]
        )

        query = None
        if query1 and query2:
            query = query1 + " & " + query2
        elif query2:
            query = query2
        elif query1:
            query = query1

        if query:
            data = data.query(query)

        image_list = json.dumps(
            sorted(data[kwargs["image_column"]].to_list()), sort_keys=True
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

            data = data.join(
                existing_annotations, how="left", lsuffix="_x", rsuffix="_y"
            )
            data[kwargs["label_column"]] = data["label_y"].fillna(
                data[f"{kwargs['label_column']}_x"]
            )
            data = data.drop(
                columns=[
                    f"{kwargs['label_column']}_x",
                    f"{kwargs['label_column']}_y",
                ]
            )
            data["changed"] = data[kwargs["label_column"]].apply(
                lambda x: True if x else False
            )

            try:
                data[kwargs["image_column"]] = data[f"{kwargs['image_column']}_x"]
                data = data.drop(
                    columns=[
                        f"{kwargs['image_column']}_x",
                        f"{kwargs['image_column']}_y",
                    ]
                )
            except KeyError:
                # Looks like the columns don't exist, so leave it be.
                pass

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
        self.annotations_file = kwargs["annotations_file"]
        self.username = kwargs["username"]
        self.stop_at_last_example = kwargs["stop_at_last_example"]
        self.show_context = kwargs["show_context"]
        self.min_values = kwargs["min_values"]
        self.max_values = kwargs["max_values"]
        self.metadata = metadata
        self.patch_width, self.patch_height = self.get_patch_size()

        # Set current index
        self.current_index = -1

        # Set max buttons
        if not self.buttons_per_row:
            self.buttons_per_row = len(self.labels)

        # Setup buttons
        self._setup_buttons()

        # Setup box for buttons
        self._setup_box()

    def _load_frames(self, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    def _setup_buttons(self) -> None:
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

    def annotate(self, show_context=None) -> None:
        """
        Renders the annotation interface for the first image.

        Returns
        -------
        None
        """
        if show_context is not None:
            self.show_context = show_context

        self.out = widgets.Output()
        display(self.box)
        display(self.out)
        self._next_example()

    def _next_example(self, *args, **kwargs) -> None:
        if self.current_index < len(self):
            if len(args) == 1 and isinstance(args[0], int):
                self.current_index = args[0]
            else:
                self.current_index += 1
            self.render(**kwargs)

    def _prev_example(self, *args) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self.render()

    def render(self, *args, **kwargs) -> None:
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

        ix = self.iloc[self.current_index].name

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
            display(
                widgets.FloatSlider(
                    value=self.label.count(),
                    min=0,
                    max=len(self),
                    step=1,
                    description=f"{self.label.count()} / {len(self)}",
                    disabled=True,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=False,
                    # readout_format='.0f',
                )
            )
            if self.at[ix, "url"]:
                url = self.at[ix, "url"]
                display(
                    widgets.HTML(
                        f'<p><a href="{url}" target="_blank">Click to see entire map.</a></p>'
                    )
                )

    def _add_annotation(self, annotation: str) -> None:
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
        def get_path(image_path, dim=True):
            if dim:
                im = Image.open(image_path)
                alpha = Image.new("L", im.size, 90)
                im.putalpha(alpha)
                im.save(".temp.png")

                return widgets.Image.from_file(".temp.png")
            return widgets.Image.from_file(image_path)

        def get_empty_square():
            im = Image.new(
                size=(self.patch_width, self.patch_height),
                mode="RGB",
                color="white",
            )
            im.save(".temp.png")
            return widgets.Image.from_file(".temp.png")

        ix = self.iloc[self.current_index].name
        current_x = self.at[ix, "min_x"]
        current_y = self.at[ix, "min_y"]
        current_parent = self.at[ix, "parent_id"]
        image_path = self.at[ix, self.image_column]

        parent_frame = self.query(f"parent_id=='{current_parent}'")

        # self.get_patch_image(ix)

        one_less_x = current_x - self.patch_width  # -1 x
        one_more_x = current_x + self.patch_width  # +1 x

        one_less_y = current_y - self.patch_height  # -1 y
        one_more_y = current_y + self.patch_height  # +1 y

        nw = parent_frame.query(f"min_x == {one_less_x} & min_y == {one_less_y}")
        n = parent_frame.query(f"min_x == {current_x} & min_y == {one_less_y}")
        ne = parent_frame.query(f"min_x == {one_more_x} & min_y == {one_less_y}")

        nw = parent_frame.query(f"min_x == {one_less_x} & min_y == {one_less_y}")
        n = parent_frame.query(f"min_x == {current_x} & min_y == {one_less_y}")
        ne = parent_frame.query(f"min_x == {one_more_x} & min_y == {one_less_y}")

        w = parent_frame.query(f"min_x == {one_less_x} & min_y == {current_y}")
        e = parent_frame.query(f"min_x == {one_more_x} & min_y == {current_y}")

        sw = parent_frame.query(f"min_x == {one_less_x} & min_y == {one_more_y}")
        s = parent_frame.query(f"min_x == {current_x} & min_y == {one_more_y}")
        se = parent_frame.query(f"min_x == {one_more_x} & min_y == {one_more_y}")

        nw_image = nw.at[nw.index[0], "image_path"] if len(nw.index) == 1 else None
        n_image = n.at[n.index[0], "image_path"] if len(n.index) == 1 else None
        ne_image = ne.at[ne.index[0], "image_path"] if len(ne.index) == 1 else None
        w_image = w.at[w.index[0], "image_path"] if len(w.index) == 1 else None
        e_image = e.at[e.index[0], "image_path"] if len(e.index) == 1 else None
        sw_image = sw.at[sw.index[0], "image_path"] if len(sw.index) == 1 else None
        s_image = s.at[s.index[0], "image_path"] if len(s.index) == 1 else None
        se_image = se.at[se.index[0], "image_path"] if len(se.index) == 1 else None

        top_row = [
            get_path(x[0], dim=x[1]) if x[0] else get_empty_square()
            for x in [(nw_image, True), (n_image, True), (ne_image, True)]
        ]
        middle_row = [
            get_path(x[0], dim=x[1]) if x[0] else get_empty_square()
            for x in [(w_image, True), (image_path, False), (e_image, True)]
        ]
        bottom_row = [
            get_path(x[0], dim=x[1]) if x[0] else get_empty_square()
            for x in [(sw_image, True), (s_image, True), (se_image, True)]
        ]

        # drop temp file
        if os.path.exists(".temp.png"):
            os.remove(".temp.png")

        return widgets.VBox(
            [
                widgets.HBox(top_row),
                widgets.HBox(middle_row),
                widgets.HBox(bottom_row),
            ]
        )
