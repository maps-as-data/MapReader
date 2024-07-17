mapreader.spot_text.runner_base
===============================

.. py:module:: mapreader.spot_text.runner_base


Classes
-------

.. autoapisummary::

   mapreader.spot_text.runner_base.Runner


Module Contents
---------------

.. py:class:: Runner

   .. py:method:: run_all(patch_df = None, return_dataframe = False, min_ioa = 0.7)

      Run the model on all images in the patch dataframe.

      :param patch_df: Dataframe containing patch information, by default None.
      :type patch_df: pd.DataFrame, optional
      :param return_dataframe: Whether to return the predictions as a pandas DataFrame, by default False
      :type return_dataframe: bool, optional
      :param min_ioa: The minimum intersection over area to consider two polygons the same, by default 0.7
      :type min_ioa: float, optional

      :returns: A dictionary of predictions for each patch image or a DataFrame if `as_dataframe` is True.
      :rtype: dict or pd.DataFrame



   .. py:method:: run_on_images(img_paths, return_dataframe = False, min_ioa = 0.7)

      Run the model on a list of images.

      :param img_paths: A list of image paths to run the model on.
      :type img_paths: str, pathlib.Path or list
      :param return_dataframe: Whether to return the predictions as a pandas DataFrame, by default False
      :type return_dataframe: bool, optional
      :param min_ioa: The minimum intersection over area to consider two polygons the same, by default 0.7
      :type min_ioa: float, optional

      :returns: A dictionary of predictions for each image or a DataFrame if `as_dataframe` is True.
      :rtype: dict or pd.DataFrame



   .. py:method:: run_on_image(img_path, return_outputs=False, return_dataframe = False, min_ioa = 0.7)

      Run the model on a single image.

      :param img_path: The path to the image to run the model on.
      :type img_path: str or pathlib.Path
      :param return_outputs: Whether to return the outputs direct from the model, by default False
      :type return_outputs: bool, optional
      :param return_dataframe: Whether to return the predictions as a pandas DataFrame, by default False
      :type return_dataframe: bool, optional
      :param min_ioa: The minimum intersection over area to consider two polygons the same, by default 0.7
      :type min_ioa: float, optional

      :returns: The predictions for the image or the outputs from the model if `return_outputs` is True.
      :rtype: dict or pd.DataFrame



   .. py:method:: convert_to_parent_pixel_bounds(patch_df = None, return_dataframe = False, deduplicate = False, min_ioa = 0.7)

      Convert the patch predictions to parent predictions by converting pixel bounds.

      :param patch_df: Dataframe containing patch information, by default None
      :type patch_df: pd.DataFrame, optional
      :param return_dataframe: Whether to return the predictions as a pandas DataFrame, by default False
      :type return_dataframe: bool, optional
      :param deduplicate: Whether to deduplicate the parent predictions, by default False.
                          Depending on size of parent images, this can be slow.
      :type deduplicate: bool, optional
      :param min_ioa: The minimum intersection over area to consider two polygons the same, by default 0.7
                      This is only used if `deduplicate` is True.
      :type min_ioa: float, optional

      :returns: A dictionary of predictions for each parent image or a DataFrame if `as_dataframe` is True.
      :rtype: dict or pd.DataFrame

      :raises ValueError: If `patch_df` is not available.



   .. py:method:: convert_to_coords(parent_df = None, return_dataframe = False)

      Convert the parent predictions to georeferenced predictions by converting pixel bounds to coordinates.

      :param parent_df: Dataframe containing parent image information, by default None
      :type parent_df: pd.DataFrame, optional
      :param return_dataframe: Whether to return the predictions as a pandas DataFrame, by default False
      :type return_dataframe: bool, optional

      :returns: A dictionary of predictions for each parent image or a DataFrame if `as_dataframe` is True.
      :rtype: dict or pd.DataFrame

      :raises ValueError: If `parent_df` is not available.



   .. py:method:: save_to_geojson(save_path)

      Save the georeferenced predictions to a GeoJSON file.

      :param save_path: Path to save the GeoJSON file
      :type save_path: str | pathlib.Path, optional



   .. py:method:: show(image_id, figsize = (10, 10), border_color = 'r', text_color = 'b', image_width_resolution = None, return_fig = False)

      Show the predictions on an image.

      :param image_id: The image ID to show the predictions on.
      :type image_id: str
      :param figsize: The size of the figure, by default (10, 10)
      :type figsize: tuple | None, optional
      :param border_color: The color of the border of the polygons, by default "r"
      :type border_color: str | None, optional
      :param text_color: The color of the text, by default "b"
      :type text_color: str | None, optional
      :param image_width_resolution: The maximum resolution of the image width, by default None
      :type image_width_resolution: int | None, optional
      :param return_fig: Whether to return the figure, by default False
      :type return_fig: bool, optional

      :returns: The matplotlib figure if `return_fig` is True.
      :rtype: fig

      :raises ValueError: If the image ID is not found in the patch or parent predictions.
