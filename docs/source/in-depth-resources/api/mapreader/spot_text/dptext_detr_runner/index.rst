mapreader.spot_text.dptext_detr_runner
======================================

.. py:module:: mapreader.spot_text.dptext_detr_runner


Classes
-------

.. autoapisummary::

   mapreader.spot_text.dptext_detr_runner.DPTextDETRRunner


Module Contents
---------------

.. py:class:: DPTextDETRRunner(patch_df = None, parent_df = None, cfg_file = './DPText-DETR/configs/DPText_DETR/ArT/R_50_poly.yaml', weights_file = './art_final.pth', device = 'cpu')

   Bases: :py:obj:`mapreader.spot_text.runner_base.Runner`


   .. py:method:: get_patch_predictions(outputs, return_dataframe = False, min_ioa = 0.7)

      Post process the model outputs to get patch predictions.

      :param outputs: The outputs from the model.
      :type outputs: dict
      :param return_dataframe: Whether to return the predictions as a pandas DataFrame, by default False
      :type return_dataframe: bool, optional
      :param min_ioa: The minimum intersection over area to consider two polygons the same, by default 0.7
      :type min_ioa: float, optional

      :returns: A dictionary containing the patch predictions or a DataFrame if `as_dataframe` is True.
      :rtype: dict or pd.DataFrame
