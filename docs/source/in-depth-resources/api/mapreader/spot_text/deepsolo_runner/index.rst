mapreader.spot_text.deepsolo_runner
===================================

.. py:module:: mapreader.spot_text.deepsolo_runner


Classes
-------

.. autoapisummary::

   mapreader.spot_text.deepsolo_runner.DeepSoloRunner


Module Contents
---------------

.. py:class:: DeepSoloRunner(patch_df = None, parent_df = None, cfg_file = './DeepSolo/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml', weights_file = './ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth', device = 'cpu')

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
