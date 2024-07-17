mapreader.process.post_process
==============================

.. py:module:: mapreader.process.post_process


Classes
-------

.. autoapisummary::

   mapreader.process.post_process.PostProcessor


Module Contents
---------------

.. py:class:: PostProcessor(patch_df, labels_map)

   A class for post-processing predictions on patches using the surrounding context.

   :param patch_df: the DataFrame containing patches and predictions
   :type patch_df: pd.DataFrame
   :param labels_map: the dictionary mapping label indices to their labels.
                      e.g. `{0: "no", 1: "railspace"}`.
   :type labels_map: dict


   .. py:method:: get_context(labels)

      Get the context of the patches with the specified labels.

      :param labels: The label(s) to get context for.
      :type labels: str | list



   .. py:method:: update_preds(remap, conf = 0.7, inplace = False)

      Update the predictions of the chosen patches based on their context.

      :param remap: A dictionary mapping the old labels to the new labels.
      :type remap: dict
      :param conf: Patches with confidence scores below this value will be relabelled, by default 0.7.
      :type conf: float, optional
      :param inplace: Whether to relabel inplace or create new dataframe columns, by default False
      :type inplace: bool, optional
