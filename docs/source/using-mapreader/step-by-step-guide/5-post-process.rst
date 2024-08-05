Post-process
=============

MapReader post-processing's sub-package currently contains one method for post-processing the predictions from your model based on the idea that features such as railways, roads, coastlines, etc. are continuous and so patches with these labels should be found near to other patches also with these labels.
For example, if a patch is predicted to be a railspace, but is surrounded by patches predicted to be non-railspace, then it is likely that the railspace patch is a false positive.

To implement this, for a given patch, the code checks whether any of the 8 surrounding patches have the same label (e.g. 'railspace') and, if not, assumes the current patch's predicted label to be a false positive.
The user can then choose how to relabel the patch (e.g. 'railspace' -> 'no').

To run the post-processing code, you will need to have saved the predictions from your model in the format expected for the post-processing code.
See the :doc:`/using-mapreader/step-by-step-guide/4-classify/index` docs for more on this.

If you have your predictions saved in a ``csv`` file, you will first need to load them into a Pandas DataFrame:

.. code-block:: python

    import pandas as pd

    preds = pd.read_csv("path/to/predictions.csv", index_col=0)


You can then run the post-processing code as follows:

.. code-block:: python

    from mapreader.process.post_process import PostProcessor

    labels_map = {
        0: "no",
        1: "railspace",
        2: "building",
        3: "railspace&building"
    }

    patches = PostProcessor(preds, labels_map=labels_map)

MapReader's post-processing will only work for features that are expected be continuous (e.g. railway, road, coastline, etc.) or clustered (e.g. a large body of water).
You will need to tell MapReader which labels to select and then get the context for each of the relevant patches in order to work out if it is isolated or part of a line/cluster.

For example, if you want to post-process patches which are predicted to be 'railspace' or 'railspace&building', you would do the following:

.. code-block:: python

    labels=["railspace", "railspace&building"]
    patches.get_context(labels=labels)


.. note:: In the above example, we needed to use both 'railspace' and 'railspace&building' as our labels, since the continuous feature we are trying to post-process is railway lines (included in both these labels).

You will also need to tell MapReader how to update the label of each patch that is isolated and therefore likely to be a false positive.
This is done using the ``remap`` argument, which takes a dictionary of the form ``{old_label: new_label}``.

For example, if you want to remap all isolated 'railspace' patches to be labelled as 'no', and all isolated 'railspace&building' patches to be labelled as 'building', you would do the following:

.. code-block:: python

    remap={"railspace": "no", "railspace&building": "building"}
    patches.update_preds(remap=remap)

By default, only patches with model confidence of below 0.7 will be relabelled.
You can adjust this by passing the ``conf`` argument.

e.g. to relabel all isolated patches with confidence below 0.9, you would do the following:

.. code-block:: python

    remap={"railspace": "no", "railspace&building": "building"}
    patches.update_preds(remap=remap, conf=0.9)

Instead of relabelling your chosen patches to an existing label, you can also choose to relabel them to a new label.
For example, to mark them as 'false_positive', you would do the following:

.. code-block:: python

    remap={"railspace": "false_positive", "railspace&building": "false_positive"}
    patches.update_preds(remap=remap)


By default, after running `update_preds`, a new column will be added to your ``patches`` DataFrame called "new_predicted_label".
This will contain the updated predictions (or NaN if the patch was not relabelled).

Alternatively, to save the updated predictions inplace you can pass the ``inplace`` argument:

.. code-block:: python

    remap={"railspace": "no", "railspace&building": "building"}
    patches.update_preds(remap=remap, inplace=True)


Finally, to save your outputs to a csv file, you can do the following:

.. code-block:: python

    patches.to_csv("path/to/save/updated_predictions.csv")
