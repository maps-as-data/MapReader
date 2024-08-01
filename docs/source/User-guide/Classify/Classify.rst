Classify (Train and Infer)
==========================

.. note:: Run these commands in a Jupyter notebook (or other IDE), ensuring you are in your `mapreader` python environment.

.. note:: You will need to update file paths to reflect your own machines directory structure.

MapReader's ``Classify`` subpackage is used to:

- Train or fine-tune a classifier on annotated patches.
- Use a classifier to infer/predict the labels of unannotated patches.

This is all done within MapReader's ``ClassifierContainer()`` class, which is used to:

- Load models (classifiers).
- Define a labels map.
- Load datasets and dataloaders.
- Define a loss function, optimizer and scheduler.
- Train and evaluate models using already annotated images.
- Predict labels of unannotated images (model inference).
- Visualize datasets and predictions.

If you already have a fine-tuned model, you can skip to the :doc:`Infer labels using a fine-tuned model </User-guide/Classify/Infer>` page.

If not, you should proceed to the :doc:`Train/fine-tune a classifier </User-guide/Classify/Train>` page.

.. toctree::
   :maxdepth: 1

   Train
   Infer
