:py:mod:`mapreader.learn.classifier_context`
============================================

.. py:module:: mapreader.learn.classifier_context


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.learn.classifier_context.classifierContext




.. py:class:: classifierContext(device = 'default')

   Bases: :py:obj:`mapreader.learn.classifier.classifier`

   .. py:method:: train(phases = ['train', 'val'], num_epochs = 25, save_model_dir = 'models', verbosity_level = 1, tensorboard_path = None, tmp_file_save_freq = 2, remove_after_load = True, print_info_batch_freq = 5)

      Train the model on the specified phases for a given number of epochs.
      Wrapper function for ``train_core`` method to capture exceptions (with
      supported exceptions so far: ``KeyboardInterrupt``). Refer to
      ``train_core`` for more information.

      Parameters
      ----------
      phases : list of str, optional
          The phases to train the model on for each epoch. Default is
          ``["train", "val"]``.
      num_epochs : int, optional
          The number of epochs to train the model for. Default is ``25``.
      save_model_dir : str or None, optional
          The directory to save the model in. Default is ``"models"``. If
          set to ``None``, the model is not saved.
      verbosity_level : int, optional
          The level of verbosity during training:

          - ``0`` is silent,
          - ``1`` is progress bar and metrics,
          - ``2`` is detailed information.

          Default is ``1``.
      tensorboard_path : str or None, optional
          The path to the directory to save TensorBoard logs in. If set to
          ``None``, no TensorBoard logs are saved. Default is ``None``.
      tmp_file_save_freq : int, optional
          The frequency (in epochs) to save a temporary file of the model.
          Default is ``2``. If set to ``0`` or ``None``, no temporary file
          is saved.
      remove_after_load : bool, optional
          Whether to remove the temporary file after loading it. Default is
          ``True``.
      print_info_batch_freq : int, optional
          The frequency (in batches) to print training information. Default
          is ``5``. If set to ``0`` or ``None``, no training information is
          printed.

      Returns
      -------
      None
          The function saves the model to the ``save_model_dir`` directory,
          and optionally to a temporary file. If interrupted with a
          ``KeyboardInterrupt``, the function tries to load the temporary
          file. If no temporary file is found, it continues without loading.


   .. py:method:: train_core(phases = ['train', 'val'], num_epochs = 25, save_model_dir = 'models', verbosity_level = 1, tensorboard_path = None, tmp_file_save_freq = 2, print_info_batch_freq = 5)

      Trains/fine-tunes a classifier for the specified number of epochs on
      the given phases using the specified hyperparameters.

      Parameters
      ----------
      phases : list of str, optional
          The phases to train the model on for each epoch. Default is
          ``["train", "val"]``.
      num_epochs : int, optional
          The number of epochs to train the model for. Default is ``25``.
      save_model_dir : str or None, optional
          The directory to save the model in. Default is ``"models"``. If
          set to ``None``, the model is not saved.
      verbosity_level : int, optional
          The level of verbosity during training:

          - ``0`` is silent,
          - ``1`` is progress bar and metrics,
          - ``2`` is detailed information.

          Default is ``1``.
      tensorboard_path : str or None, optional
          The path to the directory to save TensorBoard logs in. If set to
          ``None``, no TensorBoard logs are saved. Default is ``None``.
      tmp_file_save_freq : int, optional
          The frequency (in epochs) to save a temporary file of the model.
          Default is ``2``. If set to ``0`` or ``None``, no temporary file
          is saved.
      print_info_batch_freq : int, optional
          The frequency (in batches) to print training information. Default
          is ``5``. If set to ``0`` or ``None``, no training information is
          printed.

      Raises
      ------
      ValueError
          If the criterion is not set. Use the ``add_criterion`` method to
          set the criterion.

          If the optimizer is not set and the phase is "train". Use the
          ``initialize_optimizer`` or ``add_optimizer`` method to set the
          optimizer.

      KeyError
          If the specified phase cannot be found in the object's dataloader
          with keys.

      Returns
      -------
      None


   .. py:method:: show_sample(set_name = 'train', batch_number = 1, print_batch_info = True, figsize = (15, 10))

      Displays a sample of training or validation data in a grid format with
      their corresponding class labels.

      Parameters
      ----------
      set_name : str, optional
          Name of the dataset (``train``/``validation``) to display the
          sample from, by default ``"train"``.
      batch_number : int, optional
          Number of batches to display, by default ``1``.
      print_batch_info : bool, optional
          Whether to print information about the batch size, by default
          ``True``.
      figsize : tuple, optional
          Figure size (width, height) in inches, by default ``(15, 10)``.

      Returns
      -------
      None
          Displays the sample images with their corresponding class labels.

      Raises
      ------
      StopIteration
          If the specified number of batches to display exceeds the total
          number of batches in the dataset.

      Notes
      -----
      This method uses the dataloader of the ``ImageClassifierData`` class
      and the ``torchvision.utils.make_grid`` function to display the sample
      data in a grid format. It also calls the ``_imshow`` method of the
      ``ImageClassifierData`` class to show the sample data.


   .. py:method:: layerwise_lr(min_lr, max_lr, ltype = 'linspace', sep_group_names = ['features1', 'features2'])

      Calculates layer-wise learning rates for a given set of model
      parameters.

      Parameters
      ----------
      min_lr : float
          The minimum learning rate to be used.
      max_lr : float
          The maximum learning rate to be used.
      ltype : str, optional
          The type of sequence to use for spacing the specified interval
          learning rates. Can be either ``"linspace"`` or ``"geomspace"``,
          where `"linspace"` uses evenly spaced learning rates over a
          specified interval and `"geomspace"` uses learning rates spaced
          evenly on a log scale (a geometric progression). Defaults to
          ``"linspace"``.
      sep_group_names : list, optional
          A list of strings containing the names of parameter groups. Layers
          belonging to each group will be assigned the same learning rate.
          Defaults to ``["features1", "features2"]``.

      Returns
      -------
      list of dicts
          A list of dictionaries containing the parameters and learning
          rates for each layer.


   .. py:method:: inference_sample_results(num_samples = 6, class_index = 0, set_name = 'train', min_conf = None, max_conf = None, figsize = (15, 15))

      Performs inference on a given dataset and displays results for a
      specified class.

      Parameters
      ----------
      num_samples : int, optional
          The number of sample results to display. Defaults to ``6``.
      class_index : int, optional
          The index of the class for which to display results. Defaults to
          ``0``.
      set_name : str, optional
          The name of the dataset split to use for inference. Defaults to
          ``"train"``.
      min_conf : float, optional
          The minimum confidence score for a sample result to be displayed.
          Samples with lower confidence scores will be skipped. Defaults to
          ``None``.
      max_conf : float, optional
          The maximum confidence score for a sample result to be displayed.
          Samples with higher confidence scores will be skipped. Defaults to
          ``None``.
      figsize : tuple[int, int], optional
          Figure size (width, height) in inches, displaying the sample
          results. Defaults to ``(15, 15)``.

      Returns
      -------
      None



