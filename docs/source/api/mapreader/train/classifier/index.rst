:py:mod:`mapreader.train.classifier`
====================================

.. py:module:: mapreader.train.classifier


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.train.classifier.classifier




.. py:class:: classifier(device = 'default')

   .. py:method:: set_classnames(classname_dict)

      Set the class names and the number of classes in the object detection
      model.

      Parameters
      ----------
      classname_dict : dict
          A dictionary containing the class IDs (as keys) and their
          corresponding names (as values). E.g.
          ``{0: "rail space", 1: "No rail space"}``

      Returns
      -------
      None


   .. py:method:: add2dataloader(dataset, set_name = None, batch_size = 16, shuffle = True, num_workers = 0, **kwds)

      Adds a PyTorch dataloader to the object's ``dataloader`` dictionary
      property and returns it.

      Parameters
      ----------
      dataset : torch.utils.data.Dataset
          The PyTorch dataset to use for the dataloader.
      set_name : str or None, optional
          The name to use when adding the dataloader to the object's
          ``dataloader`` dictionary property (e.g., ``"train"``, ``"val"``
          or ``"test"``).

          If ``None`` (default), the dataloader is returned without being
          added to the dictionary.
      batch_size : int, optional
          The batch size to use for the dataloader. Default is ``16``.
      shuffle : bool, optional
          Whether to shuffle the dataset during training. Default is
          ``True``.
      num_workers : int, optional
          The number of worker threads to use for loading data. Default is
          ``0``.
      **kwds :
          Additional keyword arguments to pass to PyTorch's ``DataLoader``
          constructor.

      Returns
      -------
      dl : torch.utils.data.DataLoader
          The dataloader that was created.


   .. py:method:: print_classes_dl(set_name = 'train')

      Prints information about the labels and class names (if available)
      associated with a dataloader.

      Parameters
      ----------
      set_name : str, optional
          The name of the dataloader to print information about, normally
          specified in ``self.add2dataloader``. Default is ``"train"``.

      Returns
      -------
      None


   .. py:method:: add_model(model, input_size = 224, is_inception = False)

      Add a PyTorch model to the classifier object.

      Parameters
      ----------
      model : nn.Module
          The PyTorch model to add to the object. See: ``torchvision.models``
      input_size : int, optional
          The expected input size of the model. Default is ``224``.
      is_inception : bool, optional
          Whether the model is an Inception-style model. Default is
          ``False``.

      Raises
      ------
      ValueError
          If the object's ``class_names`` attribute is ``None``. They should
          be specified with the ``set_classnames`` method.

      Returns
      -------
      None


   .. py:method:: del_model()

      Deletes the PyTorch model from the classifier object.

      Parameters
      ----------
      None

      Returns
      -------
      None

      Notes
      -----
      This function deletes the PyTorch model from the object and resets any
      associated metadata, such as the expected input size and whether the
      model is an Inception-style model. It also resets any associated
      metrics and best epoch/loss values.


   .. py:method:: layerwise_lr(min_lr, max_lr, ltype = 'linspace')

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

      Returns
      -------
      list of dicts
          A list of dictionaries containing the parameters and learning
          rates for each layer.


   .. py:method:: initialize_optimizer(optim_type = 'adam', params2optim = 'infer', optim_param_dict = {'lr': 0.001}, add_optim = True)

      Initializes an optimizer for the model and adds it to the classifier
      object.

      Parameters
      ----------
      optim_type : str, optional
          The type of optimizer to use. Can be set to ``"adam"`` (default),
          ``"adamw"``, or ``"sgd"``.
      params2optim : str or iterable, optional
          The parameters to optimize. If set to ``"infer"``, all model
          parameters that require gradients will be optimized, by default
          ``"infer"``.
      optim_param_dict : dict, optional
          The parameters to pass to the optimizer constructor as a
          dictionary, by default ``{"lr": 1e-3}``.
      add_optim : bool, optional
          If ``True``, adds the optimizer to the classifier object, by
          default ``True``.

      Returns
      -------
      optimizer : torch.optim.Optimizer
          The initialized optimizer. Only returned if ``add_optim`` is set to
          ``False``.

      Notes
      -----
      If ``add_optim`` is True, the optimizer will be added to object.

      Note that the first argument of an optimizer is parameters to optimize,
      e.g. ``params2optimize = model_ft.parameters()``:

      - ``model_ft.parameters()``: all parameters are being optimized
      - ``model_ft.fc.parameters()``: only parameters of final layer are being optimized

      Here, we use:

      .. code-block:: python

          filter(lambda p: p.requires_grad, self.model.parameters())


   .. py:method:: add_optimizer(optimizer)

      Add an optimizer to the classifier object.

      Parameters
      ----------
      optimizer : torch.optim.Optimizer
          The optimizer to add to the classifier object.

      Returns
      -------
      None


   .. py:method:: initialize_scheduler(scheduler_type = 'steplr', scheduler_param_dict = {'step_size': 10, 'gamma': 0.1}, add_scheduler = True)

      Initializes a learning rate scheduler for the optimizer and adds it to
      the classifier object.

      Parameters
      ----------
      scheduler_type : str, optional
          The type of learning rate scheduler to use. Can be either
          ``"steplr"`` (default) or ``"onecyclelr"``.
      scheduler_param_dict : dict, optional
          The parameters to pass to the scheduler constructor, by default
          ``{"step_size": 10, "gamma": 0.1}``.
      add_scheduler : bool, optional
          If ``True``, adds the scheduler to the classifier object, by
          default ``True``.

      Raises
      ------
      ValueError
          If the specified ``scheduler_type`` is not implemented.

      Returns
      -------
      scheduler : torch.optim.lr_scheduler._LRScheduler
          The initialized learning rate scheduler. Only returned if
          ``add_scheduler`` is set to False.


   .. py:method:: add_scheduler(scheduler)

      Add a scheduler to the classifier object.

      Parameters
      ----------
      scheduler : torch.optim.lr_scheduler._LRScheduler
          The scheduler to add to the classifier object.

      Raises
      ------
      ValueError
          If no optimizer has been set. Use ``initialize_optimizer`` or
          ``add_optimizer`` to set an optimizer first.

      Returns
      -------
      None


   .. py:method:: add_criterion(criterion)

      Add a loss criterion to the classifier object.

      Parameters
      ----------
      criterion : torch.nn.modules.loss._Loss
          The loss criterion to add to the classifier object.

      Returns
      -------
      None
          The function only modifies the ``criterion`` attribute of the
          classifier and does not return anything.


   .. py:method:: model_summary(only_trainable = False, print_space = [40, 20, 20])

      Print a summary of the model including the modules, the number of
      parameters in each module, and the dimension of the output tensor of
      each module. If ``only_trainable`` is ``True``, it only prints the
      trainable parameters.

      Other ways to check params:

      .. code-block:: python

          sum(p.numel() for p in myclassifier.model.parameters())

      .. code-block:: python

          sum(p.numel() for p in myclassifier.model.parameters()
              if p.requires_grad)

      And:

      .. code-block:: python

          for name, param in self.model.named_parameters():
              n = name.split(".")[0].split("_")[0]
              print(name, param.requires_grad)

      Parameters
      ----------
      only_trainable : bool, optional
          If ``True``, only the trainable parameters will be printed.
          Defaults to ``False``.
      print_space : list, optional
          A list with three integers defining the width of each column in
          the printed table. By default, ``[40, 20, 20]``.

      Returns
      -------
      None

      Notes
      -----
      Credit: this function is the modified version of
      https://stackoverflow.com/a/62508086.


   .. py:method:: freeze_layers(layers_to_freeze = [])

      Freezes the specified layers in the neural network by setting
      ``requires_grad`` attribute to False for their parameters.

      Parameters
      ----------
      layers_to_freeze : list of str, optional
          List of names of the layers to freeze. If a layer name ends with
          an asterisk (``"*"``), then all parameters whose name contains the
          layer name (excluding the asterisk) are frozen. Otherwise,
          only the parameters with an exact match to the layer name
          are frozen. By default, ``[]``.

      Returns
      -------
      None
          The function only modifies the ``requires_grad`` attribute of the
          specified parameters and does not return anything.

      Notes
      -----
      Wildcards are accepted in the ``layers_to_freeze`` parameter.


   .. py:method:: unfreeze_layers(layers_to_unfreeze = [])

      Unfreezes the specified layers in the neural network by setting
      ``requires_grad`` attribute to True for their parameters.

      Parameters
      ----------
      layers_to_unfreeze : list of str, optional
          List of names of the layers to unfreeze. If a layer name ends with
          an asterisk (``"*"``), then all parameters whose name contains the
          layer name (excluding the asterisk) are unfrozen. Otherwise,
          only the parameters with an exact match to the layer name
          are unfrozen. By default, ``[]``.

      Returns
      -------
      None
          The function only modifies the ``requires_grad`` attribute of the
          specified parameters and does not return anything.

      Notes
      -----
      Wildcards are accepted in the ``layers_to_unfreeze`` parameter.


   .. py:method:: only_keep_layers(only_keep_layers_list = [])

      Only keep the specified layers (``only_keep_layers_list``) for
      gradient computation during the backpropagation.

      Parameters
      ----------
      only_keep_layers_list : list, optional
          List of layer names to keep. All other layers will have their
          gradient computation turned off. Default is ``[]``.

      Returns
      -------
      None
          The function only modifies the ``requires_grad`` attribute of the
          specified parameters and does not return anything.


   .. py:method:: inference(set_name = 'infer', verbosity_level = 0, print_info_batch_freq = 5)

      Run inference on a specified dataset (``set_name``).

      Parameters
      ----------
      set_name : str, optional
          The name of the dataset to run inference on, by default
          ``"infer"``.
      verbosity_level : int, optional
          The verbosity level of the output messages, by default ``0``.
      print_info_batch_freq : int, optional
          The frequency of printouts, by default ``5``.

      Returns
      -------
      None

      Notes
      -----
      This method calls the
      :meth:`mapreader.train.classifier.classifier.train` method with the
      ``num_epochs`` set to ``1`` and all the other parameters specified in
      the function arguments.


   .. py:method:: train_component_summary()

      Print a summary of the optimizer, criterion and trainable model
      components.

      Returns:
      --------
      None


   .. py:method:: train(phases = ['train', 'val'], num_epochs = 25, save_model_dir = 'models', verbosity_level = 1, tensorboard_path = None, tmp_file_save_freq = 2, remove_after_load = True, print_info_batch_freq = 5)

      Train the model on the specified phases for a given number of epochs.

      Wrapper function for
      :meth:`mapreader.train.classifier.classifier.train_core` method to
      capture exceptions (``KeyboardInterrupt`` is the only supported
      exception currently).

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

      Notes
      -----
      Refer to the documentation of
      :meth:`mapreader.train.classifier.classifier.train_core` for more
      information.


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
          If the specified phase cannot be found in the keys of the object's
          ``dataloader`` dictionary property.

      Returns
      -------
      None


   .. py:method:: calculate_add_metrics(y_true, y_pred, y_score, phase, epoch = -1, tboard_writer=None)

      Calculate and add metrics to the classifier's metrics dictionary.

      Parameters
      ----------
      y_true : array-like of shape (n_samples,)
          True binary labels or multiclass labels. Can be considered ground
          truth or (correct) target values.

      y_pred : array-like of shape (n_samples,)
          Predicted binary labels or multiclass labels. The estimated
          targets as returned by a classifier.

      y_score : array-like of shape (n_samples, n_classes)
          Predicted probabilities for each class. Only required when
          ``y_pred`` is not binary.

      phase : str
          Name of the current phase, typically ``"train"`` or ``"val"``. See
          ``train`` function.

      epoch : int, optional
          Current epoch number. Default is ``-1``.

      tboard_writer : object, optional
          TensorBoard SummaryWriter object to write the metrics. Default is
          ``None``.

      Returns
      -------
      None

      Notes
      -----
      This method uses both the
      ``sklearn.metrics.precision_recall_fscore_support`` and
      ``sklearn.metrics.roc_auc_score`` functions from ``scikit-learn`` to
      calculate the metrics for each average type (``"micro"``, ``"macro"``
      and ``"weighted"``). The results are then added to the ``metrics``
      dictionary. It also writes the metrics to the TensorBoard
      SummaryWriter, if ``tboard_writer`` is not None.


   .. py:method:: gen_epoch_msg(phase, epoch_msg)

      Generates a log message for an epoch during training or validation.
      The message includes information about the loss, F-score, and recall
      for a given phase (training or validation).

      Parameters
      ----------
      phase : str
          The training phase, either ``"train"`` or ``"val"``.
      epoch_msg : str
          The message string to be modified with the epoch metrics.

      Returns
      -------
      epoch_msg : str
          The updated message string with the epoch metrics.


   .. py:method:: plot_metric(y_axis, y_label, legends, x_axis = 'epoch', x_label = 'epoch', colors = 5 * ['k', 'tab:red'], styles = 10 * ['-'], markers = 10 * ['o'], figsize = (10, 5), plt_yrange = None, plt_xrange = None)

      Plot the metrics of the classifier object.

      Parameters
      ----------
      y_axis : list of str
          A list of metric names to be plotted on the y-axis.
      y_label : str
          The label for the y-axis.
      legends : list of str
          The legend labels for each metric.
      x_axis : str, optional
          The metric to be used as the x-axis. Can be ``"epoch"`` (default)
          or any other metric name present in the dataset.
      x_label : str, optional
          The label for the x-axis. Defaults to ``"epoch"``.
      colors : list of str, optional
          The colors to be used for the lines of each metric. It must be at
          least the same size as ``y_axis``. Defaults to
          ``5 * ["k", "tab:red"]``.
      styles : list of str, optional
          The line styles to be used for the lines of each metric. It must
          be at least the same size as ``y_axis``. Defaults to
          ``10 * ["-"]``.
      markers : list of str, optional
          The markers to be used for the lines of each metric. It must be at
          least the same size as ``y_axis``. Defaults to ``10 * ["o"]``.
      figsize : tuple of int, optional
          The size of the figure in inches. Defaults to ``(10, 5)``.
      plt_yrange : tuple of float, optional
          The range of values for the y-axis. Defaults to ``None``.
      plt_xrange : tuple of float, optional
          The range of values for the x-axis. Defaults to ``None``.

      Returns
      -------
      None

      Notes
      -----
      This function requires the ``matplotlib`` package.


   .. py:method:: initialize_model(model_name, pretrained = True, last_layer_num_classes = 'default', add_model = True)

      Initializes a PyTorch model with the option to change the number of
      classes in the last layer (``last_layer_num_classes``).

      The function handles six PyTorch models: ResNet, AlexNet, VGG,
      SqueezeNet, DenseNet, and Inception v3.

      Parameters
      ----------
      model_name : str
          Name of a PyTorch model. See
          https://pytorch.org/vision/0.8/models.html
      pretrained : bool, optional
          Use pretrained version, by default ``True``
      last_layer_num_classes : str or int, optional
          Number of elements in the last layer. If ``"default"``, sets it to
          the number of classes. By default, ``"default"``.
      add_model : bool, optional
          If ``True`` (default), adds the initialized model to the instance
          of the class.

      Returns
      -------
      model : PyTorch model
          The initialized PyTorch model with the changed last layer.
      input_size : int
          Input size of the model.
      is_inception : bool
          True if the model is Inception v3.

      Raises
      ------
      ValueError
          If an invalid model name is passed.

      Notes
      -----
      Inception v3 requires the input size to be ``(299, 299)``, whereas all
      of the other models expect ``(224, 224)``.

      See https://pytorch.org/vision/0.8/models.html for available models.


   .. py:method:: show_sample(set_name = 'train', batch_number = 1, print_batch_info = True, figsize = (15, 10))

      Displays a sample of training or validation data in a grid format with
      their corresponding class labels.

      Parameters
      ----------
      set_name : str, optional
          Name of the dataset (``"train"``/``"validation"``) to display the
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


   .. py:method:: batch_info(set_name = 'train')

      Print information about a dataset's batches, samples, and batch-size.

      Parameters
      ----------
      set_name : str, optional
          Name of the dataset to display batch information for (default is
          ``"train"``).

      Returns
      -------
      None


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


   .. py:method:: save(save_path = 'default.obj', force = False)

      Save the object to a file.

      Parameters
      ----------
      save_path : str, optional
          The path to the file to write. If the file already exists and
          ``force`` is not ``True``, a ``FileExistsError`` is raised.
          Defaults to ``"default.obj"``.
      force : bool, optional
          Whether to overwrite the file if it already exists. Defaults to
          ``False``.

      Raises
      ------
      FileExistsError
          If the file already exists and ``force`` is not ``True``.

      Notes
      -----
      The object is saved in two parts. First, a serialized copy of the
      object's dictionary is written to the specified file using the
      ``joblib.dump`` function. The object's ``model`` attribute is excluded
      from this dictionary and saved separately using the ``torch.save``
      function, with a filename derived from the original ``save_path``.


   .. py:method:: load(load_path, remove_after_load = False, force_device = False)

      This function loads the state of a class instance from a saved file
      using the joblib library. It also loads a PyTorch model from a
      separate file and maps it to the device used to load the class
      instance.

      Parameters
      ----------
      load_path : str
          Path to the saved file to load.
      remove_after_load : bool, optional
          Whether to remove the saved file after loading. Defaults to
          ``False``.
      force_device : bool or str, optional
          Whether to force the use of a specific device, or the name of the
          device to use. If set to ``True``, the default device is used.
          Defaults to ``False``.

      Raises
      ------
      FileNotFoundError
          If the specified file does not exist.

      Modifies
      ----------
      self.__dict__ : dict
          The state of the class instance is updated with the contents of
          the saved file.
      os.environ["CUDA_VISIBLE_DEVICES"] : str
          The CUDA_VISIBLE_DEVICES environment variable is updated if the
          ``force_device`` argument is specified.

      Returns
      -------
      None


   .. py:method:: get_time()

      Get the current date and time as a formatted string.

      Returns
      -------
      str
          A string representing the current date and time.


   .. py:method:: cprint(type_info, bc_color, text)

      Print colored text with additional information.

      Parameters
      ----------
      type_info : str
          The type of message to display.
      bc_color : str
          The color to use for the message text.
      text : str
          The text to display.

      Returns
      -------
      None
          The colored message is displayed on the standard output stream.


   .. py:method:: update_progress(progress, text = '', barLength = 30)

      Update the progress bar.

      Parameters
      ----------
      progress : float or int
          The progress value to display, between ``0`` and ``1``.
          If an integer is provided, it will be converted to a float.
          If a value outside the range ``[0, 1]`` is provided, it will be
          clamped to the nearest valid value.
      text : str, optional
          Additional text to display after the progress bar, defaults to
          ``""``.
      barLength : int, optional
          The length of the progress bar in characters, defaults to ``30``.

      Raises
      ------
      TypeError
          If progress is not a floating point value or an integer.

      Returns
      -------
      None
          The progress bar is displayed on the standard output stream.



