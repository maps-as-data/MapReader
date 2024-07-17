mapreader.classify.classifier
=============================

.. py:module:: mapreader.classify.classifier


Classes
-------

.. autoapisummary::

   mapreader.classify.classifier.ClassifierContainer


Module Contents
---------------

.. py:class:: ClassifierContainer(model, labels_map, dataloaders = None, device = 'default', input_size = (224, 224), is_inception = False, load_path = None, force_device = False, **kwargs)

   .. py:method:: generate_layerwise_lrs(min_lr, max_lr, spacing = 'linspace')

      Calculates layer-wise learning rates for a given set of model
      parameters.

      :param min_lr: The minimum learning rate to be used.
      :type min_lr: float
      :param max_lr: The maximum learning rate to be used.
      :type max_lr: float
      :param spacing: The type of sequence to use for spacing the specified interval
                      learning rates. Can be either ``"linspace"`` or ``"geomspace"``,
                      where `"linspace"` uses evenly spaced learning rates over a
                      specified interval and `"geomspace"` uses learning rates spaced
                      evenly on a log scale (a geometric progression). By default ``"linspace"``.
      :type spacing: str, optional

      :returns: A list of dictionaries containing the parameters and learning
                rates for each layer.
      :rtype: list of dicts



   .. py:method:: initialize_optimizer(optim_type = 'adam', params2optimize = 'default', optim_param_dict = None, add_optim = True)

      Initializes an optimizer for the model and adds it to the classifier
      object.

      :param optim_type: The type of optimizer to use. Can be set to ``"adam"`` (default),
                         ``"adamw"``, or ``"sgd"``.
      :type optim_type: str, optional
      :param params2optimize: The parameters to optimize. If set to ``"default"``, all model
                              parameters that require gradients will be optimized.
                              Default is ``"default"``.
      :type params2optimize: str or iterable, optional
      :param optim_param_dict: The parameters to pass to the optimizer constructor as a
                               dictionary, by default ``{"lr": 1e-3}``.
      :type optim_param_dict: dict, optional
      :param add_optim: If ``True``, adds the optimizer to the classifier object, by
                        default ``True``.
      :type add_optim: bool, optional

      :returns: **optimizer** -- The initialized optimizer. Only returned if ``add_optim`` is set to
                ``False``.
      :rtype: torch.optim.Optimizer

      .. rubric:: Notes

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

      :param optimizer: The optimizer to add to the classifier object.
      :type optimizer: torch.optim.Optimizer

      :rtype: None



   .. py:method:: initialize_scheduler(scheduler_type = 'steplr', scheduler_param_dict = None, add_scheduler = True)

      Initializes a learning rate scheduler for the optimizer and adds it to
      the classifier object.

      :param scheduler_type: The type of learning rate scheduler to use. Can be either
                             ``"steplr"`` (default) or ``"onecyclelr"``.
      :type scheduler_type: str, optional
      :param scheduler_param_dict: The parameters to pass to the scheduler constructor, by default
                                   ``{"step_size": 10, "gamma": 0.1}``.
      :type scheduler_param_dict: dict, optional
      :param add_scheduler: If ``True``, adds the scheduler to the classifier object, by
                            default ``True``.
      :type add_scheduler: bool, optional

      :raises ValueError: If the specified ``scheduler_type`` is not implemented.

      :returns: **scheduler** -- The initialized learning rate scheduler. Only returned if
                ``add_scheduler`` is set to False.
      :rtype: torch.optim.lr_scheduler._LRScheduler



   .. py:method:: add_scheduler(scheduler)

      Add a scheduler to the classifier object.

      :param scheduler: The scheduler to add to the classifier object.
      :type scheduler: torch.optim.lr_scheduler._LRScheduler

      :raises ValueError: If no optimizer has been set. Use ``initialize_optimizer`` or
          ``add_optimizer`` to set an optimizer first.

      :rtype: None



   .. py:method:: add_criterion(criterion = 'cross entropy')

      Add a loss criterion to the classifier object.

      :param criterion: The loss criterion to add to the classifier object.
                        Accepted string values are "cross entropy" or "ce" (cross-entropy), "bce" (binary cross-entropy) and "mse" (mean squared error).
      :type criterion: str or torch.nn.modules.loss._Loss

      :returns: The function only modifies the ``criterion`` attribute of the
                classifier and does not return anything.
      :rtype: None



   .. py:method:: model_summary(input_size = None, trainable_col = False, **kwargs)

      Print a summary of the model.

      :param input_size: The size of the input data.
                         If None, input size is taken from "train" dataloader (``self.dataloaders["train"]``).
      :type input_size: tuple or list, optional
      :param trainable_col: If ``True``, adds a column showing which parameters are trainable.
                            Defaults to ``False``.
      :type trainable_col: bool, optional
      :param \*\*kwargs: Keyword arguments to pass to ``torchinfo.summary()`` (see https://github.com/TylerYep/torchinfo).
      :type \*\*kwargs: Dict

      .. rubric:: Notes

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



   .. py:method:: freeze_layers(layers_to_freeze = None)

      Freezes the specified layers in the neural network by setting
      ``requires_grad`` attribute to False for their parameters.

      :param layers_to_freeze: List of names of the layers to freeze. If a layer name ends with
                               an asterisk (``"*"``), then all parameters whose name contains the
                               layer name (excluding the asterisk) are frozen. Otherwise,
                               only the parameters with an exact match to the layer name
                               are frozen. By default, ``[]``.
      :type layers_to_freeze: list of str, optional

      :returns: The function only modifies the ``requires_grad`` attribute of the
                specified parameters and does not return anything.
      :rtype: None

      .. rubric:: Notes

      Wildcards are accepted in the ``layers_to_freeze`` parameter.



   .. py:method:: unfreeze_layers(layers_to_unfreeze = None)

      Unfreezes the specified layers in the neural network by setting
      ``requires_grad`` attribute to True for their parameters.

      :param layers_to_unfreeze: List of names of the layers to unfreeze. If a layer name ends with
                                 an asterisk (``"*"``), then all parameters whose name contains the
                                 layer name (excluding the asterisk) are unfrozen. Otherwise,
                                 only the parameters with an exact match to the layer name
                                 are unfrozen. By default, ``[]``.
      :type layers_to_unfreeze: list of str, optional

      :returns: The function only modifies the ``requires_grad`` attribute of the
                specified parameters and does not return anything.
      :rtype: None

      .. rubric:: Notes

      Wildcards are accepted in the ``layers_to_unfreeze`` parameter.



   .. py:method:: only_keep_layers(only_keep_layers_list = None)

      Only keep the specified layers (``only_keep_layers_list``) for
      gradient computation during the backpropagation.

      :param only_keep_layers_list: List of layer names to keep. All other layers will have their
                                    gradient computation turned off. Default is ``[]``.
      :type only_keep_layers_list: list, optional

      :returns: The function only modifies the ``requires_grad`` attribute of the
                specified parameters and does not return anything.
      :rtype: None



   .. py:method:: inference(set_name = 'infer', verbose = False, print_info_batch_freq = 5)

      Run inference on a specified dataset (``set_name``).

      :param set_name: The name of the dataset to run inference on, by default
                       ``"infer"``.
      :type set_name: str, optional
      :param verbose: Whether to print verbose outputs, by default False.
      :type verbose: bool, optional
      :param print_info_batch_freq: The frequency of printouts, by default ``5``.
      :type print_info_batch_freq: int, optional

      :rtype: None

      .. rubric:: Notes

      This method calls the
      :meth:`mapreader.train.classifier.classifier.train` method with the
      ``num_epochs`` set to ``1`` and all the other parameters specified in
      the function arguments.



   .. py:method:: train_component_summary()

      Print a summary of the optimizer, criterion, and trainable model
      components.

      Returns:
      --------
      None



   .. py:method:: train(phases = None, num_epochs = 25, save_model_dir = 'models', verbose = False, tensorboard_path = None, tmp_file_save_freq = 2, remove_after_load = True, print_info_batch_freq = 5)

      Train the model on the specified phases for a given number of epochs.

      Wrapper function for
      :meth:`mapreader.train.classifier.classifier.train_core` method to
      capture exceptions (``KeyboardInterrupt`` is the only supported
      exception currently).

      :param phases: The phases to run through during each training iteration. Default is
                     ``["train", "val"]``.
      :type phases: list of str, optional
      :param num_epochs: The number of epochs to train the model for. Default is ``25``.
      :type num_epochs: int, optional
      :param save_model_dir: The directory to save the model in. Default is ``"models"``. If
                             set to ``None``, the model is not saved.
      :type save_model_dir: str or None, optional
      :param verbose: Whether to print verbose outputs, by default ``False``.
      :type verbose: int, optional
      :param tensorboard_path: The path to the directory to save TensorBoard logs in. If set to
                               ``None``, no TensorBoard logs are saved. Default is ``None``.
      :type tensorboard_path: str or None, optional
      :param tmp_file_save_freq: The frequency (in epochs) to save a temporary file of the model.
                                 Default is ``2``. If set to ``0`` or ``None``, no temporary file
                                 is saved.
      :type tmp_file_save_freq: int, optional
      :param remove_after_load: Whether to remove the temporary file after loading it. Default is
                                ``True``.
      :type remove_after_load: bool, optional
      :param print_info_batch_freq: The frequency (in batches) to print training information. Default
                                    is ``5``. If set to ``0`` or ``None``, no training information is
                                    printed.
      :type print_info_batch_freq: int, optional

      :returns: The function saves the model to the ``save_model_dir`` directory,
                and optionally to a temporary file. If interrupted with a
                ``KeyboardInterrupt``, the function tries to load the temporary
                file. If no temporary file is found, it continues without loading.
      :rtype: None

      .. rubric:: Notes

      Refer to the documentation of
      :meth:`mapreader.train.classifier.classifier.train_core` for more
      information.



   .. py:method:: train_core(phases = None, num_epochs = 25, save_model_dir = 'models', verbose = False, tensorboard_path = None, tmp_file_save_freq = 2, print_info_batch_freq = 5)

      Trains/fine-tunes a classifier for the specified number of epochs on
      the given phases using the specified hyperparameters.

      :param phases: The phases to run through during each training iteration. Default is
                     ``["train", "val"]``.
      :type phases: list of str, optional
      :param num_epochs: The number of epochs to train the model for. Default is ``25``.
      :type num_epochs: int, optional
      :param save_model_dir: The directory to save the model in. Default is ``"models"``. If
                             set to ``None``, the model is not saved.
      :type save_model_dir: str or None, optional
      :param verbose: Whether to print verbose outputs, by default ``False``.
      :type verbose: bool, optional
      :param tensorboard_path: The path to the directory to save TensorBoard logs in. If set to
                               ``None``, no TensorBoard logs are saved. Default is ``None``.
      :type tensorboard_path: str or None, optional
      :param tmp_file_save_freq: The frequency (in epochs) to save a temporary file of the model.
                                 Default is ``2``. If set to ``0`` or ``None``, no temporary file
                                 is saved.
      :type tmp_file_save_freq: int, optional
      :param print_info_batch_freq: The frequency (in batches) to print training information. Default
                                    is ``5``. If set to ``0`` or ``None``, no training information is
                                    printed.
      :type print_info_batch_freq: int, optional

      :raises ValueError: If the criterion is not set. Use the ``add_criterion`` method to
          set the criterion.

          If the optimizer is not set and the phase is "train". Use the
          ``initialize_optimizer`` or ``add_optimizer`` method to set the
          optimizer.
      :raises KeyError: If the specified phase cannot be found in the keys of the object's
          ``dataloaders`` dictionary property.

      :rtype: None



   .. py:method:: calculate_add_metrics(y_true, y_pred, y_score, phase, epoch = -1, tboard_writer=None)

      Calculate and add metrics to the classifier's metrics dictionary.

      :param y_true: True binary labels or multiclass labels. Can be considered ground
                     truth or (correct) target values.
      :type y_true: array-like of shape (n_samples,)
      :param y_pred: Predicted binary labels or multiclass labels. The estimated
                     targets as returned by a classifier.
      :type y_pred: array-like of shape (n_samples,)
      :param y_score: Predicted probabilities for each class. Only required when
                      ``y_pred`` is not binary.
      :type y_score: array-like of shape (n_samples, n_classes)
      :param phase: Name of the current phase, typically ``"train"`` or ``"val"``. See
                    ``train`` function.
      :type phase: str
      :param epoch: Current epoch number. Default is ``-1``.
      :type epoch: int, optional
      :param tboard_writer: TensorBoard SummaryWriter object to write the metrics. Default is
                            ``None``.
      :type tboard_writer: object, optional

      :rtype: None

      .. rubric:: Notes

      This method uses both the
      ``sklearn.metrics.precision_recall_fscore_support`` and
      ``sklearn.metrics.roc_auc_score`` functions from ``scikit-learn`` to
      calculate the metrics for each average type (``"micro"``, ``"macro"``
      and ``"weighted"``). The results are then added to the ``metrics``
      dictionary. It also writes the metrics to the TensorBoard
      SummaryWriter, if ``tboard_writer`` is not None.



   .. py:method:: plot_metric(y_axis, y_label, legends, x_axis = 'epoch', x_label = 'epoch', colors = 5 * ['k', 'tab:red'], styles = 10 * ['-'], markers = 10 * ['o'], figsize = (10, 5), plt_yrange = None, plt_xrange = None)

      Plot the metrics of the classifier object.

      :param y_axis: A list of metric names to be plotted on the y-axis.
      :type y_axis: list of str
      :param y_label: The label for the y-axis.
      :type y_label: str
      :param legends: The legend labels for each metric.
      :type legends: list of str
      :param x_axis: The metric to be used as the x-axis. Can be ``"epoch"`` (default)
                     or any other metric name present in the dataset.
      :type x_axis: str, optional
      :param x_label: The label for the x-axis. Defaults to ``"epoch"``.
      :type x_label: str, optional
      :param colors: The colors to be used for the lines of each metric. It must be at
                     least the same size as ``y_axis``. Defaults to
                     ``5 * ["k", "tab:red"]``.
      :type colors: list of str, optional
      :param styles: The line styles to be used for the lines of each metric. It must
                     be at least the same size as ``y_axis``. Defaults to
                     ``10 * ["-"]``.
      :type styles: list of str, optional
      :param markers: The markers to be used for the lines of each metric. It must be at
                      least the same size as ``y_axis``. Defaults to ``10 * ["o"]``.
      :type markers: list of str, optional
      :param figsize: The size of the figure in inches. Defaults to ``(10, 5)``.
      :type figsize: tuple of int, optional
      :param plt_yrange: The range of values for the y-axis. Defaults to ``None``.
      :type plt_yrange: tuple of float, optional
      :param plt_xrange: The range of values for the x-axis. Defaults to ``None``.
      :type plt_xrange: tuple of float, optional

      :rtype: None

      .. rubric:: Notes

      This function requires the ``matplotlib`` package.



   .. py:method:: show_sample(set_name = 'train', batch_number = 1, print_batch_info = True, figsize = (15, 10))

      Displays a sample of training or validation data in a grid format with
      their corresponding class labels.

      :param set_name: Name of the dataset (``"train"``/``"validation"``) to display the
                       sample from, by default ``"train"``.
      :type set_name: str, optional
      :param batch_number: Which batch to display, by default ``1``.
      :type batch_number: int, optional
      :param print_batch_info: Whether to print information about the batch size, by default
                               ``True``.
      :type print_batch_info: bool, optional
      :param figsize: Figure size (width, height) in inches, by default ``(15, 10)``.
      :type figsize: tuple, optional

      :returns: Displays the sample images with their corresponding class labels.
      :rtype: None

      :raises StopIteration: If the specified number of batches to display exceeds the total
          number of batches in the dataset.

      .. rubric:: Notes

      This method uses the dataloader of the ``ImageClassifierData`` class
      and the ``torchvision.utils.make_grid`` function to display the sample
      data in a grid format. It also calls the ``_imshow`` method of the
      ``ImageClassifierData`` class to show the sample data.



   .. py:method:: print_batch_info(set_name = 'train')

      Print information about a dataset's batches, samples, and batch-size.

      :param set_name: Name of the dataset to display batch information for (default is
                       ``"train"``).
      :type set_name: str, optional

      :rtype: None



   .. py:method:: show_inference_sample_results(label, num_samples = 6, set_name = 'test', min_conf = None, max_conf = None, figsize = (15, 15))

      Shows a sample of the results of the inference.

      :param label: The label for which to display results.
      :type label: str, optional
      :param num_samples: The number of sample results to display. Defaults to ``6``.
      :type num_samples: int, optional
      :param set_name: The name of the dataset split to use for inference. Defaults to
                       ``"test"``.
      :type set_name: str, optional
      :param min_conf: The minimum confidence score for a sample result to be displayed.
                       Samples with lower confidence scores will be skipped. Defaults to
                       ``None``.
      :type min_conf: float, optional
      :param max_conf: The maximum confidence score for a sample result to be displayed.
                       Samples with higher confidence scores will be skipped. Defaults to
                       ``None``.
      :type max_conf: float, optional
      :param figsize: Figure size (width, height) in inches, displaying the sample
                      results. Defaults to ``(15, 15)``.
      :type figsize: tuple[int, int], optional

      :rtype: None



   .. py:method:: save(save_path = 'default.obj', force = False)

      Save the object to a file.

      :param save_path: The path to the file to write.
                        If the file already exists and ``force`` is not ``True``, a ``FileExistsError`` is raised.
                        Defaults to ``"default.obj"``.
      :type save_path: str, optional
      :param force: Whether to overwrite the file if it already exists. Defaults to
                    ``False``.
      :type force: bool, optional

      :raises FileExistsError: If the file already exists and ``force`` is not ``True``.

      .. rubric:: Notes

      The object is saved in two parts. First, a serialized copy of the
      object's dictionary is written to the specified file using the
      ``joblib.dump`` function. The object's ``model`` attribute is excluded
      from this dictionary and saved separately using the ``torch.save``
      function, with a filename derived from the original ``save_path``.



   .. py:method:: save_predictions(set_name, save_path = None, delimiter = ',')


   .. py:method:: load_dataset(dataset, set_name, batch_size = 16, sampler = None, shuffle = False, num_workers = 0, **kwargs)

      Creates a DataLoader from a PatchDataset and adds it to the ``dataloaders`` dictionary.

      :param dataset: The dataset to add
      :type dataset: PatchDataset
      :param set_name: The name to use for the dataset
      :type set_name: str
      :param batch_size: The batch size to use when creating the DataLoader, by default 16
      :type batch_size: Optional[int], optional
      :param sampler: The sampler to use when creating the DataLoader, by default None
      :type sampler: Optional[Union[Sampler, None]], optional
      :param shuffle: Whether to shuffle the PatchDataset, by default False
      :type shuffle: Optional[bool], optional
      :param num_workers: The number of worker threads to use for loading data, by default 0.
      :type num_workers: Optional[int], optional



   .. py:method:: load(load_path, force_device = False)

      This function loads the state of a class instance from a saved file
      using the joblib library. It also loads a PyTorch model from a
      separate file and maps it to the device used to load the class
      instance.

      :param load_path: Path to the saved file to load.
      :type load_path: str
      :param force_device: Whether to force the use of a specific device, or the name of the
                           device to use. If set to ``True``, the default device is used.
                           Defaults to ``False``.
      :type force_device: bool or str, optional

      :raises FileNotFoundError: If the specified file does not exist.

      :rtype: None



   .. py:method:: cprint(type_info, bc_color, text)

      Print colored text with additional information.

      :param type_info: The type of message to display.
      :type type_info: str
      :param bc_color: The color to use for the message text.
      :type bc_color: str
      :param text: The text to display.
      :type text: str

      :returns: The colored message is displayed on the standard output stream.
      :rtype: None



   .. py:method:: update_progress(progress, text = '', barLength = 30)

      Update the progress bar.

      :param progress: The progress value to display, between ``0`` and ``1``.
                       If an integer is provided, it will be converted to a float.
                       If a value outside the range ``[0, 1]`` is provided, it will be
                       clamped to the nearest valid value.
      :type progress: float or int
      :param text: Additional text to display after the progress bar, defaults to
                   ``""``.
      :type text: str, optional
      :param barLength: The length of the progress bar in characters, defaults to ``30``.
      :type barLength: int, optional

      :raises TypeError: If progress is not a floating point value or an integer.

      :returns: The progress bar is displayed on the standard output stream.
      :rtype: None
