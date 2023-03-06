:py:mod:`mapreader.train.classifier`
====================================

.. py:module:: mapreader.train.classifier


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.train.classifier.classifier




.. py:class:: classifier(device='default')

   .. py:method:: set_classnames(classname_dict)

      Set names of the classes in the dataset

      Parameters
      ----------
      classname_dict : dictionary
          name of the classes in the dataset,
          e.g., {0: "rail space", 1: "No rail space"}


   .. py:method:: add2dataloader(dataset, set_name=None, batch_size=16, shuffle=True, num_workers=0, **kwds)

      Create and add a dataloader

      Parameters
      ----------
      dataset : pytorch dataset
      set_name : name of the dataset, e.g., train/val/test, optional
      batch_size : int, optional
      shuffle : bool, optional
      num_workers : int, optional


   .. py:method:: print_classes_dl(set_name: str = 'train')

      Print classes and classnames (if available)

      Parameters
      ----------
      set_name : str, optional
          Name of the dataset (normally specified in self.add2dataloader), by default "train"


   .. py:method:: add_model(model, input_size=224, is_inception=False)

      Add a model to classifier object

      Parameters
      ----------
      model : PyTorch model
          See: from torchvision import models
      input_size : int, optional
          input size, by default 224
      is_inception : bool, optional
          is this a inception-type model?, by default False


   .. py:method:: del_model()

      Delete the model


   .. py:method:: layerwise_lr(min_lr: float, max_lr: float, ltype: str = 'linspace')

      Define layer-wise learning rates

      linspace: use evenly spaced learning rates over a specified interval
      geomspace: use learning rates spaced evenly on a log scale (a geometric progression)

      Parameters
      ----------
      min_lr : float
          minimum learning rate
      max_lr : float
          maximum learning rate
      ltype : str, optional
          how to space the specified interval, by default "linspace"


   .. py:method:: initialize_optimizer(optim_type: str = 'adam', params2optim='infer', optim_param_dict: dict = {'lr': 0.001}, add_optim: bool = True)

      Initialize an optimizer
      if add_optim is True, the optimizer will be added to object

      Note that the first argument of an optimizer is:
          parameters to optimize, e.g.,
              model_ft.parameters(): all parameters are being optimized
              model_ft.fc.parameters(): only parameters of final layer are being optimized
              params2optimize = model_ft.parameters()
          Here, we use filter(lambda p: p.requires_grad, self.model.parameters())

      Parameters
      ----------
      optim_type : str, optional
          optimizer type, e.g., adam, sgd, by default "adam"
      optim_param_dict : dict, optional
          optimizer parameters, by default {"lr": 1e-3}
      add_optim : bool, optional
          add optimizer to the object, by default True


   .. py:method:: add_optimizer(optimizer)

      Add an optimizer to the object


   .. py:method:: initialize_scheduler(scheduler_type: str = 'steplr', scheduler_param_dict: dict = {'step_size': 10, 'gamma': 0.1}, add_scheduler: bool = True)

      Initialize a scheduler

      Parameters
      ----------
      scheduler_type : str, optional
          scheduler type, by default "steplr"
      scheduler_param_dict : dict, optional
          scheduler parameters, by default {"step_size": 10, "gamma": 0.1}
      add_scheduler : bool, optional
          add scheduler to the object, by default True


   .. py:method:: add_scheduler(scheduler)

      Add a scheduler to the object


   .. py:method:: add_criterion(criterion)

      Add a criterion to the object


   .. py:method:: model_summary(only_trainable=False, print_space=[40, 20, 20])

      Print model summary

      Credit: this function is the modified version of https://stackoverflow.com/a/62508086

      Other ways to check params:
          sum(p.numel() for p in myclassifier.model.parameters())
          sum(p.numel() for p in myclassifier.model.parameters() if p.requires_grad)

          # Also:
          for name, param in self.model.named_parameters():
              n = name.split(".")[0].split("_")[0]
              print(name, param.requires_grad)

      Parameters
      ----------
      only_trainable : bool, optional
          print only trainable params, by default False
      print_space : list, optional
          print params, how many spaces should be added in each column, by default [40, 20, 20]


   .. py:method:: freeze_layers(layers_to_freeze: list = [])

      Freeze a list of layers, wildcard is accepted

      Parameters
      ----------
      layers_to_freeze : list, optional
          List of layers to freeze, by default []


   .. py:method:: unfreeze_layers(layers_to_unfreeze: list = [])

      Unfreeze a list of layers, wildcard is accepted

      Parameters
      ----------
      layers_to_unfreeze : list, optional
          List of layers to unfreeze, by default []


   .. py:method:: only_keep_layers(only_keep_layers_list: list = [])

      Only keep this list of layers in training

      Parameters
      ----------
      only_keep_layers_list : list, optional
          List of layers to keep, by default []


   .. py:method:: inference(set_name='infer', verbosity_level=0, print_info_batch_freq: int = 5)

      Model inference on dataset: set_name


   .. py:method:: train_component_summary()

      Print some info about optimizer/criterion/model...


   .. py:method:: train(phases: list = ['train', 'val'], num_epochs: int = 25, save_model_dir: Union[None, str] = 'models', verbosity_level: int = 1, tensorboard_path: Union[None, str] = None, tmp_file_save_freq: int = 2, remove_after_load: bool = True, print_info_batch_freq: int = 5)

      Wrapper function for train_core method to capture exceptions. Supported exceptions so far:
      - KeyboardInterrupt

      Refer to train_core for more information.


   .. py:method:: train_core(phases: list = ['train', 'val'], num_epochs: int = 25, save_model_dir: Union[None, str] = 'models', verbosity_level: int = 1, tensorboard_path: Union[None, str] = None, tmp_file_save_freq: int = 2, print_info_batch_freq: int = 5)

      Train/fine-tune a classifier

      Parameters
      ----------
      phases : list, optional
          at each epoch, perform this list of phases, e.g., train and val, by default ["train", "val"]
      num_epochs : int, optional
          number of epochs, by default 25
      save_model_dir : Union[None, str], optional
          Parent directory to save models, by default "models"
      verbosity_level : int, optional
          verbosity level: -1 (quiet), 0 (normal), 1 (verbose), 2 (very verbose), 3 (debug)
      tensorboard_path : Union[None, str], optional
          Parent directory to save tensorboard files, by default None
      tmp_file_save_freq : int, optional
          frequency (in epoch) to save a temporary checkpoint


   .. py:method:: calculate_add_metrics(y_true, y_pred, y_score, phase, epoch=-1, tboard_writer=None)

      Calculate various evaluation metrics (e.g., precision, recall and F1) and add to self.metrics

      Parameters
      ----------
      y_true : list
          Ground truth (correct) target values
      y_pred : list
          Estimated targets as returned by a classifier.
      y_score : list
          Target scores
      phase : str
          Specified phase in training (see train function)
      epoch : int
          Epoch
      tboard_writer : optional
          tensorboard writer initialized by SummaryWriter, by default None


   .. py:method:: gen_epoch_msg(phase, epoch_msg)


   .. py:method:: plot_metric(y_axis, y_label, legends, x_axis='epoch', x_label='epoch', colors=5 * ['k', 'tab:red'], styles=10 * ['-'], markers=10 * ['o'], figsize=(10, 5), plt_yrange=None, plt_xrange=None)

      Plot content of self.metrics

      Parameters
      ----------
      y_axis : list
          items to be plotted on y-axis
      y_label : list
      legends : list
      x_axis : str, optional
          item to be plotted on x-axis, by default "epoch"
      x_label : str, optional
      colors : list, optional
          list of colors, at least the same size as y_axis, by default 5*["k", "tab:red"]
      styles : list, optional
          list of line styles, at least the same size as y_axis, by default 10*["-"]
      markers : list, optional
          list of line markers, at least the same size as y_axis, by default 10*["o"]
      figsize : tuple, optional
      plt_yrange : list, optional
      plt_xrange : list, optional


   .. py:method:: initialize_model(model_name, pretrained=True, last_layer_num_classes='default', add_model=True)

      Initialize a PyTorch model
      This method changes the number of classes in the last layer (see last_layer_num_classes)

      NOTES
      -----
      inception_v3 requires the input size to be (299,299), whereas all of the other models expect (224,224).

      models:see https://pytorch.org/vision/0.8/models.html)

      Parameters
      ----------
      model_name : str
          Name of a PyTorch model, see https://pytorch.org/vision/0.8/models.html
      pretrained : bool, optional
          Use pretrained version, by default True
      last_layer_num_classes : str, optional
          Number of elements in the last layer, by default "default"


   .. py:method:: show_sample(set_name='train', batch_number=1, print_batch_info=True, figsize=(15, 10))

      Show samples from specified dataset

      Parameters
      ----------
      set_name : str, optional
          name of the dataset, by default "train"
      batch_number : int, optional
          batch number to be plotted, by default 1
      figsize : tuple, optional
          size of the figure, by default (15, 10)


   .. py:method:: batch_info(set_name='train')

      Print info about samples/batch-size/...

      Parameters
      ----------
      set_name : str, optional
          name of the dataset, by default "train"


   .. py:method:: inference_sample_results(num_samples: int = 6, class_index: int = 0, set_name: str = 'train', min_conf: Union[None, float] = None, max_conf: Union[None, float] = None, figsize: tuple = (15, 15))

      Plot some samples (specified by num_samples) for inference outputs

      Parameters
      ----------
      num_samples : int, optional
      class_index : int, optional
          class index to be plotted, by default 0
      set_name : str, optional
          name of the dataset, by default "train"
      min_conf : Union[None, float], optional
          min prediction confidence, by default None
      max_conf : Union[None, float], optional
          max prediction confidence, by default None
      figsize : tuple, optional


   .. py:method:: save(save_path='default.obj', force=False)

      Save object


   .. py:method:: load(load_path, remove_after_load=False, force_device=False)

      load class


   .. py:method:: get_time()


   .. py:method:: cprint(type_info, bc_color, text)

      simple print function used for colored logging


   .. py:method:: update_progress(progress, text='', barLength=30)



