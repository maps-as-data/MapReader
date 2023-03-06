:py:mod:`mapreader.train.classifier_context`
============================================

.. py:module:: mapreader.train.classifier_context


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   mapreader.train.classifier_context.classifierContext




.. py:class:: classifierContext(device='default')

   Bases: :py:obj:`mapreader.train.classifier.classifier`

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


   .. py:method:: layerwise_lr(min_lr: float, max_lr: float, ltype: str = 'linspace', sep_group_names=['features1', 'features2'])

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



