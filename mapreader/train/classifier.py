#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import socket
import sys
import time

from datetime import datetime
from typing import Union, List, Optional, Tuple, Dict, Any, Hashable

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

import torch
from torch import optim
import torch.nn as nn

from torch.utils.data import DataLoader

import torchvision
from torchvision import models

# import pickle
# from tqdm.autonotebook import tqdm
# from torch.nn.modules.module import _addindent


class classifier:
    def __init__(self, device: Optional[str] = "default"):
        """
        Initialize a classifier object.

        Parameters
        ----------
        device : str, optional
            The device to be used for training and storing models. Can be set
            to ``"default"``, ``"cpu"``, ``"cuda:0"``, etc. By default
            ``"default"``.

        Attributes
        ----------
        device : torch.device
            The device being used for training and storing models.
        dataloader : dict
            A dictionary to store dataloaders for the model.
        dataset_sizes : dict
            A dictionary to store sizes of datasets for the model.
        class_names : None or list of str
            A list of class names for the model.
        model : None or torch.nn.Module
            The model being trained.
        optimizer : None or torch.optim.Optimizer
            The optimizer being used for training the model.
        scheduler : None or torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler being used for training the model.
        input_size : None or tuple of int
            The size of the input to the model.
        is_inception : None or bool
            A flag indicating if the model is an Inception model.
        metrics : dict
            A dictionary to store the metrics computed during training.
        last_epoch : int
            The last epoch number completed during training.
        best_loss : torch.Tensor
            The best validation loss achieved during training.
        best_epoch : int
            The epoch in which the best validation loss was achieved during
            training.
        tmp_save_filename : str
            A temporary file name to save checkpoints during training and
            validation.
        """

        if device in ["default", None]:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
        print(f"[INFO] Device is set to {self.device}")

        self.dataloader = {}
        self.dataset_sizes = {}
        self.class_names = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.input_size = None
        self.is_inception = None
        self.metrics = {}
        self.last_epoch = 0
        self.best_loss = torch.tensor(np.inf)
        self.best_epoch = 0
        # temp file to save checkpoints during training/validation
        self.tmp_save_filename = (
            f"tmp_{random.randint(0, 1e10)}_checkpoint.pkl"
        )

        # add colors for printing/logging
        self._print_colors()

    def set_classnames(self, classname_dict: dict) -> None:
        """
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
        """
        self.class_names = classname_dict
        self.num_classes = len(self.class_names)

    def add2dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        set_name: Optional[Union[str, None]] = None,
        batch_size: Optional[int] = 16,
        shuffle: Optional[bool] = True,
        num_workers: Optional[int] = 0,
        **kwds,
    ) -> torch.utils.data.DataLoader:
        """
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
        """

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwds,
        )

        if set_name is None:
            return dl
        else:
            self.dataloader[set_name] = dl
            self.dataset_sizes[set_name] = len(
                self.dataloader[set_name].dataset
            )
            print(
                f"[INFO] added '{set_name}' dataloader with {self.dataset_sizes[set_name]} elements."  # noqa
            )

    def print_classes_dl(self, set_name: Optional[str] = "train"):
        """
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
        """
        print(
            f"[INFO] labels:   {self.dataloader[set_name].dataset.uniq_labels}"
        )
        if self.class_names is not None:
            print(f"[INFO] class-names: {self.class_names}")

    def add_model(
        self,
        model: nn.Module,
        input_size: Optional[int] = 224,
        is_inception: Optional[bool] = False,
    ) -> None:
        """
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
        """
        if self.class_names is None:
            raise ValueError(
                "[ERROR] specify class names using set_classnames method."
            )
        else:
            self.print_classes_dl()

        self.model = model.to(self.device)
        self.input_size = input_size
        self.is_inception = is_inception

    def del_model(self) -> None:
        """
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
        """
        self.model = None
        self.input_size = None
        self.is_inception = None
        self.metrics = {}
        self.last_epoch = 0
        self.best_loss = torch.tensor(np.inf)
        self.best_epoch = 0

    def layerwise_lr(
        self,
        min_lr: float,
        max_lr: float,
        ltype: Optional[str] = "linspace",
    ) -> List[Dict]:
        """
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
        """
        if ltype.lower() in ["line", "linear", "linspace"]:
            list_lrs = np.linspace(
                min_lr, max_lr, len(list(self.model.named_parameters()))
            )
        elif ltype.lower() in ["log", "geomspace"]:
            list_lrs = np.geomspace(
                min_lr, max_lr, len(list(self.model.named_parameters()))
            )
        else:
            raise NotImplementedError(
                "Implemented methods are: linspace and geomspace"
            )

        list2optim = []
        for i, (name, params) in enumerate(self.model.named_parameters()):
            list2optim.append({"params": params, "lr": list_lrs[i]})

        return list2optim

    def initialize_optimizer(
        self,
        optim_type: Optional[str] = "adam",
        params2optim: Optional[str] = "infer",
        optim_param_dict: Optional[dict] = {"lr": 1e-3},
        add_optim: Optional[bool] = True,
    ) -> Union[torch.optim.Optimizer, None]:
        """
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
            - ``model_ft.fc.parameters()``: only parameters of final layer are
              being optimized

        Here, we use:

        .. code-block:: python

            filter(lambda p: p.requires_grad, self.model.parameters())
        """
        if params2optim == "infer":
            params2optim = filter(
                lambda p: p.requires_grad, self.model.parameters()
            )

        if optim_type.lower() in ["adam"]:
            optimizer = optim.Adam(params2optim, **optim_param_dict)
        elif optim_type.lower() in ["adamw"]:
            optimizer = optim.AdamW(params2optim, **optim_param_dict)
        elif optim_type.lower() in ["sgd"]:
            optimizer = optim.SGD(params2optim, **optim_param_dict)

        if add_optim:
            self.add_optimizer(optimizer)
        else:
            return optimizer

    def add_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Add an optimizer to the classifier object.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to add to the classifier object.

        Returns
        -------
        None
        """
        self.optimizer = optimizer

    def initialize_scheduler(
        self,
        scheduler_type: Optional[str] = "steplr",
        scheduler_param_dict: Optional[dict] = {"step_size": 10, "gamma": 0.1},
        add_scheduler: Optional[bool] = True,
    ) -> Union[torch.optim.lr_scheduler._LRScheduler, None]:
        """
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
        """
        if scheduler_type.lower() in ["steplr"]:
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **scheduler_param_dict
            )
        elif scheduler_type.lower() in ["onecyclelr"]:
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, **scheduler_param_dict
            )
        else:
            raise NotImplementedError(
                f"[ERROR] scheduler of type: {scheduler_type} is not implemented."  # noqa
            )

        if add_scheduler:
            self.add_scheduler(scheduler)
        else:
            return scheduler

    def add_scheduler(
        self, scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> None:
        """
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
        """
        if self.optimizer is None:
            raise ValueError(
                "[ERROR] optimizer is needed. Use initialize_optimizer or add_optimizer"  # noqa
            )

        self.scheduler = scheduler

    def add_criterion(self, criterion: torch.nn.modules.loss._Loss) -> None:
        """
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
        """
        self.criterion = criterion

    def model_summary(
        self,
        only_trainable: Optional[bool] = False,
        print_space: Optional[List[int]] = [40, 20, 20],
    ) -> None:
        """
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
        """

        if self.model is None:
            raise ValueError("[ERROR] no model is added. Use .add_model")

        # header
        col1, col2, col3 = "modules", "parameters", "dim"
        line_divider = sum(print_space) * "-" + "----------"
        print(line_divider)
        print(
            f"| {col1:>{print_space[0]}} | {col2:>{print_space[1]}} | {col3:>{print_space[2]}}"  # noqa
        )
        print(line_divider)

        # body
        total_params = 0
        total_trainable_params = 0
        for name, parameter in self.model.named_parameters():
            if (not parameter.requires_grad) and only_trainable:
                continue
            elif not parameter.requires_grad:
                cbeg, cend = self.color_red, self.color_reset
            else:
                cbeg = cend = ""

            param = parameter.numel()

            print(
                f"{cbeg}| {name:>{print_space[0]}} | {param:>{print_space[1]}} | {str(list(parameter.shape)):>{print_space[2]}} |{cend}"  # noqa
            )

            total_params += param
            if parameter.requires_grad:
                total_trainable_params += param

        # footer
        print(line_divider)
        if not only_trainable:
            print(
                f"| {'Total params':>{print_space[0]}} | {total_params:>{print_space[1]}} | {'':>{print_space[2]}} |"  # noqa
            )
        print(
            f"| {'Total trainable params':>{print_space[0]}} | {total_trainable_params:>{print_space[1]}} | {'':>{print_space[2]}} |"  # noqa
        )
        print(line_divider)

        # add 6 to sum(print_space) as we have two times: " | " with size 3 in
        # the other/above prints
        print(f"| {'Other parameters:':<{sum(print_space) + 6}} |")
        print(
            f"| {'* input size:   '+str(self.input_size):<{sum(print_space) + 6}} |"  # noqa
        )
        print(
            f"| {'* is_inception: '+str(self.is_inception):<{sum(print_space) + 6}} |"  # noqa
        )
        print(line_divider)

    def freeze_layers(
        self, layers_to_freeze: Optional[List[str]] = []
    ) -> None:
        """
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
        """

        for layer in layers_to_freeze:
            for name, param in self.model.named_parameters():
                if (layer[-1] == "*") and (layer.replace("*", "") in name):
                    param.requires_grad = False
                elif (layer[-1] != "*") and (layer == name):
                    param.requires_grad = False

    def unfreeze_layers(self, layers_to_unfreeze: Optional[List[str]] = []):
        """
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
        """

        for layer in layers_to_unfreeze:
            for name, param in self.model.named_parameters():
                if (layer[-1] == "*") and (layer.replace("*", "") in name):
                    param.requires_grad = True
                elif (layer[-1] != "*") and (layer == name):
                    param.requires_grad = True

    def only_keep_layers(
        self, only_keep_layers_list: Optional[List[str]] = []
    ) -> None:
        """
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
        """
        for name, param in self.model.named_parameters():
            if name in only_keep_layers_list:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def inference(
        self,
        set_name: Optional[str] = "infer",
        verbosity_level: Optional[int] = 0,
        print_info_batch_freq: Optional[int] = 5,
    ):
        """
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
        """
        self.train(
            phases=[set_name],
            num_epochs=1,
            save_model_dir=None,
            verbosity_level=verbosity_level,
            tensorboard_path=None,
            tmp_file_save_freq=2,
            remove_after_load=False,
            print_info_batch_freq=print_info_batch_freq,
        )

    def train_component_summary(self) -> None:
        """
        Print a summary of the optimizer, criterion and trainable model
        components.

        Returns:
        --------
        None
        """
        divider = 20 * "="
        print(divider)
        print("* Optimizer:")
        print(str(self.optimizer))
        print(divider)
        print("* Criterion:")
        print(str(self.criterion))
        print(divider)
        print("* Model:")
        self.model_summary(only_trainable=True)

    def train(
        self,
        phases: Optional[List[str]] = ["train", "val"],
        num_epochs: Optional[int] = 25,
        save_model_dir: Optional[Union[str, None]] = "models",
        verbosity_level: Optional[int] = 1,
        tensorboard_path: Optional[Union[str, None]] = None,
        tmp_file_save_freq: Optional[Union[int, None]] = 2,
        remove_after_load: Optional[bool] = True,
        print_info_batch_freq: Optional[Union[int, None]] = 5,
    ) -> None:
        """
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
        """

        try:
            self.train_core(
                phases,
                num_epochs,
                save_model_dir,
                verbosity_level,
                tensorboard_path,
                tmp_file_save_freq,
                print_info_batch_freq=print_info_batch_freq,
            )
        except KeyboardInterrupt:
            print("KeyboardInterrupted...Exiting...")
            if os.path.isfile(self.tmp_save_filename):
                print(f"File found: {self.tmp_save_filename}...load...")
                self.load(
                    self.tmp_save_filename, remove_after_load=remove_after_load
                )
            else:
                print("No temporary file was found.")

    def train_core(
        self,
        phases: Optional[List[str]] = ["train", "val"],
        num_epochs: Optional[int] = 25,
        save_model_dir: Optional[Union[str, None]] = "models",
        verbosity_level: Optional[int] = 1,
        tensorboard_path: Optional[Union[str, None]] = None,
        tmp_file_save_freq: Optional[Union[int, None]] = 2,
        print_info_batch_freq: Optional[Union[int, None]] = 5,
    ) -> None:
        """
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
        """

        if self.criterion is None:
            raise ValueError(
                "[ERROR] criterion is needed. Use add_criterion method"
            )

        for phase in phases:
            if phase not in self.dataloader.keys():
                raise KeyError(
                    f"[ERROR] specified phase: {phase} cannot be find in object's dataloader with keys: {self.dataloader.keys()}"  # noqa
                )

        if verbosity_level >= 1:
            self.train_component_summary()

        since = time.time()

        # initialize variables
        train_phase_names = ["train", "training"]
        valid_phase_names = ["val", "validation", "eval", "evaluation"]
        best_model_wts = copy.deepcopy(self.model.state_dict())
        self.pred_conf = []
        self.pred_label = []
        self.orig_label = []
        if save_model_dir is not None:
            save_model_dir = os.path.abspath(save_model_dir)

        # Check if SummaryWriter (for tensorboard) can be imported
        tboard_writer = None
        if tensorboard_path is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tboard_writer = SummaryWriter(tensorboard_path)
            except ImportError:
                print(
                    "[WARNING] could not import SummaryWriter from torch.utils.tensorboard"  # noqa
                )
                print("[WARNING] continue without tensorboard.")
                tensorboard_path = None

        start_epoch = self.last_epoch + 1
        end_epoch = self.last_epoch + num_epochs
        # --- Main train loop
        for epoch in range(start_epoch, end_epoch + 1):
            # --- loop, phases
            for phase in phases:
                if phase.lower() in train_phase_names:
                    self.model.train()
                else:
                    self.model.eval()

                # initialize vars with one epoch lifetime
                running_loss = 0.0
                running_pred_conf = []
                running_pred_label = []
                running_orig_label = []

                # TQDM
                # batch_loop = tqdm(iter(self.dataloader[phase]), total=len(self.dataloader[phase]), leave=False) # noqa
                # if phase.lower() in train_phase_names+valid_phase_names:
                #     batch_loop.set_description(f"Epoch {epoch}/{end_epoch}")

                phase_batch_size = self.dataloader[phase].batch_size
                total_inp_counts = len(self.dataloader[phase].dataset)

                # --- loop, batches
                for batch_idx, (inputs, labels) in enumerate(
                    self.dataloader[phase]
                ):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    if self.optimizer is None:
                        if phase.lower() in train_phase_names:
                            raise ValueError(
                                f"[ERROR] optimizer should be set when phase is {phase}. Use initialize_optimizer or add_optimizer."  # noqa
                            )
                    else:
                        self.optimizer.zero_grad()

                    if phase.lower() in train_phase_names + valid_phase_names:
                        # forward, track history if only in train
                        with torch.set_grad_enabled(
                            phase.lower() in train_phase_names
                        ):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training,
                            # it has an auxiliary output.
                            #     In train mode we calculate the loss by
                            #     summing the final output and the auxiliary
                            #     output but in testing we only consider the
                            #     final output.
                            if self.is_inception and (
                                phase.lower() in train_phase_names
                            ):
                                outputs, aux_outputs = self.model(inputs)
                                loss1 = self.criterion(outputs, labels)
                                loss2 = self.criterion(aux_outputs, labels)
                                # XXX From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958 # noqa
                                loss = loss1 + 0.4 * loss2
                            else:
                                outputs = self.model(inputs)
                                # labels = labels.long().squeeze_()
                                loss = self.criterion(outputs, labels)

                            _, pred_label = torch.max(outputs, dim=1)

                            # backward + optimize only if in training phase
                            if phase.lower() in train_phase_names:
                                loss.backward()
                                self.optimizer.step()

                        # XXX (why multiply?)
                        running_loss += loss.item() * inputs.size(0)

                        # TQDM
                        # batch_loop.set_postfix(loss=loss.data)
                        # batch_loop.refresh()
                    else:
                        outputs = self.model(inputs)
                        _, pred_label = torch.max(outputs, dim=1)

                    running_pred_conf.extend(
                        torch.nn.functional.softmax(outputs, dim=1)
                        .cpu()
                        .tolist()
                    )
                    running_pred_label.extend(pred_label.cpu().tolist())
                    running_orig_label.extend(labels.cpu().tolist())

                    if batch_idx % print_info_batch_freq == 0:
                        curr_inp_counts = min(
                            total_inp_counts,
                            (batch_idx + 1) * phase_batch_size,
                        )
                        progress_perc = (
                            curr_inp_counts / total_inp_counts * 100.0
                        )
                        tmp_str = f"{curr_inp_counts}/{total_inp_counts} ({progress_perc:5.1f}%)"  # noqa

                        epoch_msg = f"{phase: <8} -- {epoch}/{end_epoch} -- "
                        epoch_msg += f"{tmp_str: >20} -- "

                        if phase.lower() in valid_phase_names:
                            epoch_msg += f"Loss: {loss.data:.3f}"
                            self.cprint("[INFO]", self.color_dred, epoch_msg)
                        elif phase.lower() in train_phase_names:
                            epoch_msg += f"Loss: {loss.data:.3f}"
                            self.cprint("[INFO]", self.color_dgreen, epoch_msg)
                        else:
                            self.cprint("[INFO]", self.color_dgreen, epoch_msg)
                    # --- END: one batch

                # scheduler
                if phase.lower() in train_phase_names and (
                    self.scheduler is not None
                ):
                    self.scheduler.step()

                if phase.lower() in train_phase_names + valid_phase_names:
                    # --- collect statistics
                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    self._add_metrics(f"epoch_loss_{phase}", epoch_loss)

                    if tboard_writer is not None:
                        tboard_writer.add_scalar(
                            f"loss/{phase}",
                            self.metrics[f"epoch_loss_{phase}"][-1],
                            epoch,
                        )

                    # other metrics (precision/recall/F1)
                    self.calculate_add_metrics(
                        running_orig_label,
                        running_pred_label,
                        running_pred_conf,
                        phase,
                        epoch,
                        tboard_writer,
                    )

                    epoch_msg = f"{phase: <8} -- {epoch}/{end_epoch} -- "
                    epoch_msg = self.gen_epoch_msg(phase, epoch_msg)

                    if phase.lower() in valid_phase_names:
                        self.cprint(
                            "[INFO]", self.color_dred, epoch_msg + "\n"
                        )
                    else:
                        self.cprint("[INFO]", self.color_dgreen, epoch_msg)

                # labels/confidence
                self.pred_conf.extend(running_pred_conf)
                self.pred_label.extend(running_pred_label)
                self.orig_label.extend(running_orig_label)

                # Update best_loss and _epoch?
                if (
                    phase.lower() in valid_phase_names
                    and epoch_loss < self.best_loss
                ):
                    self.best_loss = epoch_loss
                    self.best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if phase.lower() in valid_phase_names:
                    if epoch % tmp_file_save_freq == 0:
                        print(
                            f"SAVE temp file: {self.tmp_save_filename} | set .last_epoch: {epoch}"  # noqa
                        )
                        tmp_str = f"[INFO] SAVE temp file: {self.tmp_save_filename} | set .last_epoch: {epoch}\n"  # noqa
                        print(self.color_lgrey + tmp_str + self.color_reset)
                        self.last_epoch = epoch
                        self.save(self.tmp_save_filename, force=True)

        time_elapsed = time.time() - since
        print(
            f"Total time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # --- SAVE model/object
        if phase.lower() in train_phase_names + valid_phase_names:
            self.last_epoch = epoch
            if save_model_dir is not None:
                save_filename = f"checkpoint_{self.best_epoch}.pkl"
                save_model_path = os.path.join(save_model_dir, save_filename)
                self.save(save_model_path, force=True)
                with open(
                    os.path.join(save_model_dir, "info.txt"), "a+"
                ) as fio:
                    fio.writelines(f"{save_filename},{self.best_loss:.5f}\n")

                print(
                    f"[INFO] Save model epoch: {self.best_epoch} with least valid loss: {self.best_loss:.4f}"  # noqa
                )
                print(f"[INFO] Path: {save_model_path}")

    def calculate_add_metrics(
        self,
        y_true,
        y_pred,
        y_score,
        phase: str,
        epoch: Optional[int] = -1,
        tboard_writer=None,
    ) -> None:
        """
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
        """
        # convert y_score to a numpy array:
        y_score = np.array(y_score)

        for avrg in ["micro", "macro", "weighted"]:
            prec, rcall, fscore, supp = precision_recall_fscore_support(
                y_true, y_pred, average=avrg
            )
            self._add_metrics(f"epoch_prec_{avrg}_{phase}", prec * 100.0)
            self._add_metrics(f"epoch_recall_{avrg}_{phase}", rcall * 100.0)
            self._add_metrics(f"epoch_fscore_{avrg}_{phase}", fscore * 100.0)
            self._add_metrics(f"epoch_supp_{avrg}_{phase}", supp)

            if tboard_writer is not None:
                tboard_writer.add_scalar(
                    f"Precision/{phase}/{avrg}",
                    self.metrics[f"epoch_prec_{avrg}_{phase}"][-1],
                    epoch,
                )
                tboard_writer.add_scalar(
                    f"Recall/{phase}/{avrg}",
                    self.metrics[f"epoch_recall_{avrg}_{phase}"][-1],
                    epoch,
                )
                tboard_writer.add_scalar(
                    f"Fscore/{phase}/{avrg}",
                    self.metrics[f"epoch_fscore_{avrg}_{phase}"][-1],
                    epoch,
                )

            # --- compute ROC AUC
            if y_score.shape[1] == 2:
                # ---- binary case
                # From scikit-learn:
                #     The probability estimates correspond to the probability
                #     of the class with the greater label, i.e.
                #     estimator.classes_[1] and thus
                #     estimator.predict_proba(X, y)[:, 1]
                roc_auc = roc_auc_score(y_true, y_score[:, 1], average=avrg)
            elif (y_score.shape[1] != 2) and (avrg in ["macro", "weighted"]):
                # ---- multiclass
                # In the multiclass case, it corresponds to an array of shape
                # (n_samples, n_classes)
                try:
                    roc_auc = roc_auc_score(
                        y_true, y_score, average=avrg, multi_class="ovr"
                    )
                except:
                    continue
            else:
                continue

            self._add_metrics(f"epoch_rocauc_{avrg}_{phase}", roc_auc * 100.0)

        prfs = precision_recall_fscore_support(y_true, y_pred, average=None)
        for i in range(len(prfs[0])):
            self._add_metrics(f"epoch_prec_{i}_{phase}", prfs[0][i] * 100.0)
            self._add_metrics(f"epoch_recall_{i}_{phase}", prfs[1][i] * 100.0)
            self._add_metrics(f"epoch_fscore_{i}_{phase}", prfs[2][i] * 100.0)
            self._add_metrics(f"epoch_supp_{i}_{phase}", prfs[3][i])

            if tboard_writer is not None:
                tboard_writer.add_scalar(
                    f"Precision/{phase}/binary_{i}",
                    self.metrics[f"epoch_prec_{i}_{phase}"][-1],
                    epoch,
                )
                tboard_writer.add_scalar(
                    f"Recall/{phase}/binary_{i}",
                    self.metrics[f"epoch_recall_{i}_{phase}"][-1],
                    epoch,
                )
                tboard_writer.add_scalar(
                    f"Fscore/{phase}/binary_{i}",
                    self.metrics[f"epoch_fscore_{i}_{phase}"][-1],
                    epoch,
                )

    def gen_epoch_msg(self, phase: str, epoch_msg: str) -> str:
        """
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
        """
        tmp_loss = self.metrics[f"epoch_loss_{phase}"][-1]
        epoch_msg += f"Loss: {tmp_loss:.3f}; "

        tmp_fscore = self.metrics[f"epoch_fscore_macro_{phase}"][-1]
        epoch_msg += f"F_macro: {tmp_fscore:.2f}; "

        tmp_recall = self.metrics[f"epoch_recall_macro_{phase}"][-1]
        epoch_msg += f"R_macro: {tmp_recall:.2f}"

        return epoch_msg

    def _add_metrics(
        self, k: Hashable, v: Union[int, float, complex, np.number]
    ) -> None:
        """
        Adds a metric value to a dictionary of metrics tracked during training.

        Parameters
        ----------
        k : hashable
            The key for the metric being tracked.
        v : numeric
            The metric value to add to the corresponding list of metric values.

        Returns
        -------
        None

        Notes
        -----
        If the key ``k`` does not exist in the dictionary of metrics, a new
        key-value pair is created with ``k`` as the key and a new list
        containing the value ``v`` as the value. If the key ``k`` already
        exists in the dictionary of metrics, the value `v` is appended to the
        list associated with the key ``k``.
        """
        if k not in self.metrics.keys():
            self.metrics[k] = [v]
        else:
            self.metrics[k].append(v)

    def plot_metric(
        self,
        y_axis: List[str],
        y_label: str,
        legends: List[str],
        x_axis: Optional[str] = "epoch",
        x_label: Optional[str] = "epoch",
        colors: Optional[List[str]] = 5 * ["k", "tab:red"],
        styles: Optional[List[str]] = 10 * ["-"],
        markers: Optional[List[str]] = 10 * ["o"],
        figsize: Optional[Tuple[int, int]] = (10, 5),
        plt_yrange: Optional[Tuple[float, float]] = None,
        plt_xrange: Optional[Tuple[float, float]] = None,
    ):
        """
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
        """

        # Font sizes
        plt_size = {
            "xlabel": 24,
            "ylabel": 24,
            "xtick": 18,
            "ytick": 18,
            "legend": 18,
        }

        fig = plt.figure(figsize=figsize)
        if x_axis == "epoch":
            from matplotlib.ticker import MaxNLocator

            # make x ticks integer
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        for i, one_item in enumerate(y_axis):
            if one_item not in self.metrics.keys():
                print(
                    f"[WARNING] requested item: {one_item} not in keys: {self.metrics.keys}"  # noqa
                )
                continue

            if x_axis == "epoch":
                x_axis_plt = range(1, len(self.metrics[one_item]) + 1)
            else:
                x_axis_plt = self.metrics[x_axis]

            plt.plot(
                x_axis_plt,
                self.metrics[one_item],
                label=legends[i],
                color=colors[i],
                ls=styles[i],
                marker=markers[i],
                lw=3,
            )

        # --- labels and ticks
        plt.xlabel(x_label, size=plt_size["xlabel"])
        plt.ylabel(y_label, size=plt_size["ylabel"])
        plt.xticks(size=plt_size["xtick"])
        plt.yticks(size=plt_size["ytick"])

        # --- legend
        plt.legend(
            fontsize=plt_size["legend"],
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            ncol=2,
            borderaxespad=0,
            loc="lower center",
        )

        # --- x/y range
        if plt_xrange is not None:
            plt.xlim(plt_xrange[0], plt_xrange[1])
        if plt_yrange is not None:
            plt.ylim(plt_yrange[0], plt_yrange[1])

        plt.grid()
        plt.show()

    def initialize_model(
        self,
        model_name: str,
        pretrained: Optional[bool] = True,
        last_layer_num_classes: Optional[Union[str, int]] = "default",
        add_model: Optional[bool] = True,
    ) -> Tuple[Any, int, bool]:
        """
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
        """

        # Initialize these variables which will be set in this if statement.
        # Each of these variables is model specific.
        model_dw = models.__getattribute__(model_name)
        model_dw = model_dw(pretrained)
        input_size = 224
        is_inception = False

        if last_layer_num_classes in ["default"]:
            last_layer_num_classes = self.num_classes
        else:
            last_layer_num_classes = int(last_layer_num_classes)

        if "resnet" in model_name:
            num_ftrs = model_dw.fc.in_features
            model_dw.fc = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "alexnet" in model_name:
            num_ftrs = model_dw.classifier[6].in_features
            model_dw.classifier[6] = nn.Linear(
                num_ftrs, last_layer_num_classes
            )

        elif "vgg" in model_name:
            # vgg11_bn
            num_ftrs = model_dw.classifier[6].in_features
            model_dw.classifier[6] = nn.Linear(
                num_ftrs, last_layer_num_classes
            )

        elif "squeezenet" in model_name:
            model_dw.classifier[1] = nn.Conv2d(
                512, last_layer_num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
            model_dw.num_classes = last_layer_num_classes

        elif "densenet" in model_name:
            num_ftrs = model_dw.classifier.in_features
            model_dw.classifier = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "inception" in model_name:
            # Inception v3:
            # Be careful, expects (299,299) sized images + has auxiliary output

            # Handle the auxilary net
            num_ftrs = model_dw.AuxLogits.fc.in_features
            model_dw.AuxLogits.fc = nn.Linear(num_ftrs, last_layer_num_classes)
            # Handle the primary net
            num_ftrs = model_dw.fc.in_features
            model_dw.fc = nn.Linear(num_ftrs, last_layer_num_classes)
            is_inception = True
            input_size = 299

        else:
            raise ValueError("Invalid model name, exiting...")

        if add_model:
            self.del_model()
            self.add_model(
                model_dw, input_size=input_size, is_inception=is_inception
            )
        else:
            return model_dw, input_size, is_inception

    def show_sample(
        self,
        set_name: Optional[str] = "train",
        batch_number: Optional[int] = 1,
        print_batch_info: Optional[bool] = True,
        figsize: Optional[Tuple[int, int]] = (15, 10),
    ):
        """
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
        """
        if print_batch_info:
            # print info about batch size
            self.batch_info()

        dl_iter = iter(self.dataloader[set_name])
        for _ in range(batch_number):
            # Get a batch of training data
            inputs, classes = next(dl_iter)

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        self._imshow(
            out,
            title=str([self.class_names[int(x)] for x in classes]),
            figsize=figsize,
        )

    def batch_info(self, set_name: Optional[str] = "train") -> None:
        """
        Print information about a dataset's batches, samples, and batch-size.

        Parameters
        ----------
        set_name : str, optional
            Name of the dataset to display batch information for (default is
            ``"train"``).

        Returns
        -------
        None
        """

        batch_size = self.dataloader[set_name].batch_size
        num_samples = len(self.dataloader[set_name].dataset)
        num_batches = int(np.ceil(num_samples / batch_size))

        print(f"[INFO] dataset: {set_name}")
        print(f"#samples:    {num_samples}")
        print(f"#batch size: {batch_size}")
        print(f"#batches:    {num_batches}")

    @staticmethod
    def _imshow(
        inp: np.ndarray,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = (15, 10),
    ) -> None:
        """
        Displays an image of a tensor using matplotlib.pyplot.

        Parameters
        ----------
        inp : numpy.ndarray
            Input image to be displayed.
        title : str, optional
            Title of the plot, default is ``None``.
        figsize : tuple, optional
            Figure size in inches as a tuple of (width, height), default is
            ``(15, 10)``.

        Returns
        -------
        None
            Displays the image of the provided tensor.
        """

        inp = inp.numpy().transpose((1, 2, 0))
        # XXX
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean

        inp = np.clip(inp, 0, 1)
        plt.figure(figsize=figsize)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()

    def inference_sample_results(
        self,
        num_samples: Optional[int] = 6,
        class_index: Optional[int] = 0,
        set_name: Optional[str] = "train",
        min_conf: Optional[Union[None, float]] = None,
        max_conf: Optional[Union[None, float]] = None,
        figsize: Optional[Tuple[int, int]] = (15, 15),
    ) -> None:
        """
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
        """

        # eval mode, keep track of the current mode
        was_training = self.model.training
        self.model.eval()

        counter = 0
        fig = plt.figure(figsize=figsize)
        with torch.no_grad():
            for inputs, labels in iter(self.dataloader[set_name]):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                pred_conf = torch.nn.functional.softmax(outputs, dim=1) * 100.0
                _, preds = torch.max(outputs, 1)

                # go through images in batch
                for j in range(len(preds)):
                    pred_ind = int(preds[j])
                    if pred_ind != class_index:
                        continue
                    if (min_conf is not None) and (
                        pred_conf[j][pred_ind] < min_conf
                    ):
                        continue
                    if (max_conf is not None) and (
                        pred_conf[j][pred_ind] > max_conf
                    ):
                        continue

                    counter += 1

                    class_name = self.class_names[pred_ind]
                    conf_score = pred_conf[j][pred_ind]
                    ax = plt.subplot(int(num_samples / 2.0), 3, counter)
                    ax.axis("off")
                    ax.set_title(f"{class_name} | {conf_score:.3f}")

                    inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    inp = np.clip(inp, 0, 1)
                    plt.imshow(inp)

                    if counter == num_samples:
                        self.model.train(mode=was_training)
                        plt.show()
                        return
            self.model.train(mode=was_training)
            plt.show()

    def save(
        self,
        save_path: Optional[str] = "default.obj",
        force: Optional[bool] = False,
    ) -> None:
        """
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
        """
        if os.path.isfile(save_path):
            if force:
                os.remove(save_path)
            else:
                raise FileExistsError(f"file already exists: {save_path}")

        # parent/base-names
        par_name = os.path.dirname(os.path.abspath(save_path))
        base_name = os.path.basename(os.path.abspath(save_path))

        # Extract model, write it separately using torch.save
        obj2write = copy.deepcopy(self.__dict__)
        mymodel = obj2write["model"]
        del obj2write["model"]

        os.makedirs(par_name, exist_ok=True)
        with open(save_path, "wb") as myfile:
            # pickle.dump(self.__dict__, myfile)
            joblib.dump(obj2write, myfile)

        torch.save(mymodel, os.path.join(par_name, f"model_{base_name}"))

    def load(
        self,
        load_path: str,
        remove_after_load: Optional[bool] = False,
        force_device: Optional[bool] = False,
    ) -> None:
        """
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
        """

        load_path = os.path.abspath(load_path)
        mydevice = self.device

        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"file not found: {load_path}")

        with open(load_path, "rb") as myfile:
            # objPickle = pickle.load(myfile)
            objPickle = joblib.load(myfile)

        if remove_after_load:
            os.remove(load_path)

        self.__dict__ = objPickle

        if force_device:
            if not isinstance(force_device, str):
                force_device = str(force_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = force_device

        par_name = os.path.dirname(load_path)
        base_name = os.path.basename(load_path)
        path2model = os.path.join(par_name, f"model_{base_name}")
        self.model = torch.load(path2model, map_location=mydevice)

        try:
            self.device = mydevice
            self.model = self.model.to(mydevice)
        except:
            pass

    def _print_colors(self):
        """Private function, setting color attributes on the object."""
        # color
        self.color_lgrey = "\033[1;90m"
        self.color_grey = "\033[90m"  # broing information
        self.color_yellow = "\033[93m"  # FYI
        self.color_orange = "\033[0;33m"  # Warning

        self.color_lred = "\033[1;31m"  # there is smoke
        self.color_red = "\033[91m"  # fire!
        self.color_dred = "\033[2;31m"  # Everything is on fire

        self.color_lblue = "\033[1;34m"
        self.color_blue = "\033[94m"
        self.color_dblue = "\033[2;34m"

        self.color_lgreen = "\033[1;32m"  # all is normal
        self.color_green = "\033[92m"  # something else
        self.color_dgreen = "\033[2;32m"  # even more interesting

        self.color_lmagenta = "\033[1;35m"
        self.color_magenta = "\033[95m"  # for title
        self.color_dmagenta = "\033[2;35m"

        self.color_cyan = "\033[96m"  # system time
        self.color_white = "\033[97m"  # final time

        self.color_black = "\033[0;30m"

        self.color_reset = "\033[0m"
        self.color_bold = "\033[1m"
        self.color_under = "\033[4m"

    def get_time(self) -> str:
        """
        Get the current date and time as a formatted string.

        Returns
        -------
        str
            A string representing the current date and time.
        """
        time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        return time

    def cprint(self, type_info: str, bc_color: str, text: str) -> None:
        """
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
        """
        ho_nam = socket.gethostname().split(".")[0][:10]

        print(
            self.color_green + self.get_time() + self.color_reset,
            self.color_magenta + ho_nam + self.color_reset,
            self.color_bold + self.color_grey + type_info + self.color_reset,
            bc_color + text + self.color_reset,
        )

    def update_progress(
        self,
        progress: Union[float, int],
        text: Optional[str] = "",
        barLength: Optional[int] = 30,
    ) -> None:
        """Update the progress bar.

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
        """

        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress provided must be float or integer\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = f"\r[{'#'*block + '-'*(barLength-block)}] {progress*100:.1f}% {status} {text}"  # noqa
        sys.stdout.write(text)
        sys.stdout.flush()
