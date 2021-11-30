#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import socket
import sys
import time
# from tqdm.autonotebook import tqdm
from typing import Union

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import torchvision
from torchvision import models

class classifier:
    def __init__(self, device='default'):
        """Instantiate class classifier

        Parameters
        ----------
        device : str, optional
            device to be used for training/storing models/..., by default 'default'
            this can be "default", "cpu", "cuda:0", ...
        """

        if device in ['default', None]:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.tmp_save_filename = f"tmp_{random.randint(0, 1e10)}_checkpoint.pkl"
        
        # add colors for printing/logging
        self._print_colors()
         
    def set_classnames(self, classname_dict):
        """Set names of the classes in the dataset

        Parameters
        ----------
        classname_dict : dictionary
            name of the classes in the dataset,
            e.g., {0: "rail space", 1: "No rail space"} 
        """
        self.class_names = classname_dict
        self.num_classes = len(self.class_names)

    def add2dataloader(self, dataset, set_name=None, batch_size=16, shuffle=True, num_workers=0, **kwds):
        """Create and add a dataloader

        Parameters
        ----------
        dataset : pytorch dataset
        set_name : name of the dataset, e.g., train/val/test, optional
        batch_size : int, optional
        shuffle : bool, optional
        num_workers : int, optional
        """

        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwds)
        if set_name == None:
            return dl 
        else:
            self.dataloader[set_name] = dl 
            self.dataset_sizes[set_name] = len(self.dataloader[set_name].dataset)
            print(f"[INFO] added '{set_name}' dataloader with {self.dataset_sizes[set_name]} elements.")

    def print_classes_dl(self, set_name: str="train"):
        """Print classes and classnames (if available)

        Parameters
        ----------
        set_name : str, optional
            Name of the dataset (normally specified in self.add2dataloader), by default "train"
        """
        print(f"[INFO] labels:      {self.dataloader[set_name].dataset.uniq_labels}")
        if self.class_names is not None:
            print(f"[INFO] class-names: {self.class_names}")
    
    def add_model(self, model, input_size=224, is_inception=False):
        """Add a model to classifier object

        Parameters
        ----------
        model : PyTorch model
            See: from torchvision import models
        input_size : int, optional
            input size, by default 224
        is_inception : bool, optional
            is this a inception-type model?, by default False
        """
        if self.class_names == None:
            raise ValueError("[ERROR] specify class names using set_classnames method.")
        else:
            self.print_classes_dl()
        
        self.model = model.to(self.device)
        self.input_size = input_size
        self.is_inception = is_inception
    
    def del_model(self):
        """Delete the model"""
        self.model = None
        self.input_size = None
        self.is_inception = None
        self.metrics = {}
        self.last_epoch = 0
        self.best_loss = torch.tensor(np.inf)
        self.best_epoch = 0
    
    def layerwise_lr(self, min_lr: float, max_lr: float, ltype: str="linspace"):
        """Define layer-wise learning rates

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
        """
        if ltype.lower() in ["line", "linear", "linspace"]:
            list_lrs = np.linspace(min_lr, max_lr, len(list(self.model.named_parameters())))
        elif ltype.lower() in ["log", "geomspace"]:
            list_lrs = np.geomspace(min_lr, max_lr, len(list(self.model.named_parameters())))
        else:
            raise NotImplementedError(f"Implemented methods are: linspace and geomspace")

        list2optim = []
        for i, (name, params) in enumerate(self.model.named_parameters()):
            list2optim.append({'params': params, 'lr': list_lrs[i]})
            
        return list2optim

    def initialize_optimizer(self, 
                             optim_type: str="adam", 
                             params2optim="infer",
                             optim_param_dict: dict={"lr": 1e-3}, 
                             add_optim: bool=True):
        """Initialize an optimizer
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
        """
   
        if params2optim == "infer":
            params2optim = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if optim_type.lower() in ['adam']:
            optimizer = optim.Adam(params2optim, **optim_param_dict)
        elif optim_type.lower() in ['adamw']:
            optimizer = optim.AdamW(params2optim, **optim_param_dict)
        elif optim_type.lower() in ['sgd']:
            optimizer = optim.SGD(params2optim, **optim_param_dict)
        
        if add_optim:
            self.add_optimizer(optimizer)
        else:
            return optimizer

    def add_optimizer(self, optimizer):
        """Add an optimizer to the object"""
        self.optimizer = optimizer
        
    def initialize_scheduler(self, 
                             scheduler_type: str="steplr", 
                             scheduler_param_dict: dict={"step_size": 10, "gamma": 0.1}, 
                             add_scheduler: bool=True):
        """Initialize a scheduler

        Parameters
        ----------
        scheduler_type : str, optional
            scheduler type, by default "steplr"
        scheduler_param_dict : dict, optional
            scheduler parameters, by default {"step_size": 10, "gamma": 0.1}
        add_scheduler : bool, optional
            add scheduler to the object, by default True
        """
        
        if scheduler_type.lower() in ["steplr"]:
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_param_dict)
        elif scheduler_type.lower() in ["onecyclelr"]:
            scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, **scheduler_param_dict)
        else:
            raise ValueError(f"[ERROR] scheduler of type: {scheduler_type} is not implemented.")
            
        if add_scheduler:
            self.add_scheduler(scheduler)
        else:
            return scheduler
    
    def add_scheduler(self, scheduler):
        """Add a scheduler to the object"""
        if self.optimizer is None:
            raise ValueError("[ERROR] optimizer is needed. Use initialize_optimizer or add_optimizer")
        
        self.scheduler = scheduler
    
    def add_criterion(self, criterion):
        """Add a criterion to the object"""
        self.criterion = criterion

    def model_summary(self, only_trainable=False, print_space=[40, 20, 20]):
        """Print model summary

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
        """

        if self.model is None:
            raise ValueError("[ERROR] no model is added. Use .add_model")

        # header
        col1, col2, col3 = "modules", "parameters", "dim"
        line_divider = sum(print_space)*"-" + "----------"
        print(line_divider)
        print(f"| {col1:>{print_space[0]}} | {col2:>{print_space[1]}} | {col3:>{print_space[2]}}")
        print(line_divider)

        # body
        total_params = 0
        total_trainable_params = 0
        for name, parameter in self.model.named_parameters():
            
            if (not parameter.requires_grad) and only_trainable: 
                continue
            elif (not parameter.requires_grad):
                cbeg, cend = self.color_red, self.color_reset
            else:
                cbeg = cend = ""

            param = parameter.numel()
            
            print(f"{cbeg}| {name:>{print_space[0]}} | {param:>{print_space[1]}} | {str(list(parameter.shape)):>{print_space[2]}} |{cend}")

            total_params += param
            if (parameter.requires_grad):
                total_trainable_params += param

        # footer
        print(line_divider)
        if not only_trainable:
            print(f"| {'Total params':>{print_space[0]}} | {total_params:>{print_space[1]}} | {'':>{print_space[2]}} |")
        print(f"| {'Total trainable params':>{print_space[0]}} | {total_trainable_params:>{print_space[1]}} | {'':>{print_space[2]}} |")
        print(line_divider)

        # add 6 to sum(print_space) as we have two times: " | " with size 3 in the other/above prints
        print(f"| {'Other parameters:':<{sum(print_space) + 6}} |")
        print(f"| {'* input size:   '+str(self.input_size):<{sum(print_space) + 6}} |")
        print(f"| {'* is_inception: '+str(self.is_inception):<{sum(print_space) + 6}} |")
        print(line_divider)
    
    def freeze_layers(self, layers_to_freeze: list=[]):
        """Freeze a list of layers, wildcard is accepted

        Parameters
        ----------
        layers_to_freeze : list, optional
            List of layers to freeze, by default []
        """
        
        for one_layer in layers_to_freeze:
            for name, param in self.model.named_parameters():
                if (one_layer[-1] == "*") and (one_layer.replace("*", "") in name):
                    param.requires_grad = False
                elif (one_layer[-1] != "*") and (one_layer == name):
                    param.requires_grad = False


    def unfreeze_layers(self, layers_to_unfreeze: list=[]):
        """Unfreeze a list of layers, wildcard is accepted

        Parameters
        ----------
        layers_to_unfreeze : list, optional
            List of layers to unfreeze, by default []
        """
        for one_layer in layers_to_unfreeze:
            for name, param in self.model.named_parameters():
                if (one_layer[-1] == "*") and (one_layer.replace("*", "") in name):
                    param.requires_grad = True
                elif (one_layer[-1] != "*") and (one_layer == name):
                    param.requires_grad = True

    def only_keep_layers(self, only_keep_layers_list: list=[]):
        """Only keep this list of layers in training

        Parameters
        ----------
        only_keep_layers_list : list, optional
            List of layers to keep, by default []
        """
        for name, param in self.model.named_parameters():
            if name in only_keep_layers_list:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def inference(self, set_name="infer", verbosity_level=0, print_info_batch_freq: int=5):
        """Model inference on dataset: set_name"""
        self.train(phases=[set_name], 
                   num_epochs=1, 
                   save_model_dir=None,
                   verbosity_level=verbosity_level,
                   tensorboard_path=None,
                   tmp_file_save_freq=2,
                   remove_after_load=False,
                   print_info_batch_freq=print_info_batch_freq)

    def train_component_summary(self):
        """Print some info about optimizer/criterion/model..."""
        print(20*"=")
        print("* Optimizer:")
        print(str(self.optimizer))
        print(20*"=")
        print("* Criterion:")
        print(str(self.criterion))
        print(20*"=")
        print("* Model:")
        self.model_summary(only_trainable=True)
    
    def train(self, 
              phases: list=["train", "val"], 
              num_epochs: int=25, 
              save_model_dir: Union[None, str]="models",
              verbosity_level: int=1,
              tensorboard_path: Union[None, str]=None,
              tmp_file_save_freq: int=2,
              remove_after_load: bool=True,
              print_info_batch_freq: int=5):
        """Wrapper function for train_core method to capture exceptions. Supported exceptions so far:
        - KeyboardInterrupt

        Refer to train_core for more information.
        """

        try:
            self.train_core(phases, 
                            num_epochs, 
                            save_model_dir, 
                            verbosity_level, 
                            tensorboard_path,
                            tmp_file_save_freq,
                            print_info_batch_freq=print_info_batch_freq)
        except KeyboardInterrupt:
            print("KeyboardInterrupted...Exiting...")
            if os.path.isfile(self.tmp_save_filename):
                print(f"File found: {self.tmp_save_filename}...load...")
                self.load(self.tmp_save_filename, remove_after_load=remove_after_load)
            else:
                print(f"No temporary file was found.")

    def train_core(self,
                   phases: list = ["train", "val"],
                   num_epochs: int = 25,
                   save_model_dir: Union[None, str] = "models",
                   verbosity_level: int = 1,
                   tensorboard_path: Union[None, str] = None,
                   tmp_file_save_freq: int = 2,
                   print_info_batch_freq: int=5):
        """Train/fine-tune a classifier

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
        """

        if self.criterion is None:
            raise ValueError("[ERROR] criterion is needed. Use add_criterion method")

        for phase in phases:
            if phase not in self.dataloader.keys():
                raise KeyError(
                    f"[ERROR] specified phase: {phase} cannot be find in object's dataloader with keys: {self.dataloader.keys()}"
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
                    f"[WARNING] could not import: from torch.utils.tensorboard import SummaryWriter"
                )
                print(f"[WARNING] continue without tensorboard.")
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
                # batch_loop = tqdm(iter(self.dataloader[phase]), total=len(self.dataloader[phase]), leave=False)
                # if phase.lower() in train_phase_names+valid_phase_names: 
                #     batch_loop.set_description(f"Epoch {epoch}/{end_epoch}")

                phase_batch_size = self.dataloader[phase].batch_size
                total_inp_counts = len(self.dataloader[phase].dataset)

                # --- loop, batches
                for batch_idx, (inputs, labels) in enumerate(self.dataloader[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    if self.optimizer is None:
                        if phase.lower() in train_phase_names:
                            raise ValueError(
                                f"[ERROR] optimizer should be set when phase is {phase}. Use initialize_optimizer or add_optimizer."
                            )
                    else:
                        self.optimizer.zero_grad()

                    if phase.lower() in train_phase_names + valid_phase_names:
                        # forward, track history if only in train
                        with torch.set_grad_enabled(phase.lower() in train_phase_names):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output.
                            #     In train mode we calculate the loss by summing the final output and the auxiliary output
                            #     but in testing we only consider the final output.
                            if self.is_inception and (
                                phase.lower() in train_phase_names
                            ):
                                outputs, aux_outputs = self.model(inputs)
                                loss1 = self.criterion(outputs, labels)
                                loss2 = self.criterion(aux_outputs, labels)
                                # XXX From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
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
                        torch.nn.functional.softmax(outputs, dim=1).cpu().tolist()
                    )
                    running_pred_label.extend(pred_label.cpu().tolist())
                    running_orig_label.extend(labels.cpu().tolist())

                    if batch_idx % print_info_batch_freq == 0: 
                        curr_inp_counts = min(total_inp_counts, (batch_idx+1) * phase_batch_size)
                        progress_perc = curr_inp_counts / total_inp_counts * 100.
                        tmp_str = f"{curr_inp_counts}/{total_inp_counts} ({progress_perc:5.1f}%)"

                        epoch_msg  = f"{phase: <8} -- {epoch}/{end_epoch} -- "
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
                if phase.lower() in train_phase_names and (self.scheduler != None):
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
                        self.cprint("[INFO]", self.color_dred, epoch_msg + "\n")
                    else:
                        self.cprint("[INFO]", self.color_dgreen, epoch_msg)

                # labels/confidence
                self.pred_conf.extend(running_pred_conf)
                self.pred_label.extend(running_pred_label)
                self.orig_label.extend(running_orig_label)

                # Update best_loss and _epoch?
                if phase.lower() in valid_phase_names and epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if phase.lower() in valid_phase_names:
                    if epoch % tmp_file_save_freq == 0:
                        print(
                            f"SAVE temp file: {self.tmp_save_filename} | set .last_epoch: {epoch}"
                        )
                        tmp_str = f"[INFO] SAVE temp file: {self.tmp_save_filename} | set .last_epoch: {epoch}\n"
                        print(self.color_lgrey + tmp_str + self.color_reset)
                        self.last_epoch = epoch
                        self.save(self.tmp_save_filename, force=True)

        time_elapsed = time.time() - since
        print(f'Total time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # --- SAVE model/object
        if phase.lower() in train_phase_names + valid_phase_names:
            self.last_epoch = epoch
            if save_model_dir is not None:
                save_filename = f"checkpoint_{self.best_epoch}.pkl"
                save_model_path = os.path.join(save_model_dir, save_filename)
                self.save(save_model_path, force=True)
                with open(os.path.join(save_model_dir, "info.txt"), "a+") as fio:
                    fio.writelines(f"{save_filename},{self.best_loss:.5f}\n")

                print(
                    f"[INFO] Save model epoch: {self.best_epoch} with least valid loss: {self.best_loss:.4f}"
                )
                print(f"[INFO] Path: {save_model_path}")
    
    def calculate_add_metrics(self, y_true, y_pred, y_score, phase, epoch=-1, tboard_writer=None):
        """Calculate various evaluation metrics (e.g., precision, recall and F1) and add to self.metrics

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
        """
        # convert y_score to a numpy array:
        y_score = np.array(y_score)

        for avrg in ["micro", "macro", "weighted"]:
            prec, rcall, fscore, supp = precision_recall_fscore_support(y_true, y_pred, average=avrg)
            self._add_metrics(f"epoch_prec_{avrg}_{phase}", prec*100.)
            self._add_metrics(f"epoch_recall_{avrg}_{phase}", rcall*100.)
            self._add_metrics(f"epoch_fscore_{avrg}_{phase}", fscore*100.)
            self._add_metrics(f"epoch_supp_{avrg}_{phase}", supp)

            if tboard_writer is not None:
                tboard_writer.add_scalar(f'Precision/{phase}/{avrg}', self.metrics[f"epoch_prec_{avrg}_{phase}"][-1], epoch)
                tboard_writer.add_scalar(f'Recall/{phase}/{avrg}', self.metrics[f"epoch_recall_{avrg}_{phase}"][-1], epoch)
                tboard_writer.add_scalar(f'Fscore/{phase}/{avrg}', self.metrics[f"epoch_fscore_{avrg}_{phase}"][-1], epoch)

            # --- compute ROC AUC
            if y_score.shape[1] == 2:
                # binary case 
                # From scikit-learn: 
                #     The probability estimates correspond to the probability of the class with the greater label, 
                #     i.e. estimator.classes_[1] and thus estimator.predict_proba(X, y)[:, 1]
                roc_auc = roc_auc_score(y_true, y_score[:, 1], average=avrg)
            elif (y_score.shape[1] != 2) and (avrg in ['macro', 'weighted']):
                # multiclass
                # In the multiclass case, it corresponds to an array of shape (n_samples, n_classes)
                try:
                    roc_auc = roc_auc_score(y_true, y_score, average=avrg, multi_class="ovr")
                except:
                    continue
            else:
                continue

            self._add_metrics(f"epoch_rocauc_{avrg}_{phase}", roc_auc*100.)

        prfs = precision_recall_fscore_support(y_true, y_pred, average=None)
        for i in range(len(prfs[0])):
            self._add_metrics(f"epoch_prec_{i}_{phase}", prfs[0][i]*100.)
            self._add_metrics(f"epoch_recall_{i}_{phase}", prfs[1][i]*100.)
            self._add_metrics(f"epoch_fscore_{i}_{phase}", prfs[2][i]*100.)
            self._add_metrics(f"epoch_supp_{i}_{phase}", prfs[3][i])
            
            if tboard_writer is not None:
                tboard_writer.add_scalar(f'Precision/{phase}/binary_{i}', self.metrics[f"epoch_prec_{i}_{phase}"][-1], epoch)
                tboard_writer.add_scalar(f'Recall/{phase}/binary_{i}', self.metrics[f"epoch_recall_{i}_{phase}"][-1], epoch)
                tboard_writer.add_scalar(f'Fscore/{phase}/binary_{i}', self.metrics[f"epoch_fscore_{i}_{phase}"][-1], epoch)
    
    def gen_epoch_msg(self, phase, epoch_msg):
        tmp_loss = self.metrics[f"epoch_loss_{phase}"][-1]
        epoch_msg += f"Loss: {tmp_loss:.3f}; "
        tmp_fscore = self.metrics[f'epoch_fscore_macro_{phase}'][-1]
        epoch_msg += f"F_macro: {tmp_fscore:.2f}; "
        tmp_recall = self.metrics[f'epoch_recall_macro_{phase}'][-1]
        epoch_msg += f"R_macro: {tmp_recall:.2f}"
        return epoch_msg
    
    def _add_metrics(self, k, v):
        """Add metric k with value v to self.metrics"""
        if not k in self.metrics.keys():
            self.metrics[k] = [v]
        else:
            self.metrics[k].append(v)

    def plot_metric(self, y_axis, y_label, legends,
                    x_axis="epoch", x_label="epoch",
                    colors=5*["k", "tab:red"], 
                    styles=10*["-"],
                    markers=10*["o"],
                    figsize=(10, 5),
                    plt_yrange=None, plt_xrange=None):
        """Plot content of self.metrics

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
        """

        # Font sizes
        plt_size = {"xlabel": 24, "ylabel": 24,
                    "xtick": 18,  "ytick": 18,
                    "legend": 18}

        fig = plt.figure(figsize=figsize)
        if x_axis == "epoch":
            from matplotlib.ticker import MaxNLocator
            # make x ticks integer
            fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        for i, one_item in enumerate(y_axis):
            if one_item not in self.metrics.keys():
                print(f"[WARNING] requested item: {one_item} not in keys: {self.metrics.keys}")
                continue
            
            if x_axis == "epoch":
                x_axis_plt = range(1, len(self.metrics[one_item])+1)
            else:
                x_axis_plt = self.metrics[x_axis]

            plt.plot(x_axis_plt, self.metrics[one_item],
                     label=legends[i], 
                     color=colors[i],
                     ls=styles[i],
                     marker=markers[i],
                     lw=3)

        # --- labels and ticks
        plt.xlabel(x_label, size=plt_size["xlabel"])
        plt.ylabel(y_label, size=plt_size["ylabel"])
        plt.xticks(size=plt_size["xtick"])
        plt.yticks(size=plt_size["ytick"])
        # --- legend
        plt.legend(fontsize=plt_size["legend"],
                   bbox_to_anchor=(0, 1.02, 1, 0.2), 
                   ncol=2, borderaxespad=0, 
                   loc="lower center")
        # --- x/y range
        if plt_xrange is not None:
            plt.xlim(plt_xrange[0], plt_xrange[1])
        if plt_yrange is not None:
            plt.ylim(plt_yrange[0], plt_yrange[1])

        plt.grid()
        plt.show()

    def initialize_model(self, model_name, pretrained=True, last_layer_num_classes="default", add_model=True):
        """Initialize a PyTorch model
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
            model_dw.classifier[6] = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "vgg" in model_name:
            # vgg11_bn
            num_ftrs = model_dw.classifier[6].in_features
            model_dw.classifier[6] = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "squeezenet" in model_name:
            model_dw.classifier[1] = nn.Conv2d(512, last_layer_num_classes, kernel_size=(1,1), stride=(1,1))
            model_dw.num_classes = last_layer_num_classes

        elif "densenet" in model_name:
            num_ftrs = model_dw.classifier.in_features
            model_dw.classifier = nn.Linear(num_ftrs, last_layer_num_classes)

        elif "inception" in model_name:
            # Inception v3, Be careful, expects (299,299) sized images and has auxiliary output

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
            self.add_model(model_dw, input_size=input_size, is_inception=is_inception)
        else:
            return model_dw, input_size, is_inception
    
    def show_sample(self, set_name="train", batch_number=1, print_batch_info=True, figsize=(15, 10)):
        """Show samples from specified dataset

        Parameters
        ----------
        set_name : str, optional
            name of the dataset, by default "train"
        batch_number : int, optional
            batch number to be plotted, by default 1
        figsize : tuple, optional
            size of the figure, by default (15, 10)
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
            out, title=str([self.class_names[int(x)] for x in classes]), figsize=figsize
        )
    
    def batch_info(self, set_name="train"):
        """Print info about samples/batch-size/...

        Parameters
        ----------
        set_name : str, optional
            name of the dataset, by default "train"
        """
        batch_size = self.dataloader[set_name].batch_size
        num_samples = len(self.dataloader[set_name].dataset)
        num_batches = int(np.ceil(num_samples / batch_size))
        print(f"[INFO] dataset: {set_name}")
        print(f"#samples:    {num_samples}")
        print(f"#batch size: {batch_size}")
        print(f"#batches:    {num_batches}")

    @staticmethod
    def _imshow(inp, title=None, figsize=(15, 10)):
        """imshow for tensors

        Parameters
        ----------
        inp : tensor
            tensor to be plotted
        title : str or None, optional
            title of the figure, by default None
        figsize : tuple, optional
            size of the figure, by default (15, 10)
        """

        inp = inp.numpy().transpose((1, 2, 0))
        # XXX
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean

        inp = np.clip(inp, 0, 1)
        plt.figure(figsize=(15, 10))
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()

    def inference_sample_results(self, 
                                 num_samples: int = 6, 
                                 class_index: int = 0, 
                                 set_name: str = "train", 
                                 min_conf: Union[None, float] = None, 
                                 max_conf: Union[None, float] = None,
                                 figsize: tuple = (15, 15)):
        """Plot some samples (specified by num_samples) for inference outputs

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
                pred_conf = torch.nn.functional.softmax(outputs, dim=1) * 100.
                _, preds = torch.max(outputs, 1)

                # go through images in batch
                for j in range(len(preds)):
                    pred_ind = int(preds[j])
                    if pred_ind != class_index:
                        continue
                    if (min_conf is not None) and (pred_conf[j][pred_ind] < min_conf):
                        continue
                    if (max_conf is not None) and (pred_conf[j][pred_ind] > max_conf):
                        continue

                    counter += 1
                    ax = plt.subplot(int(num_samples/2.), 3, counter)
                    ax.axis('off')
                    ax.set_title(f'{self.class_names[pred_ind]} | {pred_conf[j][pred_ind]:.3f}')

                    inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    inp = np.clip(inp, 0, 1)
                    plt.imshow(inp)

                    if counter == num_samples:
                        self.model.train(mode=was_training)
                        plt.show()
                        return
            self.model.train(mode=was_training)
            plt.show()
    
    def save(self, save_path="default.obj", force=False):
        """Save object"""
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
        with open(save_path, 'wb') as myfile:
            #pickle.dump(self.__dict__, myfile)
            joblib.dump(obj2write, myfile)

        torch.save(mymodel, os.path.join(par_name, f"model_{base_name}"))

    def load(self, load_path, remove_after_load=False, force_device=False):
        """load class"""


        load_path = os.path.abspath(load_path)        
        mydevice = self.device
        
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"file not found: {load_path}")

        with open(load_path, 'rb') as myfile:
            #objPickle = pickle.load(myfile)
            objPickle = joblib.load(myfile)

        if remove_after_load:
            os.remove(load_path)

        self.__dict__ = objPickle 

        if force_device:
            if not isinstance(force_device, str):
                force_device = str(force_device)
            os.environ['CUDA_VISIBLE_DEVICES'] = force_device

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
        # color
        self.color_lgrey = '\033[1;90m'
        self.color_grey = '\033[90m'           # broing information
        self.color_yellow = '\033[93m'         # FYI
        self.color_orange = '\033[0;33m'       # Warning

        self.color_lred = '\033[1;31m'         # there is smoke
        self.color_red = '\033[91m'            # fire!
        self.color_dred = '\033[2;31m'         # Everything is on fire

        self.color_lblue = '\033[1;34m'
        self.color_blue = '\033[94m'
        self.color_dblue = '\033[2;34m'

        self.color_lgreen = '\033[1;32m'       # all is normal
        self.color_green = '\033[92m'          # something else
        self.color_dgreen = '\033[2;32m'       # even more interesting

        self.color_lmagenta = '\033[1;35m'
        self.color_magenta = '\033[95m'        # for title
        self.color_dmagenta = '\033[2;35m'

        self.color_cyan = '\033[96m'           # system time
        self.color_white = '\033[97m'          # final time

        self.color_black = '\033[0;30m'

        self.color_reset = '\033[0m'
        self.color_bold = '\033[1m'
        self.color_under = '\033[4m'

    def get_time(self):
        time = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
        return time

    def cprint(self, type_info, bc_color, text):
        """
        simple print function used for colored logging
        """
        ho_nam = socket.gethostname().split('.')[0][:10]
    
        print(self.color_green                  + self.get_time() + self.color_reset,
              self.color_magenta                + ho_nam          + self.color_reset,
              self.color_bold + self.color_grey + type_info       + self.color_reset,
              bc_color                          + text            + self.color_reset)

    def update_progress(self, progress, text="", barLength=30):
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength*progress))
        text = f"\r[{'#'*block + '-'*(barLength-block)}] {progress*100:.1f}% {status} {text}"
        sys.stdout.write(text)
        sys.stdout.flush()
