#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time
# from tqdm.autonotebook import tqdm
from typing import Union

import torch
from .classifier import classifier

class classifierContext(classifier):

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
                for batch_idx, (inputs1, inputs2, labels) in enumerate(self.dataloader[phase]):
                    inputs1 = inputs1.to(self.device)
                    inputs2 = inputs2.to(self.device)
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
                                outputs, aux_outputs = self.model(inputs1, inputs2)
                                loss1 = self.criterion(outputs, labels)
                                loss2 = self.criterion(aux_outputs, labels)
                                # XXX From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                loss = loss1 + 0.4 * loss2
                            else:
                                outputs = self.model(inputs1, inputs2)
                                # labels = labels.long().squeeze_()
                                loss = self.criterion(outputs, labels)

                            _, pred_label = torch.max(outputs, dim=1)

                            # backward + optimize only if in training phase
                            if phase.lower() in train_phase_names:
                                loss.backward()
                                self.optimizer.step()

                        # XXX (why multiply?)
                        running_loss += loss.item() * inputs1.size(0)

                        # TQDM
                        # batch_loop.set_postfix(loss=loss.data)
                        # batch_loop.refresh()
                    else:
                        outputs = self.model(inputs1, inputs2)
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
            inputs1, inputs2, classes = next(dl_iter)

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs1)
        self._imshow(
            out, title=str([self.class_names[int(x)] for x in classes]), figsize=figsize
        )

        out = torchvision.utils.make_grid(inputs2)
        self._imshow(
            out, title=str([self.class_names[int(x)] for x in classes]), figsize=figsize
        )

    def layerwise_lr(self, min_lr: float, max_lr: float, ltype: str="linspace", sep_group_names=["features1", "features2"]):
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
                
        list2optim = []
        for one_grp in range(len(sep_group_names)):

            # count number of layers in this group
            num_grp_layers = 0
            for i, (name, params) in enumerate(self.model.named_parameters()):
                if sep_group_names[one_grp] in name:
                    num_grp_layers += 1

            # define layer-wise learning rates
            if ltype.lower() in ["line", "linear", "linspace"]:
                list_lrs = np.linspace(min_lr, max_lr, num_grp_layers)
            elif ltype.lower() in ["log", "geomspace"]:
                list_lrs = np.geomspace(min_lr, max_lr, num_grp_layers)
            else:
                raise NotImplementedError(f"Implemented methods are: linspace and geomspace")

            # assign learning rates
            i_count = 0
            for i, (name, params) in enumerate(self.model.named_parameters()):
                if not sep_group_names[one_grp] in name:
                    continue
                list2optim.append({'params': params, 'lr': list_lrs[i_count]})
                i_count += 1
           
        return list2optim

    def inference_sample_results(
        self,
        num_samples: int = 6,
        class_index: int = 0,
        set_name: str = "train",
        min_conf: Union[None, float] = None,
        max_conf: Union[None, float] = None,
        figsize: tuple = (15, 15)
    ):
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
            for inputs1, inputs2, labels in iter(self.dataloader[set_name]):
                inputs1 = inputs1.to(self.device)
                labels = labels.to(self.device)
                inputs2 = inputs2.to(self.device)

                outputs = self.model(inputs1, inputs2)
                pred_conf = torch.nn.functional.softmax(outputs, dim=1) * 100.0
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
                    ax = plt.subplot(int(num_samples / 2.0), 3, counter)
                    ax.axis("off")
                    ax.set_title(
                        f"{self.class_names[pred_ind]} | {pred_conf[j][pred_ind]:.3f}"
                    )

                    inp = inputs1.cpu().data[j].numpy().transpose((1, 2, 0))
                    inp = np.clip(inp, 0, 1)
                    plt.imshow(inp)

                    if counter == num_samples:
                        self.model.train(mode=was_training)
                        plt.show()
                        return
            self.model.train(mode=was_training)
            plt.show()
