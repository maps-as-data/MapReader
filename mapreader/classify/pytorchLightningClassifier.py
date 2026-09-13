#!/usr/bin/env python
from __future__ import annotations
import os
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import optim
from torchinfo import summary
from torchvision import models
from pytorch_lightning import LightningModule
from itertools import islice

class PytorchLightningClassifier(LightningModule):

    """
    A class to store and train a PyTorch model as a pytorch-lightning model.

    Parameters
    ----------
    model : str, nn.Module or None
        The PyTorch model to add to the object.

        - If passed as a string, will run ``_initialize_model(model, **kwargs)``. See https://pytorch.org/vision/0.8/models.html for options.
        - Must be ``None`` if ``load_path`` is specified as model will be loaded from file.

    labels_map: Dict or None
        A dictionary containing the mapping of each label index to its label, with indices as keys and labels as values (i.e. idx: label).
        Can only be ``None`` if ``load_path`` is specified as labels_map will be loaded from file.
    dataloaders: Dict or None
        A dictionary containing set names as keys and dataloaders as values (i.e. set_name: dataloader).
    device : str, optional
        The device to be used for training and storing models.
        Can be set to "default", "cpu", "cuda:0", etc. By default, "default".
    input_size : int, optional
        The expected input size of the model. Default is ``(224,224)``.
    is_inception : bool, optional
        Whether the model is an Inception-style model.
        Default is ``False``.
    load_path : str, optional
        The path to an ``.obj`` file containing a
    force_device : bool, optional
        Whether to force the use of a specific device.
        If set to ``True``, the default device is used.
        Defaults to ``False``.
    kwargs : Dict
        Keyword arguments to pass to the
        :meth:`~.classify.classifier.ClassifierContainer._initialize_model`
        method (if passing ``model`` as a string).

    Attributes
    ----------
    device : torch.device
        The device being used for training and storing models.
    dataloaders : dict
        A dictionary to store dataloaders for the model.
    labels_map : dict
        A dictionary mapping label indices to their labels.
    dataset_sizes : dict
        A dictionary to store sizes of datasets for the model.
    model : torch.nn.Module
        The model.
    input_size : Tuple of int
        The size of the input to the model.
    is_inception : bool
        A flag indicating if the model is an Inception model.
    optimizer : None or torch.optim.Optimizer
        The optimizer being used for training the model.
    scheduler : None or torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler being used for training the model.
    loss_fn : None or nn.modules.loss._Loss
        The loss function to use for training the model.
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

    def __init__(
        self,
        model: str | nn.Module | None = None,
        labels_map: dict[int, str] | None = None,
        loss_fn: str | nn.modules.loss._Loss | None = None,
        input_size: int = (224, 224),
        is_inception: bool = False,
        load_path: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._initialize_model(model, input_size, is_inception, load_path, **kwargs)
        # check if loading an pre-existing object
        if load_path:
            if model:
                print("[WARNING] Ignoring ``model`` as ``load_path`` is specified.")

            self.load(load_path=load_path, force_device=self.device)

            self.labels_map = labels_map ## TODOL ConsiderLoading this from load from trainer,dataloader, dataset, labels map during training rather than parsing it. 

            # set up model and move to device
            print("[INFO] Initializing model.")
            if isinstance(model, nn.Module):
                self.model = model.to(self.device)
                self.input_size = input_size
                self.is_inception = is_inception


            elif isinstance(model, str):
                self._initialize_model(model, **kwargs)
                self.forward= self.forward_logits
      
        if isinstance(loss_fn, str):
            if loss_fn in ["cross entropy", "ce", "cross_entropy", "cross-entropy"]:
                loss_fn = nn.CrossEntropyLoss()
            elif loss_fn in [
                "bce",
                "binary_cross_entropy",
                "binary cross entropy",
                "binary cross-entropy",
            ]:
                loss_fn = nn.BCELoss()
            elif loss_fn in [
                "mse",
                "mean_square_error",
                "mean_squared_error",
                "mean squared error",
            ]:
                loss_fn = nn.MSELoss()
            else:
                raise NotImplementedError(
                    '[ERROR] At present, if passing ``loss_fn`` as a string, the loss function can only be "cross entropy" or "ce" (cross-entropy), "bce" (binary cross-entropy) or "mse" (mean squared error).'
                )

            print(f'[INFO] Using "{loss_fn}" as loss function.')

        elif not isinstance(loss_fn, nn.modules.loss._Loss):
            raise ValueError(
                '[ERROR] Please pass ``loss_fn`` as a string ("cross entropy", "bce" or "mse") or torch.nn loss function (see https://pytorch.org/docs/stable/nn.html).'
            )

        self.loss_fn = loss_fn
        if self.is_inception:
            self.loss_calc = self.inception_loss_calc
    
    def forward_logits(self, inputs):
        return self.model(*inputs).logits

    def forward_tensor(self, x):
        return self.model(*x)
    
    def _initialize_model(
        self,
        model_name: str,
        weights: str | None = "DEFAULT",
        last_layer_num_classes: str | int | None = "default",
    ) -> tuple[Any, int, bool]:
        """
        Initializes a PyTorch model with the option to change the number of
        classes in the last layer (``last_layer_num_classes``).

        Parameters
        ----------
        model_name : str
            Name of a PyTorch model. See https://pytorch.org/vision/0.8/models.html for options.
        weights : str, optional
            Weights to load into the model. If ``"DEFAULT"``, loads the default weights for the chosen model.
            By default, ``"DEFAULT"``.
        last_layer_num_classes : str or int, optional
            Number of elements in the last layer. If ``"default"``, sets it to
            the number of classes. By default, ``"default"``.

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

        See https://pytorch.org/vision/0.8/models.html.
        """

        # Initialize these variables which will be set in this if statement.
        # Each of these variables is model specific.
        model_dw = models.get_model(model_name, weights=weights)
        input_size = (224, 224)
        is_inception = False

        if last_layer_num_classes in ["default"]:
            last_layer_num_classes = len(self.labels_map)
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
            raise NotImplementedError(
                "[ERROR] Invalid model name. Try loading your model directly and this as the `model` argument instead."
            )

        self.model = model_dw.to(self.device)
        self.input_size = input_size
        self.is_inception = is_inception

    def generate_layerwise_lrs(
        self,
        min_lr: float,
        max_lr: float,
        spacing: str | None = "linspace",
    ) -> list[dict]:
        """
        Calculates layer-wise learning rates for a given set of model
        parameters.

        Parameters
        ----------
        min_lr : float
            The minimum learning rate to be used.
        max_lr : float
            The maximum learning rate to be used.
        spacing : str, optional
            The type of sequence to use for spacing the specified interval
            learning rates. Can be either ``"linspace"`` or ``"geomspace"``,
            where `"linspace"` uses evenly spaced learning rates over a
            specified interval and `"geomspace"` uses learning rates spaced
            evenly on a log scale (a geometric progression). By default ``"linspace"``.

        Returns
        -------
        list of dicts
            A list of dictionaries containing the parameters and learning
            rates for each layer.
        """

        if spacing.lower() not in ["linspace", "geomspace"]:
            raise NotImplementedError(
                '[ERROR] ``spacing`` must be one of "linspace" or "geomspace"'
            )

        if spacing.lower() == "linspace":
            lrs = np.linspace(min_lr, max_lr, len(list(self.model.named_parameters())))
        elif spacing.lower() in ["log", "geomspace"]:
            lrs = np.geomspace(min_lr, max_lr, len(list(self.model.named_parameters())))
        params2optimize = [
            {"params": params, "learning rate": lr}
            for (_, params), lr in zip(self.model.named_parameters(), lrs)
        ]
        self.params2optimize = params2optimize
        return params2optimize

    #call the pytorch lightning optimizer function
    def configure_optimizers(self,):
        optimizer, scheduler = None, None
        if self.optim_param_dict is None:
            optim_param_dict = {"lr": 0.001}
        
        if self.params2optimize == "default":
            params2optimize = filter(lambda p: p.requires_grad, self.model.parameters())
        elif isinstance(self.params2optimize, list):
            params2optimize = self.params2optimize
        else:
            raise ValueError(
                '[ERROR] ``params2optimize`` must be a list of dictionaries containing the parameters and learning rates for each layer.'
            )

        if self.optim_type.lower() in ["adam"]:
            optimizer = optim.Adam(params2optimize, **optim_param_dict)
        elif self.optim_type.lower() in ["adamw"]:
            optimizer = optim.AdamW(params2optimize, **optim_param_dict)
        elif self.optim_type.lower() in ["sgd"]:
            optimizer = optim.SGD(params2optimize, **optim_param_dict)
        else:
            raise NotImplementedError(
                '[ERROR] At present, only Adam ("adam"), AdamW ("adamw") and SGD ("sgd") are options for ``optim_type``.'
            )
       
        if self.scheduler_type.lower() == "steplr":
            if scheduler_param_dict is None:
                scheduler_param_dict = {"step_size": 10, "gamma": 0.1}
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, **scheduler_param_dict
            )
        else:
            raise NotImplementedError(
                '[ERROR] At present, only StepLR ("steplr") is implemented. \n\n\
            Use ``torch.optim.lr_scheduler`` directly and then the ``add_scheduler`` method for other schedulers.'  # noqa
            )
        return optimizer, scheduler


    def model_summary(
        self,
        input_size: tuple | list | None = None,
        trainable_col: bool = False,
        **kwargs,
    ) -> None:
        """
        Print a summary of the model.

        Parameters
        ----------
        input_size : tuple or list, optional
            The size of the input data.
            If None, input size is taken from "train" dataloader (``self.dataloaders["train"]``).
        trainable_col : bool, optional
            If ``True``, adds a column showing which parameters are trainable.
            Defaults to ``False``.
        **kwargs : Dict
            Keyword arguments to pass to ``torchinfo.summary()`` (see https://github.com/TylerYep/torchinfo).

        Notes
        -----
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
        """
        if not input_size:
            if "train" in self.dataloaders.keys():
                batch_size = self.dataloaders["train"].batch_size
                channels = len(self.dataloaders["train"].dataset.image_mode)
                input_size = (batch_size, channels, *self.input_size)
            else:
                raise ValueError("[ERROR] Please pass an input size.")

        if trainable_col:
            col_names = ["num_params", "output_size", "trainable"]
        else:
            col_names = ["output_size", "output_size", "num_params"]

        model_summary = summary(
            self.model, input_size=input_size, col_names=col_names, **kwargs
        )
        print(model_summary)

    def freeze_layers(self, layers_to_freeze: list[str]) -> None:
        """
        Freezes the specified layers in the neural network by setting
        ``requires_grad`` attribute to False for their parameters.

        Parameters
        ----------
        layers_to_freeze : list of str
            List of names of the layers to freeze.
            If a layer name ends with an asterisk (``"*"``), then all parameters whose name contains the layer name (excluding the asterisk) are frozen. Otherwise, only the parameters with an exact match to the layer name are frozen.

        Returns
        -------
        None
            The function only modifies the ``requires_grad`` attribute of the
            specified parameters and does not return anything.

        Notes
        -----
        e.g. ["layer1*", "layer2*"] will freeze all parameters whose name contains "layer1" and "layer2" (excluding the asterisk).
        e.g. ["layer1", "layer2"] will freeze all parameters with an exact match to "layer1" and "layer2".
        """
        if not isinstance(layers_to_freeze, list):
            raise ValueError(
                '[ERROR] ``layers_to_freeze`` must be a list of strings. E.g. ["layer1*", "layer2*"].'
            )

        for layer in layers_to_freeze:
            for name, param in self.model.named_parameters():
                if (layer.endswith("*")) and (
                    layer.strip("*") in name
                ):  # if using asterix wildcard
                    param.requires_grad = False
                elif (not layer.endswith("*")) and (
                    layer == name
                ):  # if using exact match
                    param.requires_grad = False

    def unfreeze_layers(self, layers_to_unfreeze: list[str]):
        """
        Unfreezes the specified layers in the neural network by setting
        ``requires_grad`` attribute to True for their parameters.

        Parameters
        ----------
        layers_to_unfreeze : list of str
            List of names of the layers to unfreeze.
            If a layer name ends with an asterisk (``"*"``), then all parameters whose name contains the layer name (excluding the asterisk) are unfrozen. Otherwise, only the parameters with an exact match to the layer name are unfrozen.

        Returns
        -------
        None
            The function only modifies the ``requires_grad`` attribute of the
            specified parameters and does not return anything.

        Notes
        -----
        e.g. ["layer1*", "layer2*"] will unfreeze all parameters whose name contains "layer1" and "layer2" (excluding the asterisk).
        e.g. ["layer1", "layer2"] will unfreeze all parameters with an exact match to "layer1" and "layer2".
        """

        if not isinstance(layers_to_unfreeze, list):
            raise ValueError(
                '[ERROR] ``layers_to_unfreeze`` must be a list of strings. E.g. ["layer1*", "layer2*"].'
            )

        for layer in layers_to_unfreeze:
            for name, param in self.model.named_parameters():
                if (layer.endswith("*")) and (layer.strip("*") in name):
                    param.requires_grad = True
                elif (not layer.endswith("*")) and (layer == name):
                    param.requires_grad = True

    def only_keep_layers(self, only_keep_layers_list: list[str]) -> None:
        """
        Only keep the specified layers (``only_keep_layers_list``) for
        gradient computation during the backpropagation.

        Parameters
        ----------
        only_keep_layers_list : list
            List of layer names to keep. All other layers will have their
            gradient computation turned off.

        Returns
        -------
        None
            The function only modifies the ``requires_grad`` attribute of the
            specified parameters and does not return anything.
        """
        if not isinstance(only_keep_layers_list, list):
            raise ValueError(
                '[ERROR] ``only_keep_layers_list`` must be a list of strings. E.g. ["layer1", "layer2"].'
            )

        for name, param in self.model.named_parameters():
            if name in only_keep_layers_list:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def on_train_start(self) -> None:
        self.pred_conf = []
        self.pred_label_indices = []
        self.gt_label_indices = []

    def on_train_epoch_start(self) -> None:
        self.running_pred_conf = []
        self.running_pred_label_indices = []
        self.running_gt_label_indices = []

    def on_train_epoch_end(self) -> None:
        #calculate PRF1 
        self.calculate_add_metrics(
                        self.running_pred_conf,
                        self.running_pred_label_indices,
                        self.running_gt_label_indices,
                        "train",
                        self.current_epoch,
                        tboard_writer=None,
                    )
        self.pred_conf.extend(self.running_pred_conf)
        self.pred_label_indices.extend(self.running_pred_label_indices)
        self.gt_label_indices.extend(self.running_gt_label_indices)

    def inception_loss_calc(self, inputs, gt_label_indices):
        outputs, aux_outputs = self.model(*inputs)
        loss1 = self.loss_fn(outputs.logits, gt_label_indices)
        loss2 = self.loss_fn(aux_outputs.logits, gt_label_indices)
        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
        loss = loss1 + 0.4 * loss2
        return loss,outputs

    def base_loss_calc(self, inputs, gt_label_indices):
        outputs = self.model(*inputs)
        loss = self.loss_fn(outputs.logits, gt_label_indices)
        return loss,outputs

    def train_step(self, batch, batch_idx):
        inputs = tuple(input.to(self.device) for input in inputs)
        gt_label_indices = gt_label_indices.to(self.device)
        loss,outputs=self.loss_calc(inputs, gt_label_indices)

        _, pred_label_indices = torch.max(outputs, dim=1)

        self.running_pred_conf.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().tolist())
        self.running_pred_label_indices.extend(pred_label_indices.cpu().tolist())
        self.running_gt_label_indices.extend(gt_label_indices.cpu().tolist())

        return loss
    
    def on_inference_start(self) -> None:
        self.pred_conf = []
        self.pred_label_indices = []

    def on_inference_epoch_start(self) -> None:
        self.running_pred_conf = []
        self.running_pred_label_indices = []
        
    def inference_step(self, batch, batch_idx):
        inputs = tuple(input.to(self.device) for input in inputs)
        outputs = self.forward(inputs)

        _, pred_label_indices = torch.max(outputs, dim=1)

        self.running_pred_conf.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().tolist())
        self.running_pred_label_indices.extend(pred_label_indices.cpu().tolist())
    

    def on_inference_epoch_end(self):
        self.pred_conf.extend(self.running_pred_conf)
        self.pred_label_indices.extend(self.running_pred_label_indices)

    def on_inference_end(self):
        self.pred_text_label = [self.labels_map.get(i, None) for i in self.pred_label_indices]

        patch_df = self.trainer.dataloader.dataset.patch_df
        patch_df["predicted_label"] = self.pred_text_label
        patch_df["pred"] = self.pred_label_indices
        patch_df["conf"] = np.array(self.pred_conf).max(axis=1)

        if save_path is None:
            save_path = f"eval_predictions_patch_df.csv"
        patch_df.to_csv(save_path, sep=',', index=False)
        print(f"[INFO] Saved predictions to {save_path}.")

    def calculate_add_metrics(
        self,
        y_true,
        y_pred,
        y_score,
    ) -> None:
        """
        Calculate and add metrics to the classifier's metrics dictionary.

        Parameters
        ----------
        y_true : 1d array-like of shape (n_samples,)
            True binary labels or multiclass labels. Can be considered ground
            truth or (correct) target values.

        y_pred : 1d array-like of shape (n_samples,)
            Predicted binary labels or multiclass labels. The estimated
            targets as returned by a classifier.

        y_score : array-like of shape (n_samples, n_classes)
            Predicted probabilities for each class.


        Returns
        -------
        None

        Notes
        -----
        This method uses both the
        :func:`sklearn.metrics.precision_recall_fscore_support` and
        :func:`sklearn.metrics.roc_auc_score` functions from ``scikit-learn``
        to calculate the metrics for each average type (``"micro"``,
        ``"macro"`` and ``"weighted"``). The results are then added to the
        ``metrics`` dictionary. It also writes the metrics to the TensorBoard
        SummaryWriter, if ``tboard_writer`` is not None.
        """
        # convert y_score to a numpy array:
        if not isinstance(y_score, np.ndarray):
            y_score = np.array(y_score)
 
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        for i in range(y_score.shape[1]):  # y_score.shape[1] represents the number of classes
                    self.log(f"precision_{i}", precision[i])
                    self.log(f"recall_{i}", recall[i])
                    self.log(f"fscore_{i}", fscore[i])
                    self.log( f"support_{i}", support[i])
        for average in [None, "micro", "macro", "weighted"]:
                precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=average)
                self.log( f"precision_{average}", precision)
                self.log( f"recall_{average}", recall)
                self.log( f"fscore_{average}", fscore)
                self.log( f"support_{average}", support)


                # --- compute ROC AUC
                if y_score.shape[1] == 2:
                    # ---- binary case
                    # From scikit-learn:
                    #     The probability estimates correspond to the probability
                    #     of the class with the greater label, i.e.
                    #     estimator.classes_[1] and thus
                    #     estimator.predict_proba(X, y)[:, 1]
                    roc_auc = roc_auc_score(y_true, y_score[:, 1], average=average)
                elif y_score.shape[1] > 2:
                    # ---- multiclass
                    # In the multiclass case, it corresponds to an array of shape
                    # (n_samples, n_classes)
                    # ovr = One-vs-rest (OvR) multiclass strategy
                    try:
                        roc_auc = roc_auc_score(
                            y_true, y_score, average=average, multi_class="ovr"
                        )
                    except:
                        continue
                else:
                    continue
                self.log( f"rocauc_{average}", roc_auc)
    @torch.no_grad()
    def show_inference_sample_results(
        self,
        label: str,
        num_samples: int | None = 6,
        set_name: str | None = "test",
        min_conf: None | float | None = None,
        max_conf: None | float | None = None,
        figsize: tuple[int, int] | None = (15, 15),
    ) -> None:
        """
        Shows a sample of the results of the inference with current model.

        Parameters
        ----------
        label : str, optional
            The label for which to display results.
        num_samples : int, optional
            The number of sample results to display. Defaults to ``6``.
        set_name : str, optional
            The name of the dataset split to use for inference. Defaults to
            ``"test"``.
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
        plt.figure(figsize=figsize)
        counter = 0
        for inputs, _, label_indices in islice(iter(self.dataloaders[set_name]), num_samples):
            inputs = tuple(input.to(self.device) for input in inputs)
            label_indices = label_indices.to(self.device)
            outputs = self.forward(inputs)
            pred_conf = torch.nn.functional.softmax(outputs, dim=1) * 100.0
            _, preds = torch.max(outputs, 1)

            # reverse the labels_map dict
            label_index_dict = {v: k for k, v in self.labels_map.items()}

            # go through images in batch
            for i, pred in enumerate(preds):
                predicted_index = int(pred)
                if predicted_index != label_index_dict[label]:
                    continue
                if (min_conf is not None) and ( pred_conf[i][predicted_index] < min_conf):
                    continue
                if (max_conf is not None) and ( pred_conf[i][predicted_index] > max_conf):
                    continue

                counter += 1

                conf_score = pred_conf[i][predicted_index]
                ax = plt.subplot(int(num_samples / 2.0), 3, counter)
                ax.axis("off")
                ax.set_title(f"{label} | {conf_score:.3f}")

                inp = inputs[0].cpu().data[i].numpy().transpose((1, 2, 0))
                inp = np.clip(inp, 0, 1)
                plt.imshow(inp)
        plt.show()


if __name__ == "__main__":

    import argparse 

    parser = argparse.ArgumentParser(description='PyTorchLightningClassifier') 
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--input_size', type=int, default=(224,224), help='input size')
    parser.add_argument('--is_inception', type=bool, default=False, help='is inception model')
    parser.add_argument('--load_path', type=str, default=None, help='load path')
    parser.add_argument('--kwargs', type=dict, default={}, help='kwargs')
    parser.add_argument('--save_path', type=str, default=None, help='save path')
    args = parser.parse_args()
    #for more sophisticated sweeps, have a look at W&B, neptune or the test-tube library

    import pytorch_lightning
    from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping

    logtool=pytorch_lightning.loggers.NeptuneLogger( project="MapReader",entity="YOURUSER",experiment="classifier Training", save_dir=args.save_path)

    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="train_loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
        #save best model
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor='train_loss',
            dirpath=dir,
            filename=os.path.join(args.save_path,"{epoch}-{train_loss:.2f}"),
            save_top_k=1,
            mode='min',
            save_last=True,),
    ]
    
    trainer=pytorch_lightning.Trainer(
            devices=1,
            accelerator="gpu",
            max_epochs=20,
            #profiler="advanced",
            #plugins=[SLURMEnvironment()],
            #https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
            logger=logtool,
            # strategy=FSDPStrategy(accelerator="gpu",
            #                        parallel_devices=6 if not EVALOnLaunch else 1,
            #                        cluster_environment=SLURMEnvironment(),
            #                        timeout=datetime.timedelta(seconds=1800),
            #                        #cpu_offload=True,
            #                        #mixed_precision=None,
            #                        #auto_wrap_policy=True,
            #                        #activation_checkpointing=True,
            #                        #sharding_strategy='FULL_SHARD',
            #                        #state_dict_type='full'
            # ),
            strategy="ddp",
            callbacks=callbacks,
            gradient_clip_val=0.25,# Not supported for manual optimization
            fast_dev_run=False,
    )


    from mapreader import AnnotationsLoader
    annotated_images = AnnotationsLoader()

    annotated_images.load("./annotations_one_inch/rail_space_#rw#.csv", images_dir="./patches_100_pixel")
    annotated_images.create_datasets(frac_train=0.7, frac_val=0.2, frac_test=0.1, context_datasets=True, context_df= "./patch_df.csv")

    dataloaders = annotated_images.create_dataloaders(batch_size=8,
                                                      prefetch_factor=3,
                                                      pin_memory=True,
                                                      num_workers=4,)
    labels_map = annotated_images.labels_map

    model=PytorchLightningClassifier(model=args.model,
                                     labels_map=labels_map,
                                     loss_fn=args.loss_fn,
                                     input_size=args.input_size,

                                     is_inception=args.is_inception,
                                     load_path=args.load_path,
                                     **args.kwargs)
    

    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]
    test_dataloader = dataloaders["test"]
    trainer.fit(model,train_dataloader=train_dataloader,val_dataloader=val_dataloader)
    trainer.test(model,test_dataloader=test_dataloader)



