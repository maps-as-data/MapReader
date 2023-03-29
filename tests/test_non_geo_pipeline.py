from mapreader import classifier
from mapreader import loadAnnotations
from mapreader import patchTorchDataset

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import models


PATH2IMAGES = "./worked_examples/non-geospatial/classification_plant_phenotype/dataset/open_access_plant/*.png"
PATH2ANNOTS = "./worked_examples/non-geospatial/classification_plant_phenotype/annotations_phenotype_open_access/phenotype_test_#kasra#.csv"


def test_patchify():
    from mapreader import loader

    myimgs = loader(PATH2IMAGES)

    # len() shows the total number of images currently read (or sliced, see below)
    print(f"Number of images: {len(myimgs)}")

    # To get more information
    print(myimgs)

    all_imgs = myimgs.list_parents()
    assert len(all_imgs) == 2, "Expected 2 parents"

    # `method` can also be set to meters
    myimgs.patchifyAll(
        path_save="./dataset/eg_slice_50_50",
        patch_size=50,  # in pixels
        square_cuts=False,
        verbose=False,
        method="pixel",
    )

    # if parent_id="XXX", only compute pixel stats for that parent
    myimgs.calc_pixel_stats()

    imgs_pd, patches_pd = myimgs.convertImages()

    assert len(imgs_pd) == len(all_imgs), "Expected same number of images"


def test_load_annotation():
    annotated_images = loadAnnotations()

    annotated_images.load(PATH2ANNOTS, path2dir="./dataset/eg_slice_50_50")

    annotated_images.annotations.columns.tolist()

    print(annotated_images)

    # We need to shift these labels so that they start from 0:
    annotated_images.adjust_labels(shiftby=-1)

    # ### Split annotations into train/val or train/val/test
    # We use a stratified method for splitting the annotations, that is, each set contains approximately the same percentage of samples of each target label as the original set.
    annotated_images.split_annotations(frac_train=0.8, frac_val=0.2, frac_test=0.0)

    annotated_images.train["label"].value_counts()

    return annotated_images


def test_classifier():
    annotated_images = test_load_annotation()
    # # Classifier
    # ## Dataset
    # Define transformations to be applied to images before being used in training or validation/inference.
    # `patchTorchDataset` has some default transformations. However, it is possible to define your own transformations and pass them to `patchTorchDataset`:

    # ------------------
    # --- Transformation
    # ------------------

    # FOR INCEPTION
    # resize2 = 299
    # otherwise:
    resize2 = 224

    # mean and standard deviations of pixel intensities in
    # all the patches in 6", second edition maps
    normalize_mean = 1 - np.array([0.82860442, 0.82515008, 0.77019864])
    normalize_std = 1 - np.array([0.1025585, 0.10527616, 0.10039222])

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(resize2),
                transforms.RandomApply(
                    [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                    ],
                    p=0.5,
                ),
                #         transforms.RandomApply([
                #             transforms.GaussianBlur(21, sigma=(0.5, 5.0)),
                #             ], p=0.25),
                transforms.RandomApply(
                    [
                        # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                        transforms.Resize((50, 50)),
                    ],
                    p=0.25,
                ),
                #          transforms.RandomApply([
                #              transforms.RandomAffine(180, translate=None, scale=None, shear=20),
                #              ], p=0.25),
                transforms.Resize((resize2, resize2)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((resize2, resize2)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        ),
    }

    train_dataset = patchTorchDataset(
        annotated_images.train, transform=data_transforms["train"]
    )
    valid_dataset = patchTorchDataset(
        annotated_images.val, transform=data_transforms["val"]
    )
    # test_dataset  = patchTorchDataset(annotated_images.test,
    #                                   transform=data_transforms["val"])

    # ## Sampler

    # -----------
    # --- Sampler
    # -----------
    # We define a sampler as we have a highly imbalanced dataset
    label_counts_dict = annotated_images.train["label"].value_counts().to_dict()

    class_sample_count = []
    for i in range(0, len(label_counts_dict)):
        class_sample_count.append(label_counts_dict[i])

    weights = 1.0 / (torch.Tensor(class_sample_count) / 1.0)
    weights = weights.double()
    print(f"Weights: {weights}")

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights[train_dataset.patchframe["label"].to_list()],
        num_samples=len(train_dataset.patchframe),
    )

    valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights[valid_dataset.patchframe["label"].to_list()],
        num_samples=len(valid_dataset.patchframe),
    )

    # ## Dataloader
    myclassifier = classifier(device="default")
    # myclassifier.load("./checkpoint_12.pkl")

    batch_size = 8

    # Add training dataset
    myclassifier.add2dataloader(
        train_dataset,
        set_name="train",
        batch_size=batch_size,
        # shuffle can be False as annotations have already been shuffled
        shuffle=False,
        num_workers=0,
        sampler=train_sampler,
    )

    # Add validation dataset
    myclassifier.add2dataloader(
        valid_dataset,
        set_name="val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        # sampler=valid_sampler
    )

    myclassifier.print_classes_dl()

    # set class names for plots
    class_names = {0: "No", 1: "Plant"}
    myclassifier.set_classnames(class_names)

    myclassifier.print_classes_dl()

    myclassifier.batch_info()

    # ### Method 2: use `.initialize_model`
    myclassifier.del_model()

    myclassifier.initialize_model(
        "resnet18", pretrained=True, last_layer_num_classes="default", add_model=True
    )

    myclassifier.model_summary(only_trainable=False)

    list2optim = myclassifier.layerwise_lr(min_lr=1e-4, max_lr=1e-3)
    # #list2optim = myclassifier.layerwise_lr(min_lr=1e-4, max_lr=1e-3, ltype="geomspace")

    optim_param_dict = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False,
    }

    # --- if list2optim is defined, e.g., by using `.layerwise_lr` method (see the previous cell):
    myclassifier.initialize_optimizer(
        optim_type="adam",
        params2optim=list2optim,
        optim_param_dict=optim_param_dict,
        add_optim=True,
    )

    scheduler_param_dict = {
        "step_size": 10,
        "gamma": 0.1,
        "last_epoch": -1,
        #    "verbose": False
    }

    myclassifier.initialize_scheduler(
        scheduler_type="steplr",
        scheduler_param_dict=scheduler_param_dict,
        add_scheduler=True,
    )

    # Add criterion
    criterion = nn.CrossEntropyLoss()

    myclassifier.add_criterion(criterion)

    # ## Train/fine-tune a model
    myclassifier.train_component_summary()

    myclassifier.train(
        num_epochs=3,
        save_model_dir="./models_plant_open",
        tensorboard_path=False,
        verbosity_level=0,
        tmp_file_save_freq=2,
        remove_after_load=False,
        print_info_batch_freq=5,
    )

    # ### Plot results
    print(list(myclassifier.metrics.keys()))
