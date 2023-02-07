Train
=====

.. contents::
    :local:

Read annotations
-----------------

Once you have annotated images, you can then use these to train/fine-tune a CV (Computer Vision) classifier.

First, load in your annotations using:

.. code :: python

    from mapreader import loadAnnotations
    
    annotated_images=loadAnnotations()
    annotated_images.load("./path/to/annotations.csv", path2dir='./path/to/images/')
    
e.g. 

.. code :: python 

    annotated_images=loadAnnotations()
    annotated_images.load("./annotations_one_inch/rail_space_#rw#.csv", path2dir='./maps/slice_50_50')

To view the data loaded in from your ``.csv``, use:

.. code :: python

    annotated_images.annotations

And, to view a summary of your annotations, use: 

.. code :: python

    print(annotated_images)

To align with python indexing, you may want to shift your labels so they start at 0. This can be done using:

.. code :: python

    annotated_images.adjust_label(shiftby=-1)

You can then view a sample of your annotated images using:

.. code :: python

    annotated_images.show_image_labels(tar_label=1)

.. image:: ../figures/show_image_labels_10.png
    :width: 400px

By default, this will show you 10 images but this can be changed by specifying ``num_sample``. 

You can also view specific images from their indices using:

.. code :: python

    annotated_images.show_image(indx=14)

.. image:: ../figures/show_image.png
    :width: 400px

Before training your CV classifier, you first need to split your annotated images into a 'train', 'validate' and 'test' sets.
MapReader uses a stratified method to do this, such that each set contains approximately the same percentage of samples of each target label as the original set.

To split your annotated images into dataframes, use: 

.. code :: python

    annotated_images.split_annotations()

By default, your annotated images will be split as follows:

    70% train
    15% validate
    15% test

However, these ratios can be changed by specifying ``frac_train``, ``frac_val`` and ``fract_test``.

e.g. : 

.. code :: python

    annotated_images.split_annotations(frac_train=0.5, frac_val=0.2, frac_test=0.3)

You can then check how many annotated images are in each set by checking the value counts of your dataframes:

.. code :: python

    annotated_images.train["label"].value_counts()
    annotated_images.val["label"].value_counts()
    annotated_images.test["label"].value_counts()

Load and prepare datasets
---------------------------

Before using your images in training, validation or inference, you will first want to define some transformations and prepare your data.
This can be done using the ``patchTorchDataset`` class. 

e.g. :

.. code :: python

    from mapreader import patchTorchDataset
    from torchvision import transforms
    
    resize=224
    normalize_mean = [0.485, 0.456, 0.406] # ImageNet means
    normalize_std = [0.229, 0.224, 0.225] # ImageNet stds

    data_transforms = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), transforms.Normalize(normalize_mean,normalize_std)])

    train_dataset = patchTorchDataset(annotated_images.train, data_transforms)
    val_dataset = patchTorchDataset(annotated_images.val, data_transforms)
    test_dataset = patchTorchDataset(annotated_images.test, data_transforms)

This produces three datasets (``train_dataset``, ``val_dataset`` and ``test_dataset``), ready for use, which can be viewed as dataframes using the ``patchframe`` method:

.. code :: python

    your_dataset.patchframe

Define a sampler
------------------

To account for inbalanced datasets, you may also want to define a sampler with weights inversely proportional to the number of instances of each label within a set. 
This ensures, when training and validating your model, each batch is ~ representative of the whole set.
To do this, use: 

.. code :: python

    import numpy as np
    import torch

    train_label_count = train_dataset.patchframe["label"].value_counts().to_list()
    val_label_count = val_dataset.patchframe["label"].value_counts().to_list()

    weights = np.reciprocal(torch.Tensor(sample_count))
    weights = weights.double()

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights[train_label_count], num_samples=len(train_dataset.patchframe))
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights[val_label_count], num_samples=len(val_dataset.patchframe))


Create batches (DataLoader)
----------------------------

``MapReader``'s ``classifier`` class is xxxxx.

.. code :: python

    from mapreader import classifier

    my_classifier = classifier()


To prepare your data for training, `PyTorch <https://pytorch.org/>`__ uses a ``DataLoader`` to create shuffled batches of data from each set. 
To load datasetsto your classifer, use: 

.. code :: python
    
    my_classifier.add2dataloader(your_dataset)

By default, your batch sizes will be set to 16 and no sampler will be used when creating them. 
This can be changed by specifying ``batch_size`` and ``sampler``.

e.g. :

.. code :: python

    batch_size=8

    my_classifier.add2dataloader(train_dataset, batch_size=batch_size, sampler=train_sampler)

You can also name your set using the ``set_name`` argument:

.. code :: python

    .. code :: python

    batch_size=8

    my_classifier.add2dataloader(train_dataset, sest_name="train", batch_size=batch_size, sampler=train_sampler)
    my_classifier.add2dataloader(val_dataset, set_name="val", batch_size=batch_size, sampler=val_sampler)

To see information about your datasets, batches and classes (labelled groups), use :

.. code :: python

    my_classifier.dataset_sizes

and 

.. code :: python 

    my_classifier.batch_info()

and 

.. code :: python
    
    my_classifier.print_classes_dl(set_name="train")
    my_classifier.print_classes_dl(set_name="val")

.. warning :: This only works if you have specified ``set_name`` when adding your datasets to the dataloader

You may also want to set ``class_names`` to help with human-readability. This is done by defining a dictionary mapping each label to a new name. 

e.g. :

.. code :: python

    class_names={0:"No", 1:"railspace"}
    my_classifier.set_classnames(class_names)
    my_classifier.print_classes_dl()

To see a sample batch, use the ``show_sample`` method:

.. code :: python

    my_classifier.show_sample()

.. image:: ../figures/show_sample_train_8.png
    :width: 400px

By default, this will show you the first batch created from your training datasest, along with corresponding batch information (``batch_info()``).
The ``batch_number`` and ``set_name``  arguments can be used to show different batches and datasets, respectively:

.. code :: python

    my_classifier.show_sample(set_name="val", batch_number=3)

.. image:: ../figures/show_sample_val_8.png
    :width: 400px

Load a PyTorch model
-----------------------

The `torchvision.models <https://pytorch.org/vision/stable/models.html>`__ subpackage contains a number of pre-trained models which can be loaded into your classifier.
These can be added in one of two ways:

    1.  Import a model directly from ``torchvision.models`` and add to your classifier using your classifiers ``.add_model`` method:

        .. code :: python

            from torchvision import models
            from torch import nn

            my_model=models.resnet18(pretrained=True)

            # reshape the final layer (FC layer) of the neural network to output the same number of nodes as classes as in your dataset
            num_input_features=my_model.fc.in_features
            my_model.fc = nn.Linear(num_input_features, my_classifier.num_classes)

            my_classifier.add_model(my_model)

        `See this tutorial for further details on fine-tuning torchvision models <https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html>`__

    2.  Using your classifiers ``.initialize_model`` method:

        .. code :: python
        
            my_classifier.initialize_model("resnet18")
    
        By default, this will initiliase a pretrained model and reshape the last layer to output the same number of nodes as classes in your dataset (as above). 
    
