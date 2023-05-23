Classify
=========

.. note:: Run these commands in a Jupyter notebook (or other IDE), ensuring you are in your `mapreader` python environment.

.. note:: You will need to update file paths to reflect your own machines directory structure.

MapReader's ``Classify`` subpackage is used to train or fine-tune a CV (computer vision) model and use it to predict the labels of patches.

If you are new to computer vision/ machine learning, `see this tutorial for details on fine-tuning torchvision models <https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html>`__.
This will help you get to grips with the basic steps needed to train/fine-tune a model.

Load annotations and prepare data
-----------------------------------

Load and check annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load in your annotations using:

.. code-block:: python

    from mapreader import AnnotationsLoader

    annotated_images = AnnotationsLoader()
    annotated_images.load(annotations = "./path/to/annotations.csv")

For example, if you have set up your directory as reccommended in our `Input Guidance <https://mapreader.readthedocs.io/en/latest/Input-guidance.html>`__, and then saved your patches and annotations using the default settings:

.. code-block:: python

    #EXAMPLE
    annotated_images = AnnotationsLoader()
    annotated_images.load("./annotations/rail_space_#rosie#.csv")

.. admonition:: Advanced usage
    :class: dropdown

    Other parameters you may want to specify when adding metadata to your images include:

    - ``delimiter`` - By default, this is set to "\t" so will assume your ``csv`` file is tab delimited. You will need to specify the ``delimiter`` argument if your file is saved in another format.
    - ``id_col``, ``patch_paths_col``, ``label_col`` - These are used to indicate the column headings for the columns which contain image IDs, patch file paths and labels respectively. By default, these are set to "image_id", "image_path" and "label".
    - ``append`` - By default, this is ``False`` and so each call to the ``.load()`` method will overwrite existing annotations. If you would like to load a second ``csv`` file into your ``AnnotationsLoader`` instance you will need to set ``append=True``. 

To view the data loaded in from your ``csv`` as a dataframe, use:

.. code-block:: python

    annotated_images.annotations

You will note a ``label_index`` column has been added to your dataframe. 
This column contains a numerical reference number for each label, which is needed when training your model.

To see how your labels map to their label indices, call the ``annotated_images.labels_map`` attribute:

.. code-block:: python

    annotated_images.labels_map

To view a sample of your annotated images use the ``show_sample()`` method.
The ``label_to_show`` argument specifies which label you would like to show. 

For example, to show your "rail_space" label:

.. code-block:: python

    #EXAMPLE
    annotated_images.show_image_labels("rail_space")

.. image:: ../figures/show_image_labels_10.png
    :width: 400px


By default, this will show you a sample of 10 images, but this can be changed by specifying ``num_sample``. 

When viewing your annotations, you may notice that you have mislabelled one of your images.
The ``.review_labels()`` method, which returns an interactive tool for adjusting your annotations, provides an easy way to fix this:

.. code-block:: python

    annotated_images.review_labels()

.. image:: ../figures/review_labels.png
    :width: 400px


.. note:: To exit, type "exit", "end", or "stop" into the text box.

Prepare datasets and dataloaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo:: "Most neural networks expect the images of a fixed size. Therefore, we will need to write some preprocessing code." Add note about this is why we resize and also comment on square images.

Before using your annotated images to train your model, you will first need to:

.. _ratios:

1.  **Split your annotated images into "train", "val" and and, optionally, "test" `datasets <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`__.**

    By default, when creating your "train", "val" and "test" datasets, MapReader will split your annotated images as follows:

    - 70% train
    - 15% validate
    - 15% test

    This is done using a stratified method, such that each dataset contains approximately the same proportions of each target label.

    .. admonition:: Stratified example
        :class: dropdown
        
        If you have twenty annotated images:

        - labels: ``"a","a","b","a","a","b","a","a","a","a","a","b","a","a","a","b","b","a","b","a"`` (14 ``"a"``\s and 6 ``"b"``\s)
        
        Your train, test and val datasets will contain:

        - train labels: ``"a","a","b","a","a","a","a","a","b","a","a","a","b","b"`` (10 ``"a"``\s and 4 ``"b"``\s)
        - val labels: ``"a","b","a"`` (2 ``"a"``\s and 1 ``"b"``)
        - test labels: ``"a","a","b"`` (2 ``"a"``\s and 1 ``"b"``)

.. _transforms:

1.  **Define some `transforms <https://pytorch.org/vision/stable/transforms.html>`_ which will be applied to your images to ensure your they are in the right format.**
    
    Some default image transforms, generated using `torchvision's transforms module <https://pytorch.org/vision/stable/transforms.html>`_, are predefined in the ``PatchDataset`` class.
    
    .. admonition:: See default transforms
        :class: dropdown
        
        **default transforms for training dataset**
        
        .. code-block:: python
            
            transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.RandomApply([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(normalize_mean, normalize_std),
                ]
            )
            
        **default transforms for val and test datasets**
        
        .. code-block:: python
            
            transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(normalize_mean, normalize_std),
                ]
            )
    
    You can access these by calling the ``.transform`` attribute on any dataset or from the ``PatchDataset`` API documentation.

.. _sampler:

1.  **Create `dataloaders <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`__ which can be used to load small batches of your dataset during training/inference and apply the transforms to each image in the batch.**

    In many cases, you will want to create batches which are approximately representative of your whole dataset.
    This requires a `sampler <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`__ with weights inversely proportional to the number of instances of each label within each dataset.

    By default, MapReader creates a sampler with weights inversely proportional to the number of instances of each label within the "train" dataset.
    
    .. admonition:: Sampler example
        :class: dropdown

        If you have fourteen images in your train dataset:

        - train labels: ``"a","a","b","a","a","a","a","a","b","a","a","a","b","b"`` (10 ``"a"``\s and 4 ``"b"``\s)

        The weights for your sampler will be:

        - ``"a"`` weights: 1/10 (one in ten chance of picking an ``"a"`` when creating a batch)
        - ``"b"`` weights: 1/4 (one in four chance of picking an ``"b"`` when creating a batch)
    
    Using a sampler to create representative batches is particularly important for inbalanced datasets (i.e. those which contain different numbers of each label). 

To split your annotated images and create your dataloaders, use: 

.. code-block:: python

    dataloaders = annotated_images.create_dataloaders()

By default, this will split your annotated images using the :ref:`default train:val:test ratios<ratios>` and apply the :ref:`default image transforms<transforms>` to each by calling the ``.create_datasets()`` method.
It will then create a dataloader for each dataset, using a batch size of 16 and the :ref:`default sampler<sampler>`.

To change the ratios used to split your annotations, you can specify ``frac_train``, ``frac_val`` and ``frac_test``:abbr:

.. code-block:: python

    #EXAMPLE
    dataloaders = annotated_images.create_dataloaders(frac_train=0.6, frac_val=0.3, frac_test=0.1)

This will result in a split of 60% (train), 30% (val) and 10% (test).

To change the batch size used when creating your dataloaders, use the ``batch_size`` argument:

.. code-block:: python

    #EXAMPLE
    dataloaders = annotated_images.create_dataloaders(batch_size=24)

.. admonition:: Advanced usage
    :class: dropdown

    Other parameters you may want to specify when adding metadata to your images include:

    - ``sampler`` - By default, this is set to ``default`` and so the :ref:`default sampler<sampler>` will be used when creating your dataloaders and batches. You can choose not to use a sampler by specifying ``sampler=None`` or, you can define a custom sampler using `pytorch's sampler class <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`__.
    - ``shuffle`` - If your datasets are ordered (e.g. ``"a","a","a","a","b","c"``), you can use ``shuffle=True`` to create dataloaders which contain shuffled batches of data. This cannot be used in conjunction with a sampler and so, by default, ``shuffle=False``. 
    - ``train_transform``, ``val_transform`` and ``test_transform`` - By default, these are set to "train", "val" and "test" respectively and so the :ref:`default image transforms<transforms>` for each of these sets are applied to the images. You can define your own transforms, using  `torchvision's transforms module <https://pytorch.org/vision/stable/transforms.html>`__, and apply these to your datasets by specifying the ``train_transform``, ``val_transform`` and ``test_transform`` arguments. 


Train
------

Initialise ``ClassifierContainer()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MapReader's ``ClassifierContainer()`` class is used to:

- Load models.
- Load dataloaders and labels map.
- Define a loss function (criterion), optimiser and scheduler.
- Train and evaluate models using already annotated images.
- Predict labels of un-annotated images (model inference).
- Visualise datasets and predictions.

You can initialise a ``ClassifierContainer()`` object (``my_classifier``) using:

.. code-block:: python

    from mapreader import ClassiferContainer

    my_classifier = ClassiferContainer(model, dataloaders, labels_map)

Your dataloaders and labels map (``annotated_images.labels_map``) should be passed as the ``dataloaders`` and ``labels_map`` arguments respectively.

There are a number of options for the ``model`` argument:

    1.  To load a model from `torchvision.models <https://pytorch.org/vision/stable/models.html>`__, pass one of the model names as the ``model`` argument.

        e.g. To load "resnet18":

        .. code-block:: python
        
            #EXAMPLE
            my_classifier = ClassiferContainer("resnet18", dataloaders, annotated_images.labels_map)

        By default, this will load a pretrained form of the model and reshape the last layer to output the same number of nodes as labels in your dataset.
        You can load an untrained model by specifying ``pretrained=False``.

    2.  To load a customised model, define a `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module>`__ and pass this as the ``model`` argument.
        
        e.g. To load a pretrained "resnet18" and reshape the last layer:

        .. code-block:: python

            #EXAMPLE
            from torchvision import models
            from torch import nn

            my_model = models.resnet18(pretrained=True)

            # reshape the final layer (FC layer) of the neural network to output the same number of nodes as label in your dataset
            num_input_features = my_model.fc.in_features
            my_model.fc = nn.Linear(num_input_features, len(annotated_images.labels_map))

            my_classifier = ClassifierContainer(my_model, dataloaders, annotated_images.labels_map)

        This is equivalent to passing ``model="resnet18"`` (as above) but further customisations are, of course, possible. 
        See `here <https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>`__ for more details of how to do this.

    3.  To load a locally-saved model, use ``torch.load()`` to load your file and then pass this as the ``model`` argument.

        If you have already trained a model using MapReader, your outputs, by default, should be saved in directory called ``models``.
        Within this directory will be ``checkpoint_X.pkl`` and ``model_checkpoint_X.pkl`` files.
        Your models are saved in the ``model_checkpoint_X.pkl`` files.

        e.g. To load one of these files:

        .. code-block:: python

            #EXAMPLE
            import torch

            my_model = torch.load("./models/model_checkpoint_6.pkl")

            my_classifier = ClassifierContainer(my_model, dataloaders, annotated_images.labels_map)

        .. admonition:: Advanced usage
            :class: dropdown
        
            The ``checkpoint_X.pkl`` files contain all the information, except for your models (which is saved in the ``model_checkpoint_X.pkl`` files), you had previously loaded in to your ``ClassifierContainer()``.
            If you have already trained a model using MapReader, you can use these files to reload your previously used ``ClassifierContainer()``.
        
            To do this, set the ``model``, ``dataloaders`` and ``label_map`` arguments to ``None`` and pass ``load_path="./models/your_checkpoint_file.pkl"`` when initialising your ``ClassifierContainer()``:
        
            .. code-block:: python
            
                #EXAMPLE
                my_classifier = ClassifierContainer(None, None, None, load_path="./models/checkpoint_6.pkl")        
            
            This will also load the corresponding model file (in this case "./models/model_checkpoint_6.pkl").

            If you use this option, your optimizer, scheduler and criterion will be loaded from last time.       

    4.  To load a `hugging face model <https://huggingface.co/models>`__, choose your model, follow the "Use in Transformers" instructions to load it and then pass this as the ``model`` argument.

        e.g. `This model <https://huggingface.co/davanstrien/autotrain-mapreader-5000-40830105612>`__ is based on our `*gold standard* dataset <https://huggingface.co/datasets/Livingwithmachines/MapReader_Data_SIGSPATIAL_2022>`__. To load it:

        .. code-block:: python

            #EXAMPLE
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification

            extractor = AutoFeatureExtractor.from_pretrained("davanstrien/autotrain-mapreader-5000-40830105612")
            my_model = AutoModelForImageClassification.from_pretrained("davanstrien/autotrain-mapreader-5000-40830105612")

            my_classifier = ClassifierContainer(my_model, dataloaders, annotated_images.labels_map) 

Define optimizer, scheduler and criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to train/fine-tune your model, will need to define:

1.  **A criterion ("loss function") - This works out how well your model is performing (the "loss").**

    To add a criterion, use ``.add_criterion()``.
    
    .. code-block:: python
    
        #EXAMPLE
        my_classifier.add_criterion("cross-entropy")
    
    In this example, we have used `PyTorch's cross-entropy loss function <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_ as our criterion. 
    You should change this to suit your needs.
    
    .. admonition:: Advanced usage
        :class: dropdown
    
        The ``add_criterion()`` method accepts any of "cross-entropy", "binary cross-entropy" and "mean squared error" as its ``criterion`` argument. 
        
        However, if you would like to use a different loss function, you can pass any `torch.nn loss function <https://pytorch.org/docs/stable/nn.html#loss-functions>`__ as the ``criterion`` argument.
    
        e.g. to use the mean absolute error as your loss function:
    
        .. code-block:: python
        
            from torch import nn
    
            criterion = nn.L1Loss()
            my_classifier.add_criterion(criterion)

2.  **An optimizer - This works out how much to adjust your model parameters by after each training cycle ("epoch").**

    The ``.initialize_optimizer()`` method is used to add an optimiser to you ``ClassifierContainer()`` (``my_classifier``):

    .. code-block:: python

        my_classifier.initialize_model()

    The ``optim_type`` argument can be used to select the `optimisation algorithm <https://pytorch.org/docs/stable/optim.html#algorithms>`__.
    By default, this is set to `"adam" <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam>`__, one of the  most commonly used algorithms.
    You should change this to suit your needs. 

    The ``params2optimise`` argument can be used to select which parameters to optimise during training.
    By default, this is set to ``"infer"``, meaning that all trainable parameters will be optimised.

    When training/fine-tuning your model, you can either use one learning rate for all layers in your neural network or define layerwise learning rates (i.e. different learning rates for each layer in your neural network). 
    Normally, when fine-tuning pre-trained models, layerwise learning rates are favoured, with smaller learning rates assigned to the first layers and larger learning rates assigned to later layers.

    To define a list of parameters to optimise within each layer, with learning rates defined for each parameter, use:

    .. code-block:: python 

        #EXAMPLE
        params2optimise = my_classifier.generate_layerwise_lrs(min_lr=1e-4, max_lr=1e-3)

    By default, a linear function is used to distribute the learning rates (using ``min_lr`` for the first layer and ``max_lr`` for the last layer). 
    This can be changed to a logarithmic function by specifying ``spacing="geomspace"``:

    .. code-block:: python 

        #EXAMPLE
        params2optimise = my_classifier.generate_layerwise_lrs(min_lr=1e-4, max_lr=1e-3, "geomspace")

    You should then pass your ``params2optimise`` list to the ``.initialize_optimizer()`` method:

    .. code-block:: python

        my_classifier.initialize_optimizer(params2optimise=params2optimise)

3.  **A scheduler - This defines how to adjust your learning rates during training.**

    To add a scheduler, use the ``.initialize_scheduler()`` method:
    
    .. code-block:: python

        my_classifier.initialize_scheduler()

    .. admonition:: Advanced usage
        :class: dropdown

        By default, your scheduler be set up to decrease your learning rates by 10% every 10 epochs. 
        These numbers can be adjusted by specifying the ``scheduler_param_dict`` argument.

        e.g. To reduce your learning rates by 2% every 5 epochs:

        .. code-block:: python

            #EXAMPLE
            my_classifier.initialize_scheduler(scheduler_param_dict= {'step_size': 5, 'gamma': 0.02})

Train/fine-tune your model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To begin training/fine-tuning your model, use:

.. code-block:: python

    my_classifier.train()

By default, this will run 25 epochs of training and validating your model and save your model in a newly created ``./models`` directory.
The ``num_epochs`` and ``save_model_dir`` arguments can be specified to change these.

e.g. to run 10 epochs of training and save your model in a newly created ``my_models_directory``:

.. code-block:: python

    #EXAMPLE
    my_classifier.train(num_epochs=10, save_model_dir="./my_models_directory")

Other arguments you may want to specify when training your model include:

- ``phases``: phases to perform at each epoch
- ``tensorboard_path``: directory to save tensorboard files
- ``verbosity_level``: -1 (quiet), 0 (normal), 1 (verbose), 2 (very verbose), 3 (debug)

Plot metrics
^^^^^^^^^^^^^^

Metrics are stored in a dictionary accesible via your ``classifier()`` objects ``.metrics`` attribute. 
To list these metrics, use:

.. code-block:: python

    list(myclassifier.metrics.keys())

To view specific metrics from training/validating, use:

.. code-block:: python

    my_classifier.metrics["metric_to_view"]

e.g. :

.. code-block:: python

    #EXAMPLE
    my_classifier.metrics["epoch_fscore_micro_train"]

Or, to help visualise the progress of your training, metrics can be plotted using ``.plot_metric()``: 

.. code-block:: python

    #EXAMPLE
    my_classifier.plot_metric(
        y_axis=["epoch_loss_train", "epoch_loss_val"],
        y_label="Loss",
        legends=["Train", "Valid"],
    )

.. image:: ../figures/loss.png
    :width: 400px


Inference 
-----------

Finally, to use your model for inference, use:

.. code-block:: python

    my_classifier.inference(set_name="your_dataset_name")

e.g. to run the trained model on the 'test' dataset, use:

.. code-block:: python

    #EXAMPLE
    my_classifier.inference(set_name="test")

By default, metrics will not be calculated or added to the ``.metrics`` dictionary during inference.
So, to add these in so that they can be viewed and plotted, use ``.calculate_add_metrics()``. 

e.g. to add metrics for the 'test' dataset: 

.. code-block:: python

    #EXAMPLE
    my_classifier.calculate_add_metrics(
        y_true=my_classifier.orig_label,
        y_pred=my_classifier.pred_label,
        y_score=my_classifier.pred_conf,
        phase="test",
    )

Metrics from this inference can then be viewed as above. 

To see a sample of your inference results, use: 

.. code-block:: python

    my_classifier.inference_sample_results(set_name="your_dataset_name")

e.g. :

.. code-block:: python

    #EXAMPLE
    my_classifier.inference_sample_results(set_name="test")

.. image:: ../figures/inference_sample_results.png
    :width: 400px


By default, this will show you 6 samples of your first class (label). 
The ``num_samples`` and ``class_index`` arguments can be specified to change this.

You may also want specify the minimum (and maximum) prediction confidence for your samples. 
This can be done using ``min_conf`` and ``max_conf``.

e.g. :

.. code-block:: python

    #EXAMPLE
    my_classifier.inference_sample_results(
        set_name="test", num_samples=3, class_index=1, min_conf=80
    )
