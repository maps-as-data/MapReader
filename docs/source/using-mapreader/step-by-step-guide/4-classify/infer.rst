Infer using a fine-tuned model
==================================

You can use any classifier (model) to predict labels on unannotated patches.

Initialize ``ClassifierContainer``
-------------------------------------

To initialize your ``ClassifierContainer`` for inference, you will need to define:

- ``model`` - The model (classifier) you would like to use.
- ``labels_map`` - A dictionary mapping your labels to their indices (e.g. ``{0: "no_railspace", 1: "railspace"}``). This labels map should be the same as that used when training/fine-tuning the classifier.
- ``device`` - The device you would like to use for inference (e.g. ``"cuda"``, ``"mps"`` or ``"cpu"``).

There are a number of options for the ``model`` argument:

    **1.  To load a locally-saved model, use ``torch.load`` to load your file and then pass this as the ``model`` argument.**

        If you have already trained a model using MapReader, your outputs, by default, should be saved in directory called ``models``.
        Within this directory will be ``checkpoint_X.pkl`` and ``model_checkpoint_X.pkl`` files.
        Your models are saved in the ``model_checkpoint_X.pkl`` files.

        e.g. To load one of these files:

        .. code-block:: python

            #EXAMPLE
            import torch
            from mapreader import ClassifierContainer

            my_model = torch.load("./models/model_checkpoint_6.pkl")
            labels_map = {0: "no_railspace", 1: "railspace"}

            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

            my_classifier = ClassifierContainer(my_model, labels_map, device=device)

        .. admonition:: Advanced usage
            :class: dropdown

            The ``checkpoint_X.pkl`` files contain all the information, except for your models (which is saved in the ``model_checkpoint_X.pkl`` files), you had previously loaded in to your ``ClassifierContainer``.
            If you have already trained a model using MapReader, you can use these files to reload your previously used ``ClassifierContainer``.

            To do this, set the ``model``, ``dataloaders`` and ``label_map`` arguments to ``None`` and pass ``load_path="./models/your_checkpoint_file.pkl"`` when initializing your ``ClassifierContainer``:

            .. code-block:: python

                #EXAMPLE
                my_classifier = ClassifierContainer(None, None, None, load_path="./models/checkpoint_6.pkl")

            This will also load the corresponding model file (in this case "./models/model_checkpoint_6.pkl").

            If you use this option, your optimizer, scheduler and loss function will be loaded from last time.

    **2.  To load a** `hugging face model <https://huggingface.co/models>`__\ **, choose your model, follow the "Use in Transformers" or "Use in timm" instructions to load it and then pass this as the ``model`` argument.**

        e.g. `This model <https://huggingface.co/davanstrien/autotrain-mapreader-5000-40830105612>`__ is based on our `*gold standard* dataset <https://huggingface.co/datasets/Livingwithmachines/MapReader_Data_SIGSPATIAL_2022>`__.
        It can be loaded using the `transformers <https://github.com/huggingface/transformers>`__ library:

        .. code-block:: python

            #EXAMPLE
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification
            from mapreader import ClassifierContainer

            extractor = AutoFeatureExtractor.from_pretrained("davanstrien/autotrain-mapreader-5000-40830105612")
            my_model = AutoModelForImageClassification.from_pretrained("davanstrien/autotrain-mapreader-5000-40830105612")
            labels_map = {0: "no_railspace", 1: "railspace"}

            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

            my_classifier = ClassifierContainer(my_model, labels_map, device=device)

        .. note:: You will need to install the `transformers <https://github.com/huggingface/transformers>`__ library to do this (``pip install transformers``).

        e.g. `This model <https://huggingface.co/timm/resnest101e.in1k>`__ is an example of one which uses the `timm <https://huggingface.co/docs/timm/index>`__ library.
        It can be loaded as follows:

        .. code-block:: python

            #EXAMPLE
            import timm
            import torch
            from mapreader import ClassifierContainer

            my_model = timm.create_model("hf_hub:timm/resnest101e.in1k", pretrained=True, num_classes=len(annotated_images.labels_map))
            labels_map = {0: "no_railspace", 1: "railspace"}

            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

            my_classifier = ClassifierContainer(my_model, labels_map, device=device)

        .. note:: You will need to install the `timm <https://huggingface.co/docs/timm/index>`__ library to do this (``pip install timm``).

Create dataset and add to ``my_classifier``
---------------------------------------------

You will then need to create a new dataset containing your unannotated patches.
This can be done by loading a dataframe containing the paths to your patches:

.. code-block:: python

    from mapreader import PatchDataset

    infer = PatchDataset("./patch_df.csv", delimiter=",", transform="test")

.. note:: You can create this CSV file using the ``convert_image(save=True)`` method on your ``MapImages`` object (follow instructions in the :doc:`Load </using-mapreader/step-by-step-guide/2-load>` user guidance). This could also be a GeoJSON file.

The ``transform`` argument is used to specify which `image transforms <https://pytorch.org/vision/stable/transforms.html>`__  to use on your patch images.
See :ref:`this section<transforms>` for more information on transforms.

You should then add this dataset to your ``ClassifierContainer`` (``my_classifier``\):

.. code-block:: python

    my_classifier.load_dataset(infer, set_name="infer")

This will create a ``DataLoader`` from your dataset and add it to your ``ClassifierContainer``\'s ``dataloaders`` attribute.

By default, the ``load_dataset`` method will create a dataloader with batch size of 16 and will not use a sampler.
You can change these by specifying the ``batch_size`` and ``sampler`` arguments respectively.
See :ref:`this section<sampler>` for more information on samplers.

Infer
------

After loading your dataset, you can then simply run the ``inference`` method to infer the labels on the patches in your dataset:

.. code-block:: python

    my_classifier.inference(set_name="infer")

As with the "test" dataset, to see a sample of your predictions, use:

.. code-block:: python

    my_classifier.show_inference_sample_results(label="railspace", set_name="infer")


Save predictions
~~~~~~~~~~~~~~~~~

To save your predictions, use the ``save_predictions`` method.
e.g. to save your predictions on the "infer" dataset:

.. code-block:: python

    my_classifier.save_predictions(set_name="infer")


Add predictions to metadata and save
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add your predictions to your patch metadata (saved in ``patch_df.csv``), you will need to load your predictions as metadata in the ``MapImages`` object.

To do this, you will need to create a new ``MapImages`` object and load in your patches and parent images:

.. code-block:: python

    from mapreader import load_patches

    my_maps = load_patches(patch_paths = "./path/to/patches/*png", parent_paths="./path/to/parents/*png")

You can then add your predictions to the metadata using the ``add_metadata`` method:

.. code-block:: python

    my_maps.add_metadata("path_to_predictions_patch_df.csv", tree_level='patch') # add dataframe as metadata

For example, to load the predictions for the "infer" dataset:

.. code-block:: python

    #EXAMPLE
    my_maps.add_metadata("./infer_predictions_patch_df.csv", tree_level='patch')

From here, you can use the ``show_patches`` method to visualize your predictions on the parent images as shown in the :doc:`Load </using-mapreader/step-by-step-guide/2-load>` user guide:

.. code-block:: python

    my_maps.add_shape()

    parent_list = my_maps.list_parents()
    my_maps.show_patches(
        parent_list[0],
        column_to_plot="conf",
        vmin=0,
        vmax=1,
        alpha=0.5
    )

Or, if your maps are georeferenced, you can use the ``explore_patches`` method instead:

.. code-block:: python

    my_maps.explore_patches(
        parent_list[0],
        column_to_plot="conf",
        xyz_url="https://geo.nls.uk/mapdata3/os/6inchfirst/{z}/{x}/{y}.png",
        vmin=0,
        vmax=1,
    )

Refer to the :doc:`Load </using-mapreader/step-by-step-guide/2-load>` user guidance for further details on how these methods work.
