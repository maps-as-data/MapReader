Load
=====

Patchify
-------------

MapReader's loader package is used to load, visualise and slice images (e.g. maps). 
Images can be loaded in using: 

.. code :: python

    from mapreader import loader
    my_files = loader("./maps/*png")

This creates a mapImages object (``my_files``) which contains information about your images. 
To see which images this contains, use: 

.. code :: python

    print(my_files)

You will see that your images are labelled as either parents or children.
This naming structure allows you to distinguish parent images (i.e. whole images) and child images (i.e. patches) as well as identify which parent image each patch has come from.

You can also load in additional information, such as metadata, using: 

.. code :: python

    my_files.add_metadata(metadata="./path/to/metadata.csv")

Once you've loaded in all your data, you'll then need to patchify your images.
This can be done using: 

.. code :: python

    my_files.sliceAll()

By default, this slices images into 100 x 100 pixel patches.
To change this, you can specify ``method`` and ``slice_size``. 

e.g : 

.. code :: python

    my_files.sliceAll(method="meters", slice_size=1)

By default, patches are saved in a newly created directory called ``./tests``. This can be changed by specifying ``path_save``.

e.g. :

.. code :: python

    my_files.sliceAll(path_save="./path/to/directory")

After patchifying, you'll see that ``print(my_files)`` shows you have both parents and children (patches).
To view an iterable list of these, you can use the ``.list_parents()`` and ``.list_children()`` methods: 

.. code :: python

    parent_list=my_files.list_parents()
    child_list=my_files.list_children()

    print(parent_list)
    print(child_list)

Or, to view these in a dataframe, use:

.. code :: python

    parent_df, patch_df = my_files.convertImages()
    patch_df

Visualise
-----------

To view a random sample of your images, use: 

.. code :: python

    my_files.show_sample(num_samples=3)

.. image:: ../figures/show_sample_parent.png
    :width: 400px


By default, this will show you a random sample of your parent images.
To see a random sample of your patches (child images) use the ``tree_level = "child"`` argument: 

.. code :: python

    my_files.show_sample(num_samples=3, tree_level="child")

.. image:: ../figures/show_sample_child.png
    :width: 400px


It can be helpful to see your patches (child images) in the context of their parent image. To do this use the ``.show()`` method. 

e.g. :

.. code :: python

    my_files.show(child_list[25:30])

.. image:: xxx
    :width: 400px


or 

.. code :: python

    files_to_show=[child_list[0], child_list[30], child_list[34]]
    my_files.show(files_to_show)

.. image:: xxx
    :width: 400px


This will show you your chosen patches, by default highlighted in red, in the context of their parent image. 

You may also want to see all the patches created from one of your parent images.
This can be done using: 

.. code :: python

    my_files.show_par(parent_list[0])

.. image:: ../figures/show_par.png
    :width: 400px


Calculate pixel intensities
------------------------------

The ``.calc_pixel_stats()`` method can be used to calculate means and standard deviations of pixel intensiites of each patch (child image) and parent image:

.. code :: python

    my_files.calc_pixel_stats()

This is useful for xxx.

To view your results in a dataframe, use the ``.convertImages()`` method (as above). 
Or, to visualise them, use the ``.show_par()`` method and specify the ``value``, ``vmin`` and ``vmax`` arguments.

e.g. :

.. code :: python

    value='mean_pixel_RGB'
    vmin=patch_df[value].min()
    vmax=patch_df[value].max()
    
    my_files.show_par(parent_list[0], value=value, vmin=vmin, vmax=vmax)

.. image:: ../figures/show_par_RGB.png
    :width: 400px


You may also want to specify the ``alpha`` argument, which sets the transparency of your plots and is by default set to 1. 
Lower ``alpha`` values allow you to see the parent image underneath:

.. code :: python

    my_files.show_par(parent_list[0], value=value, vmin=vmin, vmax=vmax, alpha=0.5)

.. image:: ../figures/show_par_RGB_0.5.png
    :width: 400px

