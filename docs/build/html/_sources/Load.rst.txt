Load
====

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

    my_files.add_metadata(metadata="./path/to/metadata")

Once you've loaded in all your data, you'll then need to patchify your images.
This can be done using: 

.. code :: python

    my_files.sliceAll()

By default, this slices images into 100 x 100 pixel patches.
To change this, you can specify ``method`` and ``slice_size``: 

.. code :: python

    my_files.sliceAll(method="meters", slice_size=1)

By default, patches are saved in a newly created directory called ``./tests``.
This can be changed by specifying ``path_save``: 

.. code :: python

    my_files.sliceAll(path_save="./path/to/directory")

After patchifying, you'll see that when you ``print(my_files)`` you have both parents and children.

To view this in an iterable list, you can use the ``.list_xxx()`` methods: 

.. code :: python

    parent_list=my_files.list_parents()
    child_list=my_files.list_children()

To view a random sample of your images, use: 

.. code :: python

    my_files.show_sample(num_samples=3)

.. image:: show_sample_parent.png
    :width: 400px

By default, this will show you a random sample of your parent images.
To see a random sample of your patches (child images) specify ``tree_level = "child"``: 

.. code :: python

    my_files.show_sample(num_samples=3, tree_level="child")

.. image:: ./show_sample_child.png
    :width: 400px

You may also want to see all the patches created from one of your parent images.
This can be done using: 

.. code :: python

    my_files.show_par(parent_list[0])

.. image:: ./show_par.png
    :width: 400px

.. Load package also contains some analysis bits, should these be on this page?
   Maybe load also wants renaming as it seems to do much more than load. (i.e. something that encompasses load, patchify and visualise) -RW 
