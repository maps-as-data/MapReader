Load & Patchify
===============

.. note:: Run these commands in a Jupyter notebook (or other IDE), ensuring you are in your `mr_py38` python environment.

.. note:: You will need to update file paths to reflect your own machines directory structure.

MapReader's ``Load`` subpackage is used to load, visualise and patchify images (e.g. maps) saved locally. 

Load images (and metadata)
----------------------------

First, images (e.g. png, jpeg, tiff or geotiff files) can be loaded in using MapReader's ``loader()`` function. 

This can be done using: 

.. code-block:: python

    from mapreader import loader

    my_files = loader("./path/to/files/*.png")

or

This can be done using: 

.. code-block:: python

    from mapreader import loader

    my_files = loader("./path/to/files/", file_ext="png")

For example, if you have downloaded your maps using the default settings of our ``Download`` subpackage or have set up your directory as reccommended in our `Input Guidance <https://mapreader.readthedocs.io/en/latest/Input-guidance.html>`__:

.. code-block:: python

    #EXAMPLE
    my_files = loader("./maps/*.png")

or

.. code-block:: python

    #EXAMPLE 
    my_files = loader("./maps", file_ext="png")

The ``loader`` function creates a ``mapImages`` object (``my_files``) which contains information about your map images. 
To see the contents of this object, use: 

.. code-block:: python

    print(my_files)

You will see that your mapImages object contains the files you have loaded and that these are labelled as 'parents'. 

If your image files are georeferenced and already contain metadata (e.g. geoTIFFs), you can add this metadata into your ``mapImages`` object using:

.. code-block:: python

    my_files.addGeoInfo()

.. note:: This function will reproject your coordinates into "EPSG:4326". To change this specify ``proj2convert``.

Or, if you have a separate metadata file (e.g. a ``.csv`` file or a pandas dataframe), use: 

.. code-block:: python

    my_files.add_metadata(metadata="./path/to/metadata.csv")

.. note:: Specific guidance on preparing your metadata files can be found on our `Input Guidance <https://mapreader.readthedocs.io/en/latest/Input-guidance.html>`__ page.

For example, if you have downloaded your maps using the default settings of our ``Download`` subpackage or have set up your directory as reccommended in our `Input Guidance <https://mapreader.readthedocs.io/en/latest/Input-guidance.html>`__:

.. code-block:: python

    #EXAMPLE
    my_files.add_metadata(metadata="./maps/metadata.csv")


Patchify 
----------

Once you've loaded in all your data, you'll then need to `'patchify' <https://mapreader.readthedocs.io/en/latest/About.html>`__ your images.

Creating patches from your parent images is a core intellectual and technical task within MapReader. 
Choosing the size of your patches (and whether you want to measure them in pixels or in meters) is an important decision and will depend upon the research question you are trying to answer:

- Smaller patches (e.g. 50m x 50m) tend to work well on very large-scale maps (like the 25- or 6-inch Ordnance Survey maps of Britain).
- Larger patches (500m x 500m) will be better suited to slightly smaller-scale maps (for example, 1-inch Ordnance Survey maps).

In any case, the patch size you choose should roughly match the size of the visual feature(s) you want to label. 
Ideally your features should be smaller (in any dimension) than your patch size and therefore fully contained within a patch. 

To patchify your maps, use: 

.. code-block:: python

    my_files.patchifyAll()

By default, this slices images into 100 x 100 pixel patches which are saved in a newly created directory called ``./patches``. 
If you are following our reccommended directory structure, after patchifying, your directory should look like this:

::

    project
    ├──your_notebook.ipynb
    └──maps        
    │   ├── map1.png
    │   ├── map2.png
    │   ├── map3.png
    │   ├── ...
    │   └── metadata.csv
    └──patches
        ├── patch-0-100-#map1.png#.png
        ├── patch-100-200-#map1.png#.png
        ├── patch-200-300-#map1.png#.png
        └── ...

.. TODO: change default save name!

This save directory can be changed by specifying ``path_save``:

.. code-block:: python

    #EXAMPLE
    my_files.patchifyAll(path_save="./maps/patches")

This will create the following directory structure:

::

    project
    ├──your_notebook.ipynb
    └──maps        
        ├── map1.png
        ├── map2.png
        ├── map3.png
        ├── ...
        ├── metadata.csv
        └── patches
             ├── patch-0-100-#map1.png#.png
             ├── patch-100-200-#map1.png#.png
             ├── patch-200-300-#map1.png#.png
             └── ...


If you would like to change the size of your patches, you can specify ``patch_size``.

e.g. to slice your maps into 500 x 500 pixel patches:

.. code-block:: python

    #EXAMPLE
    my_files.patchifyAll(patch_size=500)

Or, if you have loaded geographic coordinates into your ``mapImages`` object, you can specify ``method = "meters"`` to slice your images by meters instead of pixels.

e.g. to slice your maps into 50 x 50 meter patches:

.. code-block:: python

    #EXAMPLE
    my_files.patchifyAll(method="meters", patch_size=50)

After patchifying, you'll see that ``print(my_files)`` shows you have both 'parents' and 'patches'.
To view an iterable list of these, you can use the ``.list_parents()`` and ``.list_patches()`` methods: 

.. code-block:: python

    parent_list = my_files.list_parents()
    patch_list = my_files.list_patches()

    print(parent_list)
    print(patch_list[0:5])  # too many to print them all!

Or, to view these in a dataframe, use:

.. code-block:: python

    parent_df, patch_df = my_files.convertImages()
    patch_df.head()

.. note:: Parent and patch dataframes **will not** automatically update so you may want to run this command again if you add new information into your ``mapImages`` object.

Visualise
----------

To view a random sample of your images, use: 

.. code-block:: python

    my_files.show_sample(num_samples=3)

.. image:: ../figures/show_sample_parent.png
    :width: 400px


By default, this will show you a random sample of your parent images.
To see a random sample of your patches use the ``tree_level="patch"`` argument: 

.. code-block:: python

    my_files.show_sample(num_samples=3, tree_level="patch")

.. image:: ../figures/show_sample_child.png
    :width: 400px


It can also be helpful to see your patches in the context of their parent image. 
To do this use the ``.show()`` method. 

e.g. :

.. code-block:: python

    #EXAMPLE
    my_files.show(patch_list[250:300])

.. image:: ../figures/show.png
    :width: 400px


or 

.. code-block:: python

    #EXAMPLE
    files_to_show = [patch_list[0], patch_list[350], patch_list[400]]
    my_files.show(files_to_show)

.. image:: ../figures/show_list.png
    :width: 400px


This will show you your chosen patches, by default highlighted with red borders, in the context of their parent image. 

You may also want to see all the patches created from one of your parent images.
This can be done using: 

.. code-block:: python

    my_files.show_par(parent_list[0])

.. image:: ../figures/show_par.png
    :width: 400px


Further analysis/visualisation  
--------------------------------

If you have loaded geographic coordinates into your ``mapImages`` object, you may want to calculate the coordinates of your patches. The ``.add_center_coord()`` method can used to do this:

.. code-block:: python

    my_files.add_center_coord()

    parent_df, patch_df = my_files.convertImages()
    patch_df.head()

After converting your images into dataframes, you will see that center coordinates have been added to your patch dataframe. 

The ``.calc_pixel_stats()`` method can be used to calculate means and standard deviations of pixel intensites of each of your patches:

.. code-block:: python

    my_files.calc_pixel_stats()

    parent_df, patch_df = my_files.convertImages()
    patch_df.head()

After converting your images into dataframes, you will see that mean and standard pixel intensities (R,G,B and, if present, Alpha) have been added to your patch dataframe. 

Specific values (e.g. 'mean_pixel_RGB') can be visualised using the ``.show()`` and ``.show_par()`` methods by specifying the ``value``, ``vmin`` and ``vmax`` arguments.

e.g. :

.. code-block:: python

    #EXAMPLE
    value = "mean_pixel_RGB"
    vmin = patch_df[value].min()
    vmax = patch_df[value].max()

    my_files.show_par(parent_list[0], value=value, vmin=vmin, vmax=vmax)

.. image:: ../figures/show_par_RGB.png
    :width: 400px

You may also want to specify the ``alpha`` argument, which sets the transparency of your plotted values. Lower ``alpha`` values allow you to see the parent image underneath.

e.g.:

.. code-block:: python

    #EXAMPLE
    my_files.show_par(parent_list[0], value=value, vmin=vmin, vmax=vmax, alpha=0.5)

.. image:: ../figures/show_par_RGB_0.5.png
    :width: 400px

To change the colormap used when plotting these values, you can also specify ``colorbar``.
This will accept any matplotlib colormap as an argument. 
