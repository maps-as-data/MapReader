.. container::

   ::

      <br>
      <p align="center">
      <h1>MapReader</h1>
      <h2>A computer vision pipeline for exploring and analyzing images at scale</h2>
      </p>

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

Contents
--------

.. toctree::
   :maxdepth: 2

`Gallery <#gallery>`__
`What is MapReader? <#what-is-mapreader>`__
`Overview <#overview>`__
`Use cases <#use-cases>`__
`How to contribute <#how-to-contribute>`__
`How to cite MapReader <#how-to-cite-mapreader>`__
`Credits and re-use terms <#credits-and-re-use-terms>`__

   `Digitized maps <#digitized-maps>`__: MapReader can retrieve maps
   from NLS via tileserver. Read the re-use terms in this section.
   `Metadata <#metadata>`__: the metadata files are stored at
   `mapreader/persistent_data <https://github.com/Living-with-machines/MapReader/tree/main/mapreader/persistent_data>`__.
   Read the re-use terms in this section.
   `Acknowledgements <#acknowledgements>`__


Gallery
-------

.. container::

   +-----------------------------------+-----------------------------------+
   | **classification_one              | **classification_p                |
   | _inch_maps_001**\ \ **Tutorial:** | lant_phenotype**\ \ **Tutorial:** |
   | train/fine-tune PyTorch CV        | train/fine-tune PyTorch CV        |
   | classifiers on historical maps    | classifiers on plant patches in   |
   | (Fig: rail infrastructure around  | images (plant phenotyping         |
   | London as predicted by a          | example).                         |
   | MapReader model).                 |                                   |
   +-----------------------------------+-----------------------------------+
   | **classi                          |                                   |
   | fication_mnist**\ \ **Tutorial:** |                                   |
   | train/fine-tune PyTorch CV        |                                   |
   | classifiers on whole MNIST images |                                   |
   | (not on patches/slices of those   |                                   |
   | images).                          |                                   |
   +-----------------------------------+-----------------------------------+
   |                                   |                                   |
   +-----------------------------------+-----------------------------------+

   **MapReader paper**\ 

What is MapReader?
------------------

MapReader is an end-to-end computer vision (CV) pipeline for exploring
and analyzing images at scale.

MapReader was developed in the `Living with
Machines <https://livingwithmachines.ac.uk/>`__ project to analyze large
collections of historical maps but is a **generalisable** computer
vision pipeline which can be applied to **any images** in a wide variety
of domains. See `Gallery <#gallery>`__ for some examples.

Refer to each tutorial/example in the `use cases <#use-cases>`__ section
for more details on MapReader’s relevant functionalities for
`non-geospatial <https://github.com/Living-with-machines/MapReader/tree/main/examples/non-geospatial>`__
and
`geospatial <https://github.com/Living-with-machines/MapReader/tree/main/examples/geospatial>`__
images.

Overview
--------

MapReader is a groundbreaking interdisciplinary tool that emerged from a
specific set of geospatial historical research questions. It was
inspired by methods in biomedical imaging and geographic information
science, which were adapted for annotation and use by historians, for
example in `JVC <https://doi.org/10.1093/jvcult/vcab009>`__ and
`MapReader <https://arxiv.org/abs/2111.15592>`__ papers. The success of
the tool subsequently generated interest from plant phenotype
researchers working with large image datasets, and so MapReader is an
example of cross-pollination between the humanities and the sciences
made possible by reproducible data science.

MapReader has two main components: preprocessing/annotation and
training/inference as shown in this figure:

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

It provides a set of tools to:

-  **load** images or maps stored locally or **retrieve** maps via
   web-servers (e.g., tileservers which can be used to retrieve maps
   from OpenStreetMap (OSM), the National Library of Scotland (NLS), or
   elsewhere). :warning: Refer to the `credits and re-use
   terms <#credits-and-re-use-terms>`__ section if you are using
   digitized maps or metadata provided by NLS.
-  **preprocess** images or maps (e.g., divide them into patches,
   resampling the images, removing borders outside the neatline or
   reprojecting the map).
-  annotate images or maps or their patches (i.e. slices of an image or
   map) using an **interactive annotation tool**.
-  **train, fine-tune, and evaluate** various CV models.
-  **predict** labels (i.e., model inference) on large sets of images or
   maps.
-  Other functionalities include:

   -  various **plotting tools** using, e.g., *matplotlib*, *cartopy*,
      *Google Earth*, and `kepler.gl <https://kepler.gl/>`__.
   -  compute mean/standard-deviation **pixel intensity** of image
      patches.

Installation
------------

Set up a conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installation via Anaconda (refer to `Anaconda website and
follow the
instructions <https://docs.anaconda.com/anaconda/install/>`__).

-  Create a new environment for ``mapreader`` called ``mr_py38``:

.. code:: bash

   conda create -n mr_py38 python=3.8

-  Activate the environment:

.. code:: bash

   conda activate mr_py38

Method 1
~~~~~~~~

-  Install ``mapreader``:

.. code:: bash

   pip install mapreader 

To work with geospatial images (e.g., maps):

.. code:: bash

   pip install "mapreader[geo]" 

-  We have provided some `Jupyter Notebooks to showcase MapReader’s
   functionalities <https://github.com/Living-with-machines/MapReader/tree/main/examples>`__.
   To allow the newly created ``mr_py38`` environment to show up in the
   notebooks:

.. code:: bash

   python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"

-  Continue with the examples in `Use cases <#use-cases>`__!

-  ⚠️ On *Windows* and for *geospatial images* (e.g., maps), you might
   need to do:

.. code:: bash

   # activate the environment
   conda activate mr_py38

   # install rasterio and fiona manually
   conda install -c conda-forge rasterio=1.2.10
   conda install -c conda-forge fiona=1.8.20

   # install git
   conda install git

   # install MapReader
   pip install git+https://github.com/Living-with-machines/MapReader.git

   # open Jupyter Notebook (if you want to test/work with the notebooks in "examples" directory)
   cd /path/to/MapReader 
   jupyter notebook

Method 2
~~~~~~~~

-  Clone ``mapreader`` source code:

.. code:: bash

   git clone https://github.com/Living-with-machines/MapReader.git 

-  Install:

.. code:: bash

   cd /path/to/MapReader
   pip install -v -e .

To work with geospatial images (e.g., maps):

.. code:: bash

   cd /path/to/MapReader
   pip install -e ."[geo]"

-  We have provided some `Jupyter Notebooks to showcase MapReader’s
   functionalities <https://github.com/Living-with-machines/MapReader/tree/main/examples>`__.
   To allow the newly created ``mr_py38`` environment to show up in the
   notebooks:

.. code:: bash

   python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"

-  Continue with the examples in `Use cases <#use-cases>`__!

Use cases
---------

`Tutorials <https://github.com/Living-with-machines/MapReader/tree/main/examples>`__
are organized in Jupyter Notebooks. Follow the hyperlinks on input type
names (“Non-Geospatial” or “Geospatial”) to read guidance specific to
those image types.

-  `Non-Geospatial <https://github.com/Living-with-machines/MapReader/tree/main/examples/non-geospatial>`__:

   -  `classification_plant_phenotype <https://github.com/Living-with-machines/MapReader/tree/main/examples/non-geospatial/classification_plant_phenotype>`__

      -  **Goal:** train/fine-tune PyTorch CV classifiers on plant
         patches in images (plant phenotyping example).
      -  **Dataset:** Example images taken from the openly accessible
         ``CVPPP2014_LSV_training_data`` dataset available from
         https://www.plant-phenotyping.org/datasets-download.
      -  **Data access:** locally stored
      -  **Annotations** are done on plant patches (i.e., slices of each
         plant image).
      -  **Classifier:** train/fine-tuned PyTorch CV models.

   -  `classification_mnist <https://github.com/Living-with-machines/MapReader/tree/main/examples/non-geospatial/classification_mnist>`__

      -  **Goal:** train/fine-tune PyTorch CV classifiers on MNIST.
      -  **Dataset:** Example images taken from
         http://yann.lecun.com/exdb/mnist/.
      -  **Data access:** locally stored
      -  **Annotations** are done on whole MNIST images, **not** on
         patches/slices of those images.
      -  **Classifier:** train/fine-tuned PyTorch CV models.

-  `Geospatial <https://github.com/Living-with-machines/MapReader/tree/main/examples/geospatial>`__:

   -  Maps:

      -  `classification_one_inch_maps_001 <https://github.com/Living-with-machines/MapReader/tree/main/examples/geospatial/classification_one_inch_maps_001>`__

         -  **Goal:** train/fine-tune PyTorch CV classifiers on
            historical maps.
         -  **Dataset:** from National Library of Scotland: `OS
            one-inch, 2nd edition
            layer <https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/index.html>`__.
         -  **Data access:** tileserver
         -  **Annotations** are done on map patches (i.e., slices of
            each map).
         -  **Classifier:** train/fine-tuned PyTorch CV models.

How to contribute
-----------------

We welcome contributions related to new applications, both with
geospatial images (other maps, remote sensing data, aerial photography)
and non-geospatial images (for example, other scientific image
datasets).

How to cite MapReader
---------------------

Please consider acknowledging MapReader if it helps you to obtain
results and figures for publications or presentations, by citing:

Link: https://dl.acm.org/doi/10.1145/3557919.3565812

.. code:: text

   Kasra Hosseini, Daniel C. S. Wilson, Kaspar Beelen, and Katherine McDonough. 2022. MapReader: a computer vision pipeline for the semantic exploration of maps at scale. In Proceedings of the 6th ACM SIGSPATIAL International Workshop on Geospatial Humanities (GeoHumanities '22). Association for Computing Machinery, New York, NY, USA, 8–19. https://doi.org/10.1145/3557919.3565812

and in BibTeX:

.. code:: bibtex

   @inproceedings{10.1145/3557919.3565812,
   author = {Hosseini, Kasra and Wilson, Daniel C. S. and Beelen, Kaspar and McDonough, Katherine},
   title = {MapReader: A Computer Vision Pipeline for the Semantic Exploration of Maps at Scale},
   year = {2022},
   isbn = {9781450395335},
   publisher = {Association for Computing Machinery},
   address = {New York, NY, USA},
   url = {https://doi.org/10.1145/3557919.3565812},
   doi = {10.1145/3557919.3565812},
   booktitle = {Proceedings of the 6th ACM SIGSPATIAL International Workshop on Geospatial Humanities},
   pages = {8–19},
   numpages = {12},
   keywords = {supervised learning, historical maps, deep learning, digital libraries and archives, computer vision, classification},
   location = {Seattle, Washington},
   series = {GeoHumanities '22}
   }

Credits and re-use terms
------------------------

Digitized maps
~~~~~~~~~~~~~~

MapReader can retrieve maps from NLS (National Library of Scotland) via
webservers. For all the digitized maps (retrieved or locally stored),
please note the re-use terms:

:warning: Use of the digitised maps for commercial purposes is currently
restricted by contract. Use of these digitised maps for non-commercial
purposes is permitted under the `Creative Commons
Attribution-NonCommercial-ShareAlike 4.0
International <https://creativecommons.org/licenses/by-nc-sa/4.0/>`__
(CC-BY-NC-SA) licence. Please refer to
https://maps.nls.uk/copyright.html#exceptions-os for details on
copyright and re-use license.

Metadata
~~~~~~~~

We have provided some metadata files in ``mapreader/persistent_data``.
For all these file, please note the re-use terms:

:warning: Use of the metadata for commercial purposes is currently
restricted by contract. Use of this metadata for non-commercial purposes
is permitted under the `Creative Commons
Attribution-NonCommercial-ShareAlike 4.0
International <https://creativecommons.org/licenses/by-nc-sa/4.0/>`__
(CC-BY-NC-SA) licence. Please refer to
https://maps.nls.uk/copyright.html#exceptions-os for details on
copyright and re-use license.

Acknowledgements
~~~~~~~~~~~~~~~~

This work was supported by Living with Machines (AHRC grant
AH/S01179X/1) and The Alan Turing Institute (EPSRC grant EP/N510129/1).
Living with Machines, funded by the UK Research and Innovation (UKRI)
Strategic Priority Fund, is a multidisciplinary collaboration delivered
by the Arts and Humanities Research Council (AHRC), with The Alan Turing
Institute, the British Library and the Universities of Cambridge, East
Anglia, Exeter, and Queen Mary University of London.
