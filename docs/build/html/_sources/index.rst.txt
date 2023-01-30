.. MapReader documentation master file, created by
   sphinx-quickstart on Wed Jan 25 11:03:05 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=========
MapReader
=========
-----------------------------------------------------------------------
A computer vision pipeline for exploring and analyzing images at scale
-----------------------------------------------------------------------

Welcome to MapReader's documentation!
=====================================

.. toctree::
   :maxdepth: 1

   Install
   User-guide
   api/index
   MapReader Paper <https://dl.acm.org/doi/10.1145/3557919.3565812>

Gallery
-------

.. list-table::
   :widths: 50 50
   :header-rows: 0
   :stub-columns: 0

   * - **classification_one_inch_maps_001** 
         .. image:: https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/tutorial_classification_one_inch_maps_001.png
            :width: 200px
     - **classification_plant_phenotype**
         .. image:: https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/tutorial_classification_plant_phenotype.png
            :width: 200px
   * - **classification_mnist**
         .. image:: https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/tutorial_classification_mnist.png
            :width: 200px
     - **MapReader paper**
         .. image:: https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/mapreader_paper.png
            :width: 200px


What is MapReader?
------------------

MapReader is an end-to-end computer vision (CV) pipeline for exploring and analyzing images at scale.

MapReader was developed in the `Living with Machines <https://livingwithmachines.ac.uk/>`__ project to analyze large collections of historical maps but is a **generalisable** computer vision pipeline which can be applied to **any images** in a wide variety of domains. See `Gallery <#gallery>`__ for some examples.

Refer to each tutorial/example in the `use cases <#use-cases>`__ section for more details on MapReader’s relevant functionalities for `non-geospatial <https://github.com/Living-with-machines/MapReader/tree/main/examples/non-geospatial>`__ and `geospatial <https://github.com/Living-with-machines/MapReader/tree/main/examples/geospatial>`__ images.

Overview
--------

MapReader is a groundbreaking interdisciplinary tool that emerged from a specific set of geospatial historical research questions. It was inspired by methods in biomedical imaging and geographic information science, which were adapted for annotation and use by historians, for example in `JVC <https://doi.org/10.1093/jvcult/vcab009>`__ and `MapReader <https://arxiv.org/abs/2111.15592>`__ papers. The success of the tool subsequently generated interest from plant phenotype researchers working with large image datasets, and so MapReader is an example of cross-pollination between the humanities and the sciences made possible by reproducible data science.

MapReader has two main components: preprocessing/annotation and training/inference as shown in this figure:

.. image:: https://raw.githubusercontent.com/Living-with-machines/MapReader/main/figs/MapReader_pipeline.png

It provides a set of tools to:

-  **load** images or maps stored locally or **retrieve** maps via web-servers (e.g., tileservers which can be used to retrieve maps from OpenStreetMap (OSM), the National Library of Scotland (NLS), or elsewhere). :warning: Refer to the `credits and re-use terms <#credits-and-re-use-terms>`__ section if you are using digitized maps or metadata provided by NLS.
-  **preprocess** images or maps (e.g., divide them into patches, resampling the images, removing borders outside the neatline or reprojecting the map).
-  annotate images or maps or their patches (i.e. slices of an image ormap) using an **interactive annotation tool**.
-  **train, fine-tune, and evaluate** various CV models.
-  **predict** labels (i.e., model inference) on large sets of images or maps.
-  Other functionalities include:
   -  various **plotting tools** using, e.g., *matplotlib*, *cartopy*, *Google Earth*, and `kepler.gl <https://kepler.gl/>`__.
   -  compute mean/standard-deviation **pixel intensity** of image patches.

How to contribute
-----------------

We welcome contributions related to new applications, both with geospatial images (other maps, remote sensing data, aerial photography) and non-geospatial images (for example, other scientific image datasets).

How to cite MapReader
---------------------

Please consider acknowledging MapReader if it helps you to obtain results and figures for publications or presentations, by citing `MapReader: a computer vision pipeline for the semantic exploration of maps at scale: <https://dl.acm.org/doi/10.1145/3557919.3565812>`__

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

MapReader can retrieve maps from NLS (National Library of Scotland) via webservers. For all the digitized maps (retrieved or locally stored), please note the re-use terms:

:warning: Use of the digitised maps for commercial purposes is currently restricted by contract. Use of these digitised maps for non-commercial purposes is permitted under the `Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International <https://creativecommons.org/licenses/by-nc-sa/4.0/>`__ (CC-BY-NC-SA) licence. Please refer to https://maps.nls.uk/copyright.html#exceptions-os for details on copyright and re-use license. 

Metadata
~~~~~~~~

We have provided some metadata files in ``mapreader/persistent_data``.
For all these file, please note the re-use terms:

:warning: Use of the metadata for commercial purposes is currently restricted by contract. Use of this metadata for non-commercial purposes is permitted under the `Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International <https://creativecommons.org/licenses/by-nc-sa/4.0/>`__ (CC-BY-NC-SA) licence. Please refer to https://maps.nls.uk/copyright.html#exceptions-os for details on copyright and re-use license.

Acknowledgements
~~~~~~~~~~~~~~~~

This work was supported by Living with Machines (AHRC grant AH/S01179X/1) and The Alan Turing Institute (EPSRC grant EP/N510129/1). Living with Machines, funded by the UK Research and Innovation (UKRI) Strategic Priority Fund, is a multidisciplinary collaboration delivered by the Arts and Humanities Research Council (AHRC), with The Alan Turing Institute, the British Library and the Universities of Cambridge, East Anglia, Exeter, and Queen Mary University of London.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

