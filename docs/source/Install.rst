Installation instructions
=========================

Set up a conda environment
---------------------------

We recommend installation via Anaconda (refer to `Anaconda website and follow the instructions <https://docs.anaconda.com/anaconda/install/>`__).

-  Create a new environment for ``mapreader`` called ``mr_py38``:

.. code:: bash

   conda create -n mr_py38 python=3.8

-  Activate the environment:

.. code:: bash

   conda activate mr_py38

Method 1 (User Install)
-----------------------

-  Install ``mapreader``:

.. code:: bash

   pip install mapreader 

To work with geospatial images (e.g., maps):

.. code:: bash

   pip install "mapreader[geo]" 

-  We have provided some `Jupyter Notebooks to showcase MapReader’s functionalities <https://github.com/Living-with-machines/MapReader/tree/main/examples>`__.

   To allow the newly created ``mr_py38`` environment to show up in the
   notebooks:

.. code:: bash

   python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"

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

Method 2 (Developer Install)
----------------------------

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


Method 3 (conada install - EXPERIMENTAL)
----------------------------------------

- Create and activate the conda environment:

.. code:: bash

   conda create -n mr_py38 python=3.8
   conda activate mr_py38

- Install MapReader directly from the conda package:

.. code:: bash

   conda install -c anothersmith -c conda-forge -c defaults --override-channels --strict-channel-priority mapreader

(Note: The conda package seems to be sensitive to the precise priority of the conda channels, hence the use of the `--override-channels --strict-channel-priority` switches is required for this to work. Until this is resolve this installation method will be marked "experimental".)

