Installation instructions
=========================

Set up a conda environment
---------------------------

We recommend installation via Anaconda (refer to `Anaconda website and follow the instructions <https://docs.anaconda.com/anaconda/install/>`__).

-  Create a new environment for ``mapreader`` called ``mr_py38``:

.. code :: bash

   conda create -n mr_py38 python=3.8

-  Activate the environment:

.. code :: bash

   conda activate mr_py38

Install via pip
------------------

-  Install ``mapreader``:

.. code :: bash

   pip install mapreader 

.. warning::
   If this fails when installing cartopy, check you have `GEOS <https://libgeos.org/>`__ and `PROJ <https://proj.org/index.html>`__ installed on your machine. As these are not PyPI packages, they cannot be installed via `pip`.

To allow the newly created ``mr_py38`` environment to show up in the notebooks:

.. code :: bash

   python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"


Install from source
----------------------

-  Clone ``mapreader`` source code:

.. code :: bash

   git clone https://github.com/Living-with-machines/MapReader.git 

-  Install:

.. code :: bash

   cd /path/to/MapReader
   pip install -v -e .

.. warning::
   As above, if this fails when installing cartopy, check you have `GEOS <https://libgeos.org/>`__ and `PROJ <https://proj.org/index.html>`__ installed on your machine. As these are not PyPI packages, they cannot be installed via `pip`.

To allow the newly created ``mr_py38`` environment to show up in the
   notebooks:

.. code :: bash

   python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"
