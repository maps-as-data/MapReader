Installation instructions
=========================

.. note:: Run these commands from your terminal

Step 1: Set up a conda environment
------------------------------------

We recommend installation via Anaconda (refer to `Anaconda website and follow the instructions <https://docs.anaconda.com/anaconda/install/>`_).

-  Create a new environment for ``mapreader`` called ``mr_py38``:

.. code-block:: bash

   conda create -n mr_py38 python=3.8

-  Activate the environment:

.. code-block:: bash

   conda activate mr_py38

Step 2: Install MapReader
--------------------------

Method 1: Install from `PyPI <https://pypi.org/project/mapreader/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Install ``mapreader``:

.. code-block:: bash

   pip install mapreader 

To allow the newly created ``mr_py38`` environment to show up in the notebooks:

.. code-block:: bash

   python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"


Method 2: Install from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Clone ``mapreader`` source code:

.. code-block:: bash

   git clone https://github.com/Living-with-machines/MapReader.git 

-  Install:

.. code-block:: bash

   cd /path/to/MapReader
   pip install -v -e .

Step 3: Add environment to notebooks
--------------------------------------

To allow the newly created ``mr_py38`` environment to show up in the
   notebooks:

.. code-block:: bash

   python -m ipykernel install --user --name mr_py38 --display-name "Python (mr_py38)"
