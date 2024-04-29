Installation Instructions
=========================

.. note:: Run these commands from your terminal.

There are three steps to setting up MapReader.
You should choose one method within each step and follow the instructions for that method.

.. note:: You do not need to use the same method between steps. i.e. It is completely fine to follow Method 1 for Step 1 and Method 2 for Step 2.

.. contents:: Table of Contents
   :depth: 2
   :local:

.. todo:: Add comments about how to get to conda in Windows

Step 1: Set up a virtual python environment
----------------------------------------------

The most recent version of MapReader supports python versions 3.9+.

Method 1: Using conda (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing MapReader using either Anaconda (`installation instructions here <https://docs.anaconda.com/anaconda/install/>`__) or miniconda (`installation instructions here <https://docs.conda.io/en/latest/miniconda.html>`__).
A discussion of which of these to choose can be found `here <https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html>`__.

Once you have installed either Ananconda or miniconda, open your terminal and use the following commands to set up your virtual python environment:

-  Create a new conda environment for ``mapreader`` (you can call this whatever you like, we use ``mapreader``):

   .. code-block:: bash

      conda create -n mapreader python=3.10

   This will create a conda enviroment for you to install MapReader and its dependencies into.

-  Activate your conda environment:

   .. code-block:: bash

      conda activate mapreader

Method 2: Using venv or other
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you would like not to use conda, you are more than welcome to set up a virtual python environment using other methods.

For example, if you would like to use venv, open your terminal and use the following commands to set up your virtual python environment:

-  First, importantly, check which version of python your system is using:

   .. code-block:: bash

      python3 --version

   If this returns a version below 3.9, you will need download an updated python version.
   You can do this by downloading from `here <https://www.python.org/downloads/>`__ (make sure you download the right one for your operating system).

   You should then run the above command again to check your python version has updated.

-  Create a new virtual python environment for ``mapreader`` (you can call this whatever you like, we use ``mapreader``):

   .. code-block:: bash

      python3 -m venv mapreader

-  Activate your virtual environment:

   .. code-block:: bash

      source mapreader/bin/activate

Step 2: Install MapReader
--------------------------

Method 1: Install from `PyPI <https://pypi.org/project/mapreader/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use the latest stable release of MapReader and do not want/need access to the worked examples or MapReader code, we recommend installing from PyPI.
This is probably the easiest way to install MapReader.

-  Install ``mapreader``:

   .. code-block:: bash

      pip install mapreader


Method 2: Install from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to keep up with the latest changes to MapReader, or want/need easy access to the worked examples or MapReader code, we reccommend installing from source.
This method will create a ``MapReader`` directory on your machine which will contain all the MapReader code, docs and worked examples.

.. note:: You will need to have `git <https://git-scm.com/>`__ installed to use this method. If you are using conda, this can be done by running ``conda install git``. Otherwise, you should install git by following the instructions on `their website <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__.

-  Clone the ``mapreader`` source code from the `MapReader GitHub repository <https://github.com/Living-with-machines/MapReader>`_:

   .. code-block:: bash

      git clone https://github.com/Living-with-machines/MapReader.git

-  Install ``mapreader``:

   .. code-block:: bash

      cd MapReader
      pip install -v -e .


..
   Method 3: Install via conda (**EXPERIMENTAL**)
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   If neither of the above methods work, you can try installing MapReader using conda.
   This method is still in development so should be avoided for now.

   - Install MapReader directly from the conda package:

   .. code:: bash

      conda install -c anothersmith -c conda-forge -c defaults --override-channels --strict-channel-priority mapreader

   .. note:: The conda package seems to be sensitive to the precise priority of the conda channels, hence the use of the `--override-channels --strict-channel-priority` switches is required for this to work. Until this is resolve this installation method will be marked "experimental".

Step 3: Add virtual python environment to notebooks
------------------------------------------------------

- To allow the newly created python virtual environment to show up in jupyter notebooks, run the following command:

.. code-block:: bash

      python -m ipykernel install --user --name mapreader --display-name "Python (mr_py)"

.. note:: if you have used a different name for your python virtual environment replace the ``mapreader`` with whatever name you have used.

Troubleshooting
----------------

M1 mac
~~~~~~~

If you are using an M1 mac and are having issues installing MapReader due to an error when installing numpy or scikit-image:

-  Try separately installing the problem packages (edit as needed) and then installing MapReader:

   .. code-block:: bash

      pip install numpy==1.21.5
      pip install scikit-image==0.18.3
      pip install mapreader

-  Try using conda to install the problem packages (edit as needed) and then pip to install MapReader:

   .. code-block:: bash

      conda install numpy==1.21.5
      conda install scikit-image==0.18.3
      pip install mapreader

-  Alternatively, you can try using a different version of openBLAS when installing:

   .. code-block:: bash

      brew install openblas
      OPENBLAS="$(brew --prefix openblas)" pip install mapreader
