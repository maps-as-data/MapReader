Step 1: Set up a virtual Python environment
===========================================

.. todo:: Add comments about how to get to conda in Windows

The most recent version of MapReader supports Python versions 3.9+.

Method 1: Using conda (recommended)
------------------------------------

We recommend installing MapReader using either Anaconda (`installation instructions here <https://docs.anaconda.com/anaconda/install/>`__) or miniconda (`installation instructions here <https://docs.conda.io/en/latest/miniconda.html>`__).
A discussion of which of these to choose can be found `here <https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html>`__.

Once you have installed either Ananconda or miniconda, open your terminal and use the following commands to set up your virtual Python environment:

-  Create a new conda environment for ``mapreader`` (you can call this whatever you like, we use ``mapreader``):

   .. code-block:: bash

      conda create -n mapreader python=3.10

   This will create a conda enviroment for you to install MapReader and its dependencies into.

-  Activate your conda environment:

   .. code-block:: bash

      conda activate mapreader

Method 2: Using venv or other
-----------------------------

If you would like not to use conda, you are more than welcome to set up a virtual Python environment using other methods.

For example, if you would like to use venv, open your terminal and use the following commands to set up your virtual Python environment:

-  First, importantly, check which version of Python your system is using:

   .. code-block:: bash

      python3 --version

   If this returns a version below 3.9, you will need download an updated Python version.
   You can do this by downloading from `here <https://www.python.org/downloads/>`__ (make sure you download the right one for your operating system).

   You should then run the above command again to check your Python version has updated.

-  Create a new virtual Python environment for ``mapreader`` (you can call this whatever you like, we use ``mapreader``):

   .. code-block:: bash

      python3 -m venv mapreader

-  Activate your virtual environment:

   .. code-block:: bash

      source mapreader/bin/activate
