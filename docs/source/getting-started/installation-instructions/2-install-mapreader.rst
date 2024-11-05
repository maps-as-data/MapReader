Step 2: Install MapReader
==========================

Method 1: Install from `PyPI <https://pypi.org/project/mapreader/>`_
--------------------------------------------------------------------

Installing MapReader from PyPI is probably the easiest way to install MapReader.

We recommend using this method if:

- You want to use the latest stable release of MapReader.
- You **only want to use MapReader's classification pipeline** (i.e. you do not need the text spotting functionality).
- You do not need access to the worked examples or MapReader code.

To install ``mapreader`` without the text spotting dependencies (i.e. just the classification pipeline):

.. code-block:: bash

   pip install mapreader

.. note:: To install the dev dependencies too, use ``pip install "mapreader[dev]"``.

To install the text-spotting dependencies, head to the :doc:`Spot text </using-mapreader/step-by-step-guide/6-spot-text>` section of the user guide!

Method 2: Install from source
-----------------------------

Installing from source is the best way to install MapReader if you want to use the text spotting functionality or access the worked examples.

We recommend using this method if:

- You want to keep up with the latest changes to MapReader.
- You **want to use the text spotting functionality** in addition to the classification pipeline.
- You want access to the worked examples.
- You want access to the MapReader code (e.g. for development purposes).

This method will clone the ``MapReader`` repository onto your machine. This folder will contain all the MapReader code, docs and worked examples.

.. note:: You will need to have `git <https://git-scm.com/>`__ installed to use this method. If you are using conda, this can be done by running ``conda install git``. Otherwise, you should install git by following the instructions on `their website <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`__.

Clone the ``mapreader`` source code from the `MapReader GitHub repository <https://github.com/Living-with-machines/MapReader>`_:

.. code-block:: bash

   git clone https://github.com/Living-with-machines/MapReader.git

Then, to install ``mapreader`` without the text spotting dependencies:

.. code-block:: bash

   cd MapReader
   pip install .

.. note:: To install the dev dependencies too, use ``pip install ".[dev]"``.

Finally, to install the text spotting dependencies, you should run:

.. code-block:: bash

   cd MapReader
   pip install -r text-requirements.txt

..
   Method 3: Install via conda (**EXPERIMENTAL**)
   ----------------------------------------------

   If neither of the above methods work, you can try installing MapReader using conda.
   This method is still in development so should be avoided for now.

   - Install MapReader directly from the conda package:

   .. code:: bash

      conda install -c anothersmith -c conda-forge -c defaults --override-channels --strict-channel-priority mapreader

   .. note:: The conda package seems to be sensitive to the precise priority of the conda channels, hence the use of the `--override-channels --strict-channel-priority` switches is required for this to work. Until this is resolve this installation method will be marked "experimental".
