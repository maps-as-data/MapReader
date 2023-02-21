Developers Guide
================

Managing version numbers
------------------------

The software version number is managed by the `versioneer <https://github.com/python-versioneer/python-versioneer>`_ package.  To update the version number, run the following command:

.. code-block:: bash

  git tag -a v1.2.3 -m "mapreader-1.2.3"
  git push --tags


Building the Conda package
--------------------------

The overall challenge with installing MapReader is that it some of its dependencies are only available on PyPI, whilst others are only available on conda-forge. 

The solution is to create and build Conda packages that wrap each of the packages that are only available on PyPI, into a local Conda channel.  This local Conda channel is then used to install MapReader. The following directory structure is used:

.. code-block:: bash

    conda
    ├── meta.yaml            # <-- Conda recipe for MapReader
    ├── parhugin
    │   └── conda
    │       └── meta.yaml    # <-- Conda recipe for parhugin
    └── ipyannotate
        └── conda
            ├── meta.yaml    # <-- Conda recipe for ipyannotate
            └── setup.py     # <-- setup script for ipyannotate (needed because the source on PyPI does not include setup.py)


The minimal build process is as follows:

.. code-block:: bash

  mkdir /path/to/local/conda/channel
  conda index /path/to/local/conda/channel
  conda-build ./conda/parhugin/conda --output-folder /path/to/local/conda/channel
  conda-build ./conda/ipyannotate/conda --output-folder /path/to/local/conda/channel
  conda-build -c file:///path/to/local/conda/channel ./conda


