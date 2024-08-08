Running tests
=============

To run the tests for MapReader, you will need to have installed the **dev dependencies** as described above.

Also, if you have followed the "Install from PyPI" instructions, you will need to clone the MapReader repository to access the tests. i.e.:

.. code-block:: bash

   git clone https://github.com/Living-with-machines/MapReader.git

You can then run the tests using from the root of the MapReader directory using the following commands:

.. code-block:: bash

   cd path/to/MapReader # change this to your path, e.g. cd ~/MapReader
   conda activate mapreader
   python -m pytest -v

If all tests pass, this means that MapReader has been installed and is working as expected.
