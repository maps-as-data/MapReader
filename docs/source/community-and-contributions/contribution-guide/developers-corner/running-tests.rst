Running tests
=============

To run the tests for MapReader, you will need to have installed the **dev dependencies** (as described :doc:`here </getting-started/installation-instructions/2-install-mapreader>`.

.. note:: If you have followed the "Install from PyPI" instructions, you will also need to clone the MapReader repository to access the tests. i.e.:

.. code-block:: bash

   git clone https://github.com/Living-with-machines/MapReader.git

You can then run the tests using from the root of the MapReader directory using the following commands:

.. code-block:: bash

   cd path/to/MapReader # change this to your path, e.g. cd ~/MapReader
   conda activate mapreader
   python -m pytest -v

If all tests pass, this means that MapReader has been installed and is working as expected.

Testing text spotting
---------------------

The tests for the text spotting code are separated from the main tests due to dependency conflicts.

You will only be able to run the text spotting tests for the text spotting framework (DPTextDETR, DeepSolo or MapTextPipeline) you have installed.

For DPTextDETR, use the following commands:

.. code-block:: bash

   cd path/to/MapReader # change this to your path, e.g. cd ~/MapReader
   conda activate mapreader
   export ADET_PATH=path/to/DPTextDETR # change this to the path where you have saved the DPTextDETR repository
   wget https://huggingface.co/rwood-97/DPText_DETR_ArT_R_50_poly/resolve/main/art_final.pth # download the model weights
   python -m pytest -v tests_text_spotting/test_dptext_runner.py


For DeepSolo:

.. code-block:: bash

   cd path/to/MapReader # change this to your path, e.g. cd ~/MapReader
   conda activate mapreader
   export ADET_PATH=path/to/DeepSolo # change this to the path where you have saved the DeepSolo repository
   wget https://huggingface.co/rwood-97/DeepSolo_ic15_res50/resolve/main/ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth # download the model weights
   python -m pytest -v tests_text_spotting/test_deepsolo_runner.py

For MapTextPipeline:

.. code-block:: bash

   cd path/to/MapReader # change this to your path, e.g. cd ~/MapReader
   conda activate mapreader
   export ADET_PATH=path/to/MapTextPipeline # change this to the path where you have saved the MapTextPipeline repository
   wget https://huggingface.co/rwood-97/MapTextPipeline_rumsey/resolve/main/rumsey-finetune.pth # download the model weights
   python -m pytest -v tests_text_spotting/test_maptext_runner.py


If all tests pass, this means that the text spotting framework has been installed and is working as expected.
