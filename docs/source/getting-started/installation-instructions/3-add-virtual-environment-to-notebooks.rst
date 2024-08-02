Step 3: Add virtual Python environment to notebooks
===================================================

- To allow the newly created Python virtual environment to show up in Jupyter notebooks, run the following command:

.. code-block:: bash

    python -m ipykernel install --user --name mapreader --display-name "Python (mr_py)"

.. note:: if you have used a different name for your Python virtual environment replace the ``mapreader`` with whatever name you have used.
