Installing and using pre-commit
--------------------------------

MapReader uses `pre-commit <https://pre-commit.com/>`_ to enforce code style and quality.  To install pre-commit, run the following commands:

.. code-block:: bash

  pip install pre-commit
  pre-commit install

This will install the pre-commit hooks in the repository.  The hooks will run automatically when you commit code.  If the hooks fail, the commit will be aborted.
