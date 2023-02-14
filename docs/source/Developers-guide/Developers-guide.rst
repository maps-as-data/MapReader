Developers Guide
================

Managing version numbers
------------------------

The software version number is managed by the `versioneer <https://github.com/python-versioneer/python-versioneer>`_ package.  To update the version number, run the following command:

.. code-block:: bash

  git tag -a v1.2.3 -m "mapreader-1.2.3"
  git push --tags
