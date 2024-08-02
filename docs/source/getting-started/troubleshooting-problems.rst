Troubleshooting
===============

M1 mac
------

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
