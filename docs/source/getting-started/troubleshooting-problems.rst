Troubleshooting
===============

numpy/scikit-image issues with M1 mac
-------------------------------------

If you are using an M1 mac and are having issues installing MapReader due to an error when installing numpy or scikit-image:

-  Try separately installing the problem packages (edit as needed) and then installing MapReader:

   .. code-block:: bash

      pip install numpy==1.21.5
      pip install scikit-image==0.18.3
      pip install mapreader

-  Alternatively, you can try using a different version of openBLAS when installing:

   .. code-block:: bash

      brew install openblas
      OPENBLAS="$(brew --prefix openblas)" pip install mapreader


detectron2 issues on Windows
----------------------------

If you are having issues installing detectron2 and running a windows machine, please try the following:

- Install `Visual Studio Build Tools <https://visualstudio.microsoft.com/downloads/?q=build+tools>`__.
- Follow instructions `here <https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst>`__ to install the required packages. (The format might be different in newer versions of Visual Studio Build Tools, so you might need to look up the specific package names.
- Once this is done, try rerunning the installation of detectron2 (`pip install "mapreader[text]"`)
