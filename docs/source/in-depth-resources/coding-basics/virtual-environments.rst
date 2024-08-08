Conda and python virtual environments
======================================

A virtual environment is a way to create an isolated environment in which you can install specific versions of packages and dependencies for your Python projects.
This is useful because it allows you to avoid conflicts between different projects that may require different versions of the same package.
In the installation instructions provided, there are two methods for setting up a virtual environment for MapReader: using Anaconda (also known as conda) or using venv, which is Python's native way of handling virtual environments.

Benefits of using virtual environments
--------------------------------------

- **Isolation**: Each virtual environment is isolated from the system Python installation and other virtual environments, so you can have different versions of packages in each environment.
- **Dependency management**: You can specify the exact versions of packages that your project depends on, making it easier to reproduce your environment on different machines.
- **Sandboxing**: Virtual environments provide a sandboxed environment for your project, so you can experiment with different packages and configurations without affecting your system Python installation.
- **Reproducibility**: By using virtual environments, you can ensure that your project will run the same way on different machines, regardless of the system Python installation or other dependencies.

Creating a virtual environment with conda
------------------------------------------

Conda is a package manager that is commonly used for data science and scientific computing projects.
It also includes a virtual environment manager that allows you to create and manage virtual environments for your Python projects.

To create a virtual environment with conda, you can use the following command:

.. code-block:: bash

    conda create --name myenv

This will create a new virtual environment named ``myenv``.

To activate the virtual environment, you can use the following command:

.. code-block:: bash

    conda activate myenv

Once the virtual environment is activated, you can install packages using ``conda`` or ``pip``, and they will be installed in the virtual environment rather than the system Python installation.

To deactivate the virtual environment, you can use the following command:

.. code-block:: bash

    conda deactivate

For more information on using conda and virtual environments, you can refer to the `official documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Creating a virtual environment with venv
-----------------------------------------

If you prefer to use Python's native virtual environment manager, you can use the ``venv`` module to create a virtual environment for your project.

To create a virtual environment with ``venv``, you can use the following command:

.. code-block:: bash

    python -m venv myenv

This will create a new virtual environment named ``myenv``.

To activate the virtual environment, you can use the following command:

.. code-block:: bash

    source myenv/bin/activate

Once the virtual environment is activated, you can install packages using ``pip``, and they will be installed in the virtual environment rather than the system Python installation.

To deactivate the virtual environment, you can use the following command:

.. code-block:: bash

    deactivate

For more information on using virtual environments in Python, you can refer to the `official documentation <https://docs.python.org/3/tutorial/venv.html>`_.

Additional Resources
--------------------

Here are some resources to help you get started with virtual environments and Anaconda:

- `Getting started with python environments (using conda) <https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307>`__
- `Why you need python environments and how to manage them with conda <https://www.freecodecamp.org/news/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c/>`__
- `Virtual environments and packages <https://docs.python.org/3/tutorial/venv.html>`__

.. more??
