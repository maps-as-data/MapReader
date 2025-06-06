How to add to or update the MapReader documentation
====================================================

MapReader's documentation is generated using `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ and hosted on `Read the docs <https://readthedocs.org/>`_.
There are a number of ways you can contribute to our documentation, these include:

- Updating or adding clarity to existing documentation.
- Fixing errors in existing documentation (e.g. typos or code inconsistencies).
- Creating new worked examples which showcasing MapReader use cases.

Documentation dependencies
--------------------------

If you would like to edit or add to the MapReader documentation, you will need to install ``sphinx`` along with the packages detailed in ``MapReader/docs/requirements.txt``.

To do this (assuming you have installed MapReader from source, as per our :doc:`Installation instructions </getting-started/installation-instructions/index>`), use:

.. code-block:: bash

    conda activate mapreader
    pip install sphinx
    pip install -r MapReader/docs/requirements.txt

.. note:: You may need to change the file path depending on where your MapReader directory is saved.

Writing in reStructuredText
---------------------------

reStructuredText (rst) is the default plaintext markup language used by `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ and is the primary language used throughout our documentation.
If you have never used or written in rst, `this primer <https://docutils.sourceforge.io/rst.html>`_ is a great place to start.
There are also numerous other rst 'cheatsheets' (e.g. `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer>`__ and `here <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`__) available online, so have a google.

To help make your rst files easier to read and review, **please start each new sentence on a new line**.
This will make no difference to how the text is displayed, but will make it much easier to read when reviewing changes in a pull request.

Before you begin
----------------

Before you begin making your changes to the documentation, you should ensure there is a corresponding issue on the `MapReader repository <https://github.com/Living-with-machines/MapReader>`_ (or create one if needed).
You should then either assign yourself to the issue or comment on it saying you will be working on making these changes.

You will then need to fork the `MapReader repository <https://github.com/Living-with-machines/MapReader>`_ in order to make your changes.

Please also be aware that API documentation is auto-generated by the ``sphinx-autoapi`` extension.
If you would like to change something in the API documentations, you will need to change the docstrings in the code itself.
Head to our :doc:`guide to updating the MapReader code </community-and-contributions/contribution-guide/getting-started-with-contributions/add-or-update-code>` if you would like to have a go at this, or, if you feel hesitant about making code changes, create an issue detailing the changes you think need making.

Style guide
-----------

- Use the following heading styles:

.. ::

    One
    ===
    Two
    ---
    Three
    ~~~~~
    Four
    ^^^^

- Use ``.. code-block:: <lang>`` to create code blocks formatted as per your given langauge (replace ``<lang>`` with the language you will be writing in). e.g. ``.. code-block:: python`` will create a code block with Python formatting.
- Use ``Link title <http://www.anexamplelink.com>`__`` to link to external pages.
- Use ``.. contents::`` to automatically generate a table of contents detailing sections within the current page. e.g.

::

    .. contents:: Table of contents
        :depth: 1

- Use ``.. toc-tree::`` to generate a table of contents (toc) linking to other pages in the documentation. e.g.

::

    .. toc-tree::
        :maxdepth: 1

        page1
        page2

- Use ``.. note::`` and ``.. warning::`` to add notes and warnings.

If you are adding to the User Guide, you may also want to consider how you structure your documentation to ensure clarity.
So far, we have chosen to first give a generic example of how to use the code, followed by a specific example labelled as ``#EXAMPLE``. e.g.

Generic example:

.. code-block:: python

    from mapreader import loader

    my_files = loader("./path/to/files/*png")

Followed by specific example:

.. code-block:: python

    #EXAMPLE
    my_files = loader("./maps/*.png")

Previewing your changes
------------------------

To preview your changes, you can build the documentation locally.

To do this, navigate to the ``MapReader/docs`` directory and run:

.. code-block:: bash

    make livehtml

This will build the documentation and open a new tab in your browser with the documentation.

.. note::
    If a new tab does not open automatically, you can navigate to ``http://127.0.0.1:8000`` in your browser to view the live documentation.

The ``livehtml`` command will automatically update the documentation as you make changes to the files.

When you are finished
----------------------

Once you are happy with the changes you have made, please create a new `pull request <https://github.com/Living-with-machines/MapReader/pulls>`_ to let us know you'd like us to review your work.

If possible, please link your pull request to any issue(s) your changes fix/address and write a thorough description of the changes you have made.
