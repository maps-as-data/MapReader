Contributing to MapReader
==========================

Welcome! We are pleased to know that you’re interested in contributing to MapReader!

We welcome all contributions to this project via GitHub issues and pull requests. 
Please follow these guidelines to make sure your contributions can be easily integrated into the project. 
As you start contributing to MapReader, don't forget that your ideas are more important than perfect pull requests. 

.. contents:: Table of Contents
    :local:

Pre-requisites
---------------

In order to contribute to MapReader, please make sure you fullfill the following requirements:

1. Make sure you have set up a `GitHub account <https://docs.github.com/en/get-started/signing-up-for-github/signing-up-for-a-new-github-account>`_.
2. Set up a virtual python environment and install MapReader (as per `Installation instuctions <https://mapreader.readthedocs.io/en/rw_docs/Install.html>`_).
3. Read this guide.

For more information about the origins of MapReader, check out our `paper <https://dl.acm.org/doi/10.1145/3557919.3565812>`_ about the pipeline design and early experiments.

Joining the community
----------------------

MapReader is a collaborative project now expanding its community beyond the initial group in the `Living with Machines <https://livingwithmachines.ac.uk/>`_ project (The Alan Turing Institute). 
We are in the process of developing a Code of Conduct. 
In the meantime, we look to the `Code of Conduct <https://github.com/alan-turing-institute/the-turing-way/blob/main/CODE_OF_CONDUCT.md>`_ from The Turing Way as a model.

Inclusivity
~~~~~~~~~~~~

MapReader aims to be inclusive of people from all walks of life and all research fields. 
These intentions must be reflected in the contributions that we make.

In addition to the Code of Conduct, we encourage intentional, inclusive actions from contributors to MapReader. 
Here are a few examples of such actions:

- use respectful, gender-neutral and inclusive language (learn more about inclusive writing on page 22 of University of Leicester Study Skills pdf, also available as a zipped html).
- aim to include perspectives of researchers from different research backgrounds such as science, humanities and social sciences by not limiting the scope to only scientific domains.
- make sure that colour palettes used throughout figures are accessible to colour-blind readers and contributors.

Get in touch
~~~~~~~~~~~~~~

There are many ways to get in touch with the MapReader team:

- Github issues and pull requests (find our Github handles `here <https://github.com/Living-with-machines/MapReader/blob/main/ways_of_working.md>`_).
- Contact Katie McDonough by via email (k.mcdonough@lancaster.ac.uk).

Contributing through GitHub
-----------------------------

`Git <https://git-scm.com/>`_ is a really useful tool for version control. GitHub sits on top of Git and supports collaborative and distributed working.

We know that it can be daunting to start using Git and GitHub if you haven't worked with them in the past, but MapReader maintainers are here to help you figure out any of the jargon or confusing instructions you encounter! 

GitHub issues - Where to start?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you open a new issue, please check if any of our open issues cover your idea already!

If you do need to open a new issue, please follow our basic guidelines laid out in our issue templates. 
There are two issue templates to choose from:

- **Bug report** (`preview here <https://github.com/Living-with-machines/MapReader/blob/main/.github/ISSUE_TEMPLATE/bug_report.md>`_): This template should be used to report bugs in the MapReader code and for reporting errors like typos and broken links (e.g. within the documentation).
- **Feature request** (`preview here <https://github.com/Living-with-machines/MapReader/blob/main/.github/ISSUE_TEMPLATE/feature_request.md>`_): This template should be used to suggest new features/functionalities that could be incoroportated into the MapReader code and suggesting updates to documentation/tutorials/etc. 

If you feel your issue does not fit either of these templates, please open a blank issue and provide as much information as possible to help the MapReader team work resolve the issue.

Making your own changes with a pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you feel able to make the required changes to the MapReader code/documentation/tutorials in order to resolve an issue, please feel free to do so. 
You should then make a pull request in order to request that the MapReader team pull these changes into the MapReader repository.

To do this:

1. Comment on an existng issue (or create a new issue) and let people know the changes you plan on making.
2. Fork the `MapReader repository <https://github.com/Living-with-machines/MapReader>`_.
3. Make your changes.
4. Submit a pull request.

Please ensure your pull request contains as much information as possible and links to the issue(s) you are resolving. 
Our pull request template (`preview here <https://github.com/Living-with-machines/MapReader/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`_) should automatically help you with this.

Once created, your pull request will be reviewed by a member of the MapReader team and, once approved, your changes will be merged into the MapReader repository.
To make this review process as easy as possible, please try to work on only one issue/problem per pull request.
You may find it useful to create separate `branches <https://www.atlassian.com/git/tutorials/using-branches>`_ for each issue/problem you are working on. 
Each of these can then be linked to their own pull request.

MapReader code
----------------

MapReader's code is written in `Python <https://www.python.org/>`_ and distributed using `PyPI <https://pypi.org/>`_. 
There are a number of ways in which you can contribute to our code, these include:

- Suggesting code changes.
- Reviewing part of the code.
- Creating a new feature request.
- Reporting a bug and (if possible) suggesting solutions.

MapReader documentation 
-------------------------

MapReader's documentation is generated using `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ and hosted on `Read the docs <https://readthedocs.org/>`_. 
There are a number of ways you can contribute to our documentation, these include:

- Suggesting and drafting a tutorial that orients new users to make the most of specific features.
- Updating or modularising existing tutorials so they better serve a specific community of users needs.
- Showcasing examples of MapReader use cases.

Writing in reStructuredText
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

reStructuredText (rst) is the default plaintext markup language used by `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ and is the language used throughout our documentation.
If you have never used or written in rst, `this primer <https://docutils.sourceforge.io/rst.html>`_ is a great place to start. There are also numerous other rst 'cheatsheets' (e.g. `here <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#rst-primer>`_ and `here <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_) available online, so have a google.

To help make your rst files easier to read and review, please start each new sentence on a new line. 
This will make no difference to how the text is displayed, but will make it much easier to read when reviewing changes in a pull request.

Acknowledgements
-----------------

This contribution guide has been adapted from `The Turing Way's guidelines <https://github.com/alan-turing-institute/the-turing-way/blob/main/CONTRIBUTING.md>`_, which were themselves an adaptation of the `BIDS Starter Kit Contribution Guidelines <https://github.com/bids-standard/bids-starter-kit/blob/main/CONTRIBUTING.md>`_ (CC-BY).