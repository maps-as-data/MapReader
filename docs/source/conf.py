# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import sphinx_rtd_theme

import mapreader

project = "MapReader"
copyright = "2023, RW"
author = "RW"

release = mapreader.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx_rtd_theme",
    "myst_parser",
    "autoapi.extension",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_togglebutton",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- autoapi configuration -----

autoapi_dirs = ["../../mapreader"]
autoapi_type = "python"
autoapi_root = "api"

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autoapi_keep_files = True
autodoc_typehints = "description"
autoapi_add_toctree_entry = False

# -- Napoleon settings ----

napoleon_numpy_docstring = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

# -- todo --
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ["_static"]
