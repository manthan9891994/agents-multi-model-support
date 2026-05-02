"""Sphinx configuration for dynamic-model-router."""
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

from classifier import __version__

project   = "dynamic-model-router"
author    = "Manthan Vaghela"
copyright = "2026, Manthan Vaghela"
release   = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
