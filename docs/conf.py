"""Sphinx configuration for spdnet-datasets."""

import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath("../src"))

project = "spdnet-training"
author = "Matthieu Gallet, Ammar Mian"
copyright = f"{datetime.now().year}, {author}"
try:
    release = package_version("spdnet-training")
except PackageNotFoundError:
    release = "dev"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

# API reference: hand-written pages with autodoc (see docs/reference/)
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_default_options = {"exclude-members": "__init__, __new__"}
add_module_names = False
python_maximum_signature_line_length = 88
autosummary_generate = False

# Docstrings are in Google style ("Args:", "Returns:")
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "spdnet-training"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/Yet-Another-Research-Organisation/spdnet-training",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#3a4a8c",
        "color-brand-content": "#3a4a8c",
    },
    "dark_css_variables": {
        "color-brand-primary": "#9fb0f0",
        "color-brand-content": "#9fb0f0",
    },
}
