"""Sphinx configuration for the KilonovaSCORER documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "KilonovaSCORER"
author = "P. Darc, C. D. Kilpatrick"
copyright = "2026, P. Darc"
release = "0.1.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "myst_parser",
    "sphinx_copybutton",
]

# Redback / bilby / lal / astropy are only needed to *simulate* light curves.
# They are optional, so the docs build must not require them.
autodoc_mock_imports = ["redback", "bilby", "lal", "astropy"]
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True

# Show the notebook outputs that are already saved in the file.
nbsphinx_execute = "never"

myst_enable_extensions = ["dollarmath", "amsmath"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

html_theme = "furo"
html_title = "KilonovaSCORER"
html_logo = "_static/logo2.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/phelipedarc/KilonovaSCORER/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {"color-brand-primary": "#7b2cbf", "color-brand-content": "#7b2cbf"},
    "dark_css_variables": {"color-brand-primary": "#c77dff", "color-brand-content": "#c77dff"},
}
