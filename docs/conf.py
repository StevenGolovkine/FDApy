# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import inspect
import os
import sys

from os.path import dirname, relpath
from typing import Mapping

import FDApy
#sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "FDApy"
author = "Steven Golovkine"

# The full version, including alpha/beta/rc tags
release = "1.0.2"
github_url = "https://github.com/StevenGolovkine/FDApy"

rtd_branch = os.environ.get(" READTHEDOCS_GIT_IDENTIFIER", "master")
language = "en"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
]


master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "use_edit_page_button": True,
    "github_url": github_url,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/FDApy/",
            "icon": "https://avatars.githubusercontent.com/u/2964877",
            "type": "url",
        },
    ]
}

html_context = {
    "github_user": "StevenGolovkine",
    "github_repo": "FDApy",
    "github_version": "master",
    "doc_path": "docs",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../examples",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    "reference_url": {
        # The module you locally document uses None
        "FDApy": None,
    },
    "backreferences_dir": "backreferences",
    "doc_module": "FDApy",
}

autosummary_generate = True
autodoc_member_order = "bysource"

autodoc_default_options = {"show-inheritance": True}


# -- Options for "sphinx.ext.linkcode" --


def linkcode_resolve(domain: str, info: Mapping[str, str]) -> str | None:
    """
    Resolve a link to source in the Github repo.

    Based on the NumPy version.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    fn = None
    lineno = None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return None

    # Ignore re-exports as their source files are not within the FDApy repo
    module = inspect.getmodule(obj)
    if module is not None and not module.__name__.startswith("FDApy"):
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
        lineno_final = lineno + len(source) - 1
    except Exception:
        lineno_final = None

    fn = relpath(fn, start=dirname(FDApy.__file__))

    if lineno:
        linespec = f"#L{lineno}-L{lineno_final}"
    else:
        linespec = ""

    return f"{github_url}/tree/{rtd_branch}/FDApy/{fn}{linespec}"
