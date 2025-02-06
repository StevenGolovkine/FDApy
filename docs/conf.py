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

from sphinx.ext.napoleon.docstring import GoogleDocstring

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
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
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


# -- Options for sphinx.ext.autodoc --


# -- Options for "sphinx.ext.autodoc.typehints" --

autodoc_typehints = "description"


# -- Options for "sphinx.ext.autosummary" --

autosummary_generate = True


# -- Options for "sphinx.ext.intersphinx" --
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}


# -- Options for "sphinx_gallery.gen_gallery" --
from sphinx_gallery.sorting import ExplicitOrder

# Gallery sections shall be displayed in the following order.
# Non-matching sections are inserted at the unsorted position

UNSORTED = "unsorted"

examples_order = [
    '../examples/representation',
    '../examples/basis',
    '../examples/smoothing',
    '../examples/fpca',
    '../examples/data_analysis',
    '../examples/simulation',
    '../examples/misc',
    UNSORTED,
]

folder_lists = [examples_order]

explicit_order_folders = [
    fd for folders in folder_lists for fd in folders[:folders.index(UNSORTED)]
]
explicit_order_folders.append(UNSORTED)
explicit_order_folders.extend([
    fd for folders in folder_lists for fd in folders[folders.index(UNSORTED):]
])


class FDApyExplicitOrder(ExplicitOrder):
    """For use within the 'subsection_order' key."""
    def __call__(self, item):
        """Return a string determining the sort order."""
        if item in self.ordered_list:
            return f"{self.ordered_list.index(item):04d}"
        else:
            return f"{self.ordered_list.index(UNSORTED):04d}{item}"

# Subsection order:
# Subsections are ordered by filename, unless they appear in the following
# lists in which case the list order determines the order within the section.
# Examples/tutorials that do not appear in a list will be appended.

list_all = [
    # **Examples
    # Representation
    "plot_dense_functional", "plot_irregular_functional", "plot_basis_functional"
    "plot_multivariate_functional",
    # Basis
    "plot_basis_1d", "plot_basis_2d", "plot_basis_multivariate_1d",
    # Smoothing
    "plot_smooth_data_1d", "plot_smooth_data_2d",
    # FPCA
    "plot_fpca_1d", "plot_fpca_1d_sparse", "plot_fpca_2d",
    "plot_mfpca_1d", "plot_mfpca_2d", "plot_mfpca_1d_2d", "plot_mfpca_1d_sparse",
    # Data analysis
    "plot_canadian_weather", "plot_cd4",
    # Simulation
    "plot_karhunen", "plot_karhunen_multivariate", "plot_brownian",
    "plot_simulation", "plot_cluster", "plot_cluster_multivariate",
    # Misc
    "plot_local_polynomials_1d", "plot_local_polynomials_2d",
    "plot_psplines_1d", "plot_psplines_2d",
]
explicit_subsection_order = [item + ".py" for item in list_all]


class FDApyExplicitSubOrder(ExplicitOrder):
    """For use within the 'within_subsection_order' key."""
    def __init__(self, src_dir):
        self.src_dir = src_dir  # src_dir is unused here
        self.ordered_list = explicit_subsection_order

    def __call__(self, item):
        """Return a string determining the sort order."""
        if item in self.ordered_list:
            return f"{self.ordered_list.index(item):04d}"
        else:
            # ensure not explicitly listed items come last.
            return "zzz" + item


# Provide the above classes for use in conf.py
sectionorder = FDApyExplicitOrder(explicit_order_folders)
subsectionorder = FDApyExplicitSubOrder


sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": "../examples",
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
    "subsection_order": sectionorder,
    'within_subsection_order': subsectionorder,
    "reference_url": {
        # The module you locally document uses None
        "FDApy": None,
    },
    "backreferences_dir": "backreferences",
    "doc_module": "FDApy",
}


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


# -- Options for "sphinx.ext.napoleon" --

napoleon_use_rtype = True


# Napoleon fix for attributes
# Taken from
# https://michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html


# first, we define new methods for any new sections and add them to the class


def parse_keys_section(self, section):
    return self._format_fields("Keys", self._consume_fields())


GoogleDocstring._parse_keys_section = parse_keys_section


def parse_attributes_section(self, section):
    return self._format_fields("Attributes", self._consume_fields())


GoogleDocstring._parse_attributes_section = parse_attributes_section


def parse_class_attributes_section(self, section):
    return self._format_fields("Class Attributes", self._consume_fields())


GoogleDocstring._parse_class_attributes_section = parse_class_attributes_section

# we now patch the parse method to guarantee that the the above methods are
# assigned to the _section dict


def patched_parse(self):
    self._sections["keys"] = self._parse_keys_section
    self._sections["class attributes"] = self._parse_class_attributes_section
    self._unpatched_parse()


GoogleDocstring._unpatched_parse = GoogleDocstring._parse
GoogleDocstring._parse = patched_parse


# -- Options for "sphinxcontrib.bibtex" --

bibtex_bibfiles = ["refs.bib"]