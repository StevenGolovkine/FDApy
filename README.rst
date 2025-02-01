
===================================================
FDApy: a Python package to analyze functional data
===================================================

.. image:: https://img.shields.io/pypi/pyversions/FDApy
		:target: https://pypi.org/project/FDApy/
		:alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/v/FDApy   
		:target: https://pypi.org/project/FDApy/
		:alt: PyPI

.. image:: https://github.com/StevenGolovkine/FDApy/actions/workflows/python_package_ubuntu.yaml/badge.svg
		:target: https://github.com/StevenGolovkine/FDApy/actions
		:alt: Github - Workflow

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
		:target: https://raw.githubusercontent.com/StevenGolovkine/FDApy/master/LICENSE
		:alt: PyPI - License

.. image:: https://codecov.io/gh/StevenGolovkine/FDApy/branch/master/graph/badge.svg?token=S2H0D3QQMR 
 		:target: https://codecov.io/gh/StevenGolovkine/FDApy
		:alt: Coverage

.. image:: https://app.codacy.com/project/badge/Grade/3d9062cffc304ad4bb7c76bf97cc965c
		:target: https://app.codacy.com/gh/StevenGolovkine/FDApy/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
		:alt: Code Quality

.. image:: https://readthedocs.org/projects/fdapy/badge/?version=latest
		:target: https://fdapy.readthedocs.io/en/latest/?badge=latest
		:alt: Documentation Status

.. image:: https://zenodo.org/badge/155183454.svg
   		:target: https://zenodo.org/badge/latestdoi/155183454
   		:alt: DOI

.. image:: https://img.shields.io/github/all-contributors/StevenGolovkine/FDApy?color=ee8449&style=flat-square
		:target: https://github.com/StevenGolovkine/FDApy/blob/master/CONTRIBUTORS.md
		:alt: Contributors
		

Description
===========

Functional data analysis (FDA) is a statistical methodology for analyzing data that can be characterized as functions. These functions could represent measurements taken over time, space, frequency, probability, etc. The goal of FDA is to extract meaningful information from these functions and to model their behavior.

The package aims to provide functionalities for creating and manipulating general functional data objects. It thus supports the analysis of various types of functional data, whether densely or irregularly sampled, multivariate, or multidimensional. Functional data can be represented over a grid of points or using a basis of functions. *FDApy* implements dimension reduction techniques and smoothing methods, facilitating the extraction of patterns from complex functional datasets. A large simulation toolbox, based on basis decomposition, is provided. It allows to configure parameters for simulating different clusters within the data. Finally, some visualization tools are also available.

Check out the `examples <https://fdapy.readthedocs.io/en/latest/auto_examples/index.html>`_ for an overview of the package functionalities.

Check out the `API reference <https://fdapy.readthedocs.io/en/latest/modules.html>`_ for an exhaustive list of the available features within the package.


Documentation
=============

The documentation is available `here <https://fdapy.readthedocs.io/en/stable/>`__, which included detailled information about API references and several examples presenting the different functionalities.


Installation
============

Up to now, *FDApy* is availlable in Python 3.10 on any Linux platforms. The stable version can be installed via `PyPI <https://pypi.org/project/FDApy/>`_:

.. code::
	
	pip install FDApy

Installation from source
------------------------

It is possible to install the latest version of the package by cloning this repository and doing the manual installation.

.. code:: bash

	git clone https://github.com/StevenGolovkine/FDApy.git
	pip install ./FDApy

Requirements
------------

*FDApy* depends on the following packages:

* `lazy_loader <https://github.com/scientific-python/lazy-loader>`_ - A loader for Python submodules
* `matplotlib <https://github.com/matplotlib/matplotlib>`_ - Plotting with Python
* `numpy <https://github.com/numpy/numpy>`_ (< 2.0.0) - The fundamental package for scientific computing with Python
* `pandas <https://github.com/pandas-dev/pandas>`_ (>= 2.0.0)- Powerful Python data analysis toolkit
* `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ (>= 1.2.0)- Machine learning in Python
* `scipy <https://github.com/scipy/scipy>`_ (>= 1.10.0) - Scientific computation in Python


Citing FDApy
============

If you use FDApy in a scientific publication, we would appreciate citations to the following software repository:

.. code-block::

  @misc{golovkine_2024_fdapy,
    author = {Golovkine, Steven},
    doi = {10.5281/zenodo.13625609},
    title = {FDApy: A Python Package to analyze functional data},
    url = {https://github.com/StevenGolovkine/FDApy},
    year = {2024}
  }


Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. Contributing guidelines are provided `here <https://github.com/StevenGolovkine/FDApy/blob/master/CONTRIBUTING.rst>`_. The people involved in the development of the package can be found in the `contributors page <https://github.com/StevenGolovkine/FDApy/blob/master/CONTRIBUTORS.md>`_.

License
=======

The package is licensed under the MIT License. A copy of the `license <https://github.com/StevenGolovkine/FDApy/blob/master/LICENSE>`_ can be found along with the code.
