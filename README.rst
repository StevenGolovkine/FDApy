
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

.. image:: https://img.shields.io/pypi/l/FDApy
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

Description
===========

Functional Data Analysis, usually referred as FDA, concerns the field of Statistics that deals with discrete observations of continuous :math:`d`-dimensional functions.

This package provide modules for the analysis of such data. It includes methods for different dimensional data as well as irregularly sampled functional data. An implementation of (multivariate) functional principal component analysis is also given. Moreover, a simulation toolbox is provided. It might be used to simulate different clusters of functional data.
Check out the documentation for more complete information on the available features within the package.

Documentation
=============

The documentation is available at `https://fdapy.readthedocs.io/en/stable/`_, which included detailled information about API references and several examples presenting the different functionalities.

The documentation of the latest version can be found at `https://fdapy.readthedocs.io/en/latest/`_.

Installation
============

Up to now, *FDApy* is availlable in Python 3.9 on any Linux platforms. The stable version can be installed via `PyPI <https://pypi.org/project/FDApy/>`_:

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

* `matplotlib <https://github.com/matplotlib/matplotlib>`_ - Plotting with Python
* `numpy <https://github.com/numpy/numpy>`_ - The fundamental package for scientific computing with Python
* `pandas <https://github.com/pandas-dev/pandas>`_ - Powerful Python data analysis toolkit
* `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ - Machine learning in Python
* `scipy <https://github.com/scipy/scipy>`_ - Scientific computation in Python

Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. Contributing guidelines are provided `here <https://github.com/StevenGolovkine/FDApy/blob/master/CONTRIBUTING.rst>`_.

License
=======

The package is licensed under the MIT License. A copy of the `license <https://github.com/StevenGolovkine/FDApy/blob/master/LICENSE>`_ can be found along with the code.