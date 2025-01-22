.. FDApy documentation master file, created by
   sphinx-quickstart on Tue Jun  9 11:47:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FDApy: a Python package to analyze functional data
==================================================

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   auto_examples/index


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:
   :caption: More documentation

   modules


Description
===========

Functional Data Analysis, usually referred as FDA, concerns the field of Statistics that deals with discrete observations of continuous :math:`d`-dimensional functions.

This package provide modules for the analysis of such data. It includes methods for different dimensional data as well as irregularly sampled functional data. An implementation of (multivariate) functional principal component analysis is also given. Moreover, a simulation toolbox is provided. It might be used to simulate different clusters of functional data.
Check out the documentation for more complete information on the available features within the package.

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

* `lazy_loader <https://github.com/scientific-python/lazy-loader>`_ - A loader for Python submodules
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
