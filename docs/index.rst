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

Functional data analysis (FDA) is a statistical methodology for analyzing data that can be characterized as functions. These functions could represent measurements taken over time, space, frequency, probability, etc. The goal of FDA is to extract meaningful information from these functions and to model their behavior.

The package aims to provide functionalities for creating and manipulating general functional data objects. It thus supports the analysis of various types of functional data, whether densely or irregularly sampled, multivariate, or multidimensional. Functional data can be represented over a grid of points or using a basis of functions. `FDApy` implements dimension reduction techniques and smoothing methods, facilitating the extraction of patterns from complex functional datasets. A large simulation toolbox, based on basis decomposition, is provided. It allows to configure parameters for simulating different clusters within the data. Finally, some visualization tools are also available.

Check out the `examples <https://fdapy.readthedocs.io/en/latest/auto_examples/index.html>`_ for an overview of the package functionalities.

Check out the `API reference <https://fdapy.readthedocs.io/en/latest/modules.html>`_ for an exhaustive list of the available features within the package.

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

Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. Contributing guidelines are provided `here <https://github.com/StevenGolovkine/FDApy/blob/master/CONTRIBUTING.rst>`_. The people involved in the development of the package can be found in the `contributors page <https://github.com/StevenGolovkine/FDApy/blob/master/CONTRIBUTORS.md>`_.

License
=======

The package is licensed under the MIT License. A copy of the `license <https://github.com/StevenGolovkine/FDApy/blob/master/LICENSE>`_ can be found along with the code.
