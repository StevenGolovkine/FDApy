=================================
Representation of functional data
=================================

The first step is the analysis of functional data is to represent them in a suitable way. The package provides several classes to represent functional data, depending on the type of data to represent. The package provides classes to represent univariate functional data, irregular functional data, multivariate functional data, and functional data in a basis representation.


Generic representation
======================

Representations of functional data are based on three abstract classes. The first one, :class:`Argvals`, represents the arguments of the functions. The second one, :class:`Values`, represents the values of the functions. The third one, :class:`FunctionalData`, represents the functional data object, which is a pair of :class:`Argvals` and :class:`Values`. The package provides several implementations of these classes, depending on the type of functional data to represent. These classes cannot be instantiated directly, but they are used as base classes for the specific implementations.

.. autosummary::
	:toctree: autosummary

	FDApy.representation.Argvals
	FDApy.representation.Values
	FDApy.representation.FunctionalData


Representing Argvals and Values
===============================

Functional data representations are based on two main components: the arguments of the functions and the values of the functions. The package provides several classes to represent these components, depending on the type of data to represent. 

.. autosummary::
	:toctree: autosummary

	FDApy.representation.DenseArgvals
	FDApy.representation.DenseValues


.. autosummary::
	:toctree: autosummary

	FDApy.representation.IrregularArgvals
	FDApy.representation.IrregularValues


Univariate Functional Data
==========================

Univariate functional data are realizations of a random process:

.. math::
	X: \mathcal{T} \subset \mathbb{R}^d \rightarrow \mathbb{R}.

The package provides two representations of univariate functional data: grid representation and basis representation.


Grid representation
-------------------

Univariate functional data can be represented as a set of values on a grid. The :class:`GridFunctionalData` class is the abstract class to represent univariate functional data on a grid. The package provides two implementations of this class: :class:`DenseFunctionalData` and :class:`IrregularFunctionalData`. The class :class:`DenseFunctionalData` represents functional data of arbitrary dimension (one for curves, two for images, etc.) on a common set of sampling points, while the class :class:`IrregularFunctionalData` represents functional data of arbitrary dimension sampled on different sets of points (the number and location of the sampling points vary between functional observations).


.. autosummary::
	:toctree: autosummary

	FDApy.representation.GridFunctionalData
	FDApy.representation.DenseFunctionalData
	FDApy.representation.IrregularFunctionalData


Basis representation
--------------------

The basis representation of univariate functional data consists of a linear combination of basis functions.

.. autosummary::
	:toctree: autosummary

	FDApy.representation.BasisFunctionalData


Multivariate Functional Data
============================

Multivariate functional data are realizations of a multivariate random process. Multivariate functional data objects are vectors of univariate functional data objects, eventually defined on different domains. The class :class:`MultivariateFunctionalData` allows for the combination of different types of functional data objects (:class:`DenseFunctionalData`, :class:`IrregularFunctionalData`, and :class:`BasisFunctionalData`). It is also possible to mix unidimensional data (curves) with multidimensional data (images, surfaces, etc.).

.. autosummary::
	:toctree: autosummary

	FDApy.representation.MultivariateFunctionalData


Basis
=====

The package provides two classes to represent basis of functions. The class :class:`Basis` represents a basis of functions, while the class :class:`MultivariateBasis` represents a multivariate basis of functions. Currently, the available bases are: Fourier basis, B-spline basis, Legendre basis and Wiener basis. The user may also define custom bases.

.. autosummary::
	:toctree: autosummary

	FDApy.representation.Basis
	FDApy.representation.MultivariateBasis


Iterators
=========

The package provides several iterators to handle functional data objects. These iterators allow for the iteration over the functional data objects (e.g. `for` loops, list comprehensions, etc.).

.. autosummary::
	:toctree: autosummary

	FDApy.representation.DenseFunctionalDataIterator
	FDApy.representation.IrregularFunctionalDataIterator
	FDApy.representation.BasisFunctionalDataIterator
