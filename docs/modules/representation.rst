=================================
Representation of functional data
=================================

The first step is the analysis of functional data is to represent them in a suitable way. The package provides several classes to represent functional data, depending on the type of data to represent. The package provides classes to represent univariate functional data, irregular functional data, multivariate functional data, and functional data in a basis representation.


Generic representation
======================

Representations of functional data are based on three abstract classes. The first one, :class:`Argvals`, represents the arguments of the functions. The second one, :class:`Values`, represents the values of the functions. The third one, :class:`FunctionalData`, represents the functional data object, which is a pair of :class:`Argvals` and :class:`Values`. The package provides several implementations of these classes, depending on the type of functional data to represent.

.. autosummary::
	:toctree: autosummary

	FDApy.representation.Argvals
	FDApy.representation.Values
	FDApy.representation.FunctionalData


Representing Argvals and Values
===============================

.. autosummary::
	:toctree: autosummary

	FDApy.representation.DenseArgvals
	FDApy.representation.IrregularArgvals

.. autosummary::
	:toctree: autosummary

	FDApy.representation.DenseValues
	FDApy.representation.IrregularValues


Univariate Functional Data
==========================

Blablabla

Grid representation
-------------------

.. autosummary::
	:toctree: autosummary

	FDApy.representation.GridFunctionalData
	FDApy.representation.DenseFunctionalData
	FDApy.representation.IrregularFunctionalData


Basis representation
--------------------

.. autosummary::
	:toctree: autosummary

	FDApy.representation.BasisFunctionalData


Multivariate Functional Data
============================

.. autosummary::
	:toctree: autosummary

	FDApy.representation.MultivariateFunctionalData

Iterators
=========

.. autosummary::
	:toctree: autosummary

	FDApy.representation.DenseFunctionalDataIterator
	FDApy.representation.IrregularFunctionalDataIterator
	FDApy.representation.BasisFunctionalDataIterator


Basis
=====

.. autosummary::
	:toctree: autosummary

	FDApy.representation.Basis
	FDApy.representation.MultivariateBasis
