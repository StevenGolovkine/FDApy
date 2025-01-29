=============
Preprocessing
=============

Before the analysis, functional data need to be preprocessed. The package provides several classes to preprocess functional data, including smoothing and dimension reduction.

Smoothing
=========

Observations of functional data are often noisy. Smoothing methods are used to remove the noise and extract the underlying patterns. The package provides two classes to smooth functional data: :class:`LocalPolynomial` and :class:`PSplines`.

.. autosummary::
    :toctree: autosummary

    FDApy.preprocessing.LocalPolynomial
    FDApy.preprocessing.PSplines


Dimension reduction
===================

Due to the infinite-dimensional nature of functional data, dimension reduction techniques are important tools in functional data analysis. They are used to extract the most relevant information. The package provides three classes to reduce the dimension of functional data: :class:`UFPCA`, :class:`MFPCA`, and :class:`FCPTPA`.

.. autosummary::
    :toctree: autosummary

    FDApy.preprocessing.UFPCA
    FDApy.preprocessing.MFPCA
    FDApy.preprocessing.FCPTPA

