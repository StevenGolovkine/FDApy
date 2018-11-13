#!/usr/bin/python3.7
# -*-coding:utf8 -*

import numpy as np

import FDApy

def plot(data, obs=None, main=""):
	"""Generic plot function for univariate, irregular and multivariate functional data.
	
	Parameters
	----------
	data : UnivariateFunctionalData, IrregularFunctionalData or MultivariateFunctionalData
		The object to plot.
	obs : int
		The observation of the object to plot.
	main : str
		Title of the graph.

	Return
	------
	fig, ax : elements for plotting using matplotlib

	"""
	if isinstance(data, 
		FDApy.univariate_functional.UnivariateFunctionalData):
		fig, ax = _plot_univariate(data, obs, main)
	elif isinstance(data, 
		FDApy.irregular_functional.IrregularFunctionalData):
		fig, ax = _plot_irregular(data, obs, main)
	elif isinstance(data, 
		FDApy.multivariate_functional.MultivariateFunctionalData):
		fig, ax = _plot_multivariate(data, obs, main)
	else:
		raise ValueError('data has to element of FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData or FDApy.multivariate_functional.MultivariateFunctionalData!')

	return fig, ax

def _plot_univariate(data, obs=None, main=""):
	"""Plot univariate functional data.
	
	Parameters
	----------
	data : UnivariateFunctionalData.
		The object to plot.
	obs : int
		The observation of the object to plot.
	main : str
		Title of the graph.

	Return
	------
	fig, ax : elements for ploting using ...
	"""
	raise NotImplemented('Not implemented')

def _plot_irregular(data, obs=None, main=""):
	"""Plot irregular functional data.
	
	Parameters
	----------
	data : IrregularFunctionalData.
		The object to plot.
	obs : int
		The observation of the object to plot.
	main : str
		Title of the graph.

	Return
	------
	fig, ax : elements for ploting using ...
	"""
	raise NotImplemented('Not implemented')

def _plot_multivariate(data, obs=None, main=""):
	"""Plot multivariate functional data.
	
	Parameters
	----------
	data : MultivariateFunctionalData.
		The object to plot.
	obs : int
		The observation of the object to plot.
	main : str
		Title of the graph.

	Return
	------
	fig, ax : elements for ploting using ...
	"""
	raise NotImplemented('Not implemented')

