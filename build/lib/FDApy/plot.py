#!/usr/bin/python3.7
# -*-coding:utf8 -*

import matplotlib.pyplot as plt
import numpy as np

import FDApy

def plot(data, main="", xlab="", ylab=""):
	"""Generic plot function for univariate, irregular and multivariate functional data.
	
	Parameters
	----------
	data : UnivariateFunctionalData, IrregularFunctionalData or MultivariateFunctionalData
		The object to plot.
	main : str
		Title of the graph.
	xlab : str or list of str
		Label of the X axis.
	ylab : str or list of str
		Label of the Y axis.

	Return
	------
	fig, ax : elements for plotting using matplotlib

	"""
	if isinstance(data, 
		FDApy.univariate_functional.UnivariateFunctionalData):
		fig, ax = _plot_univariate(data, main, xlab, ylab)
	elif isinstance(data, 
		FDApy.irregular_functional.IrregularFunctionalData):
		fig, ax = _plot_irregular(data, main, xlab, ylab)
	elif isinstance(data, 
		FDApy.multivariate_functional.MultivariateFunctionalData):
		fig, ax = _plot_multivariate(data, main, xlab, ylab)
	else:
		raise ValueError('data has to be elements of FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData or FDApy.multivariate_functional.MultivariateFunctionalData!')

	return fig, ax

def _plot_univariate(data, main="", xlab="", ylab=""):
	"""Plot univariate functional data.
	
	Parameters
	----------
	data : UnivariateFunctionalData.
		The object to plot.
	main : str
		Title of the graph.
	xlab : str
		Label of the X axis.
	ylab : str
		Label of the Y axis.

	Return
	------
	fig, ax : elements for ploting using matplotlib
	"""
	fig, ax = plt.subplots(1, 1)

	ax.set_title(main)
	ax.set_xlabel(xlab)
	ax.set_ylabel(ylab)
	for obs in data.values:
		ax.plot(data.argvals[0], obs)

	return fig, ax

def _plot_irregular(data, main="", xlab="", ylab=""):
	"""Plot irregular functional data.
	
	Parameters
	----------
	data : IrregularFunctionalData.
		The object to plot.
	main : str
		Title of the graph.
	xlab : str
		Label of the X axis.
	ylab : str
		Label of the Y axis.

	Return
	------
	fig, ax : elements for ploting using matplotlib
	"""
	raise NotImplementedError('Not implemented')

def _plot_multivariate(data, main="", xlab="", ylab=""):
	"""Plot multivariate functional data.
	
	Parameters
	----------
	data : MultivariateFunctionalData.
		The object to plot.
	main : str or list of str
		Title of the graph.
	xlab : str or list of str
		Label of the X axis.
	ylab : str or list of str
		Label of the Y axis.

	Return
	------
	figs, axex : elements for ploting using matplotlib
	"""
	nFunc = data.nFunctions()

	if isinstance(main, list) and len(main) != nFunc:
		raise ValueError('The parameter `main` has not the right length!')
	if isinstance(xlab, list) and len(xlab) != nFunc:
		raise ValueError('The parameter `xlab` has not the right length!')
	if isinstance(ylab, list) and len(ylab) != nFunc:
		raise ValueError('The parameter `ylab` has not the right length!')

	if isinstance(main, str):
		main = [main for i in range(nFunc)]
	if isinstance(xlab, str):
		xlab = [xlab for i in range(nFunc)]
	if isinstance(ylab, str):
		ylab = [ylab for i in range(nFunc)]
		
	figs = []
	axes = []
	
	for i, func in enumerate(data):

		if isinstance(
				func, FDApy.univariate_functional.UnivariateFunctionalData):
			fig, ax = _plot_univariate(func, main[i], xlab[i], ylab[i])
		elif isinstance(
				func, FDApy.irregular_functional.IrregularFunctionalData):
			fig, ax = _plot_irregular(func, main[i], xlab[i], ylab[i])
		else:
			raise ValueError('data has to be elements of FDApy.univariate_functional.UnivariateFunctionalData or FDApy.irregular_functional.IrregularFunctionalData!')
		figs.append(fig)
		axes.append(ax)
	return figs, axes


