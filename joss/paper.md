---
title: '\texttt{FDApy}: a Python package for functional data'
tags:
  - functional data analysis
  - multivariate functional data
  - open source
  - Python
authors:
  - name: Steven Golovkine
    orcid: 0000-0002-5994-2671
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: MACSI, Department of Mathematics and Statistics, University of Limerick, Limerick, Ireland
   index: 1
   ror: 00a0n9e72
date: 28 October 2024
bibliography: paper.bib

---

# Summary

Functional data analysis (FDA) is a statistical methodology for analyzing data that can be characterized as functions. These functions could represent measurements taken over time, space, frequency, probability, etc. The goal of FDA is to extract meaningful information from these functions and to model their behavior. See, e.g., @ramsayFunctionalDataAnalysis2005,@horvathInferenceFunctionalData2012a, and @kokoszkaIntroductionFunctionalData2017 for some references on FDA. FDA has been successfully applied in different context, such as identifying patterns of movements in sport biomechanics [@warmenhovenBivariateFunctionalPrincipal2019], analyzing changes in brain activity in neuroscience [@songSparseMultivariateFunctional2022], fault detection of batch processes [@wangFaultDetectionBatch2015] or in autonomous driving [@golovkineClusteringMultivariateFunctional2022].


# Statement of need

In order to apply FDA to real datasets, there is a need for appropriate softwares with up-to-date methodological implementation and easy addition of new theoretical developments. The seminal R package for FDA is \textsf{fda} [@ramsayFdaFunctionalData2023], based on work cited in @ramsayFunctionalDataAnalysis2005 and @ramsayFunctionalDataAnalysis2009. Most of the R packages that implements FDA methods are highly specialized and are built upon \textsf{fda}. For example, one may cite \textsf{FDboost} [@brockhausBoostingFunctionalRegression2020] and \textsf{refund} [@goldsmithRefundRegressionFunctional2023] for regression and classification, \textsf{funFEM} [@bouveyronFunFEMClusteringDiscriminative2021] and \textsf{funLBM} [@bouveyronFunLBMModelBasedCoClustering2022] for clustering or \textsf{fdasrvf} [@tuckerFdasrvfElasticFunctional2023] for functional data registration. For most packages, the functional data are however restricted to univariate functional data that are well described by their coefficients in a given basis of functions. The \textsf{funData} package [@happ-kurzObjectOrientedSoftwareFunctional2020] aims to provide a unified framework to handle univariate and multivariate functional data defined on different dimensional domains. Sparse functional data are also considered. The \textsf{MFPCA} [@happ-kurzMFPCAMultivariateFunctional2022] package, built on top of the \textsf{funData} package, implements multivariate functional principal components analysis (MFPCA) for data defined on different dimensional domains [@happMultivariateFunctionalPrincipal2018]. Consider looking at the CRAN webpage\footnote{\url{https://cran.r-project.org/web/views/FunctionalData.html}} on functional data to have a complete overview of the R packages.

Concerning the Python community, there are only few packages that are related to FDA. One may cite \textsf{sktime} [@loningSktimeSktimeV02022] and \textsf{tslearn} [@tavenardTslearnMachineLearning2020] that provide tools for the analysis of time series as a \textsf{scikit-learn} compatible API. They implement specific time series methods such as DTW-based ones or shapelets learning. The only one that develops specific methods for FDA is \textsf{scikit-fda} [@ramos-carrenoScikitfdaPythonPackage2023]. In particular, it implements diverse registration techniques as well as statistical data depths for functional data. However, most of the methods are for one-dimensional data and they only accept multivariate functional data defined on the same domain.

\textsf{FDApy} supports the analysis diverse types of functional data (densely or irregularly sampled, multivariate and multidimensional). It implements dimension reduction techniques. A large simulation toolbox, based on basis decomposition, is provided. By providing a flexible and robust toolset for functional data analysis, it aims to support researchers and practitioners in uncovering insights from complex functional datasets.

\texttt{FDApy} was used in @golovkineClusteringMultivariateFunctional2022 and @golovkineUseGramMatrix2023 and is also presented in the author's doctoral dissertation.


# Code Quality and Documentation


\texttt{FDApy} is hosted on GitHub\footnote{\url{https://github.com/StevenGolovkine/FDApy}}. Examples and API documentation are available on the platform Read the Docs\footnote{\url{https://fdapy.readthedocs.io}}. We provide installation guides, algorithm introductions, examples of using the package. The package is available on Linux, macOS and Windows for Python $3.9-3.11$. It can be installed with \textsf{pip install FDApy}. 

To ensure high code quality, all implementations adhere to the \texttt{PEP8} code style [@vanrossumPEP8StyleGuide2001], enforced by \texttt{flake8}, the code formatter \texttt{black} and the static analyzer \texttt{prospector}. The documentation is provided through docstrings using the \texttt{numpy} conventions and build using \texttt{Sphinx}. The code is accompanied by unit tests covering $94\%$ of the lines that are automatically executed in a continuous integration workflow upon commits.

# Acknowledgements

Steven Golovkine wish to thank Groupe Renault and the ANRT (French National Association for Research and Technology) for their financial support via the CIFRE convention No. 2017/1116. Steven Golovkine is partially supported by Science Foundation Ireland under Grant No. 19/FFP/7002 and co-funded under the European Regional Development Fund.

# References