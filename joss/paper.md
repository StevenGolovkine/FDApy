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
    affiliation: 1
affiliations:
 - name: MACSI, Department of Mathematics and Statistics, University of Limerick, Limerick, Ireland
   index: 1
   ror: 00a0n9e72
date: 28 October 2024
bibliography: paper.bib

---

# Summary

Functional data analysis (FDA) is a statistical methodology for analyzing data that can be characterized as functions. These functions could represent measurements taken over time, space, frequency, probability, etc. The goal of FDA is to extract meaningful information from these functions and to model their behavior. See, e.g., @ramsayFunctionalDataAnalysis2005,@horvathInferenceFunctionalData2012a, and @kokoszkaIntroductionFunctionalData2017 for some references on FDA. FDA has been successfully applied in different contexts, such as identifying patterns of movements in sport biomechanics [@warmenhovenBivariateFunctionalPrincipal2019], analyzing changes in brain activity in neuroscience [@songSparseMultivariateFunctional2022], fault detection of batch processes [@wangFaultDetectionBatch2015] or in autonomous driving [@golovkineClusteringMultivariateFunctional2022]. In this paper, we introduce `FDApy`, a library developed for the FDA community and Python users, designed to facilitate the manipulation and processing of (multivariate) functional data.


# Statement of need

In order to apply FDA to real datasets, there is a need for appropriate softwares with up-to-date methodological implementation and easy addition of new theoretical developments. The seminal R package for FDA is `fda` [@ramsayFdaFunctionalData2023], based on work cited in @ramsayFunctionalDataAnalysis2005 and @ramsayFunctionalDataAnalysis2009. Most of the R packages that implement FDA methods are highly specialized and are built upon `fda`. For example, one may cite `FDboost` [@brockhausBoostingFunctionalRegression2020] and `refund` [@goldsmithRefundRegressionFunctional2023] for regression and classification, `funFEM` [@bouveyronFunFEMClusteringDiscriminative2021] and `funLBM` [@bouveyronFunLBMModelBasedCoClustering2022] for clustering or `fdasrvf` [@tuckerFdasrvfElasticFunctional2023] for functional data registration. For most packages, the functional data are however restricted to univariate functional data that are well described by their coefficients in a given basis of functions. The `funData` package [@happ-kurzObjectOrientedSoftwareFunctional2020] aims to provide a unified framework to handle univariate and multivariate functional data defined on different dimensional domains. Sparse functional data are also considered. The `MFPCA` [@happ-kurzMFPCAMultivariateFunctional2022] package, built on top of the `funData` package, implements multivariate functional principal components analysis (MFPCA) for data defined on different dimensional domains [@happMultivariateFunctionalPrincipal2018]. Consider looking at the CRAN webpage\footnote{\url{https://cran.r-project.org/web/views/FunctionalData.html}} on functional data to have a complete overview of the R packages.

Concerning the Python community, there are only few packages that are related to FDA. One may cite `sktime` [@loningSktimeSktimeV02022] and `tslearn` [@tavenardTslearnMachineLearning2020] that provide tools for the analysis of time series as a `scikit-learn` compatible API. They implement specific time series methods such as DTW-based ones or shapelets learning. The only one that develops specific methods for FDA is `scikit-fda` [@ramos-carrenoScikitfdaPythonPackage2024]. In particular, it implements diverse registration techniques as well as statistical data depths for functional data. However, most of the methods are for one-dimensional data and, in most cases, they only accept multivariate functional data defined on the same domain.

`FDApy` supports the analysis of diverse types of functional data (densely or irregularly sampled, multivariate and multidimensional), represented over a grid of points or using a basis of functions. It implements dimension reduction techniques and smoothing functionalities. A large simulation toolbox, based on basis decomposition, is provided. By providing a flexible and robust toolset for functional data analysis, it aims to support researchers and practitioners in uncovering insights from complex functional datasets.

`FDApy` was used in @golovkineClusteringMultivariateFunctional2022, @yoshidaDetectingDifferencesGait2022, @golovkineUseGramMatrix2023 and @nguyenLearningDomainspecificCameras2024 and is also presented in the author's doctoral dissertation.


# Code Quality and Documentation


`FDApy` is hosted on GitHub\footnote{\url{https://github.com/StevenGolovkine/FDApy}}. Examples and API documentation are available on the platform Read the Docs\footnote{\url{https://fdapy.readthedocs.io}}. We provide installation guides, algorithm introductions, and examples of using the package. The package is available on Linux, macOS and Windows for Python $3.9-3.11$. It can be installed with `pip install FDApy`. 

To ensure high code quality, all implementations adhere to the `PEP8` code style [@vanrossumPEP8StyleGuide2001], enforced by `flake8`, the code formatter `black` and the static analyzer `prospector`. The documentation is provided through docstrings using the `NumPy` conventions and build using `Sphinx`. The code is accompanied by unit tests covering $94\%$ of the lines that are automatically executed in a continuous integration workflow upon commits.

# Acknowledgements

Steven Golovkine wishes to thank Groupe Renault and the ANRT (French National Association for Research and Technology) for their financial support via the CIFRE convention No. 2017/1116. Steven Golovkine is partially supported by Science Foundation Ireland under Grant No. 19/FFP/7002 and co-funded under the European Regional Development Fund.

# References