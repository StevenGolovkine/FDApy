#!/usr/bin/env python
"""FDApy: a Python package to analyze functional data.

Functional Data Analysis, usually referred as FDA, concerns the field of
Statistics that deals with discrete observations of continuous d-dimensional
functions.

This package provide modules for the analysis of such data. It includes
methods for different dimensional data as well as irregularly sampled
functional data. An implementation of (multivariate) functional principal
component analysis is also given. Moreover, a simulation toolbox is provided.
It might be used to simulate different clusters of functional data.

Check out the `documentation <https://fdapy.readthedocs.io/en/stable/>`_ for
more complete information on the available features within the package.
"""
from setuptools import find_packages, setup


DOCLINES = (__doc__ or '').split('\n')

setup(
    name='FDApy',
    version='1.0.0',
    python_requires='>= 3.9, <4',
    description=DOCLINES[1],
    long_description='\n'.join(DOCLINES[3:]),
    long_description_content_type='text/x-rst',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords='functional data analysis',
    url='https://github.com/StevenGolovkine/FDApy',
    author='Steven Golovkine',
    author_email='steven_golovkine@icloud.com',
    license='MIT',
    package_dir={'FDApy': 'FDApy'},
    packages=find_packages(),
    install_requires=[
        'csaps>=1.1.0',
        'ggplot>=0.11.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.2.0',
        "scipy>=1.10.0"
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False
)
