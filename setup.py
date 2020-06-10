#!/usr/bin/env python

import Cython.Build

from setuptools import Extension, find_packages, setup


def get_readme():
    with open('README.rst') as f:
        return f.read()


setup(name='FDApy',
      version='0.3.5',
      description='Python package for Functional Data Analysis',
      long_description=get_readme(),
      classifiers=[
                  'Programming Language :: Python :: 3.7',
                  'Topic :: Scientific/Engineering :: Mathematics'],
      keywords='functional data analysis',
      url='https://github.com/StevenGolovkine/FDApy',
      author='Steven Golovkine',
      author_email='steven_golovkine@icloud.com',
      license='MIT',
      cmdclass={'build_ext': Cython.Build.build_ext},
      package_dir={'FDApy': 'FDApy'},
      packages=find_packages(),
      install_requires=['ggplot',
                        'cython',
                        'numpy',
                        'pandas',
                        'pygam',
                        'sklearn'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      ext_modules=[Extension('FDApy.src.sigma',
                             sources=['FDApy/src/sigma.pyx'],
                             language='c++')],
      setup_requires=['cython'],
      zip_safe=False)
