from setuptools import setup
from Cython.Build import cythonize


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='FDApy',
      version='0.3',
      description='Python package for Functional Data Analysis',
      long_description='',
      classifiers=[
                  'Programming Language :: Python :: 3.7',
                  'Topic :: Scientific/Engineering :: Mathematics'],
      keywords='functional data analysis',
      url='https://github.com/StevenGolovkine/FDApy',
      author='Steven Golovkine',
      author_email='steven_golovkine@icloud.com',
      license='MIT',
      packages=['FDApy'],
      install_require=['ggplot',
                       'itertools',
                       'numpy',
                       'pandas',
                       'pygam',
                       'sklearn'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      ext_modules=cythonize("FDApy/src/sigma.pyx"),
      zip_safe=False)
