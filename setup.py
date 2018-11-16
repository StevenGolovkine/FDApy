from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='FDApy',
	version='0.1',
	description='Python package for Functional Data Analysis',
	long_description='',
	classifiers=[
		'Programming Language :: Python :: 3.7',
		'Topic :: Scientific/Engineering :: Mathematics',
	],
	keywords='functional data analysis',
	url='https://github.com/StevenGolovkine/FDApy',
	author='Steven Golovkine',
	author_email='steven_golovkine@icloud.com',
	license='MIT',
	packages=['FDApy'],
	install_require=[
		'ggplot',
		'itertools',
		'numpy',
		'pandas',
		'sklearn'
	],
	test_suite='nose.collector',
	tests_require=['nose'],
	include_package_data=True,
	zip_safe=False)