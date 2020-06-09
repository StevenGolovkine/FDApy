.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_canadian_weather.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_canadian_weather.py:


Canadian weather analysis
=========================

This notebook shows how to deal with univariate and multivariate functional
data by analyzing the canadian weather dataset.


.. code-block:: default


    # Author: Steven Golovkine <steven_golovkine@icloud.com>
    # License: MIT

    # shinx_gallery_thumbnail_number = 2

    import numpy as np
    import pandas as pd

    from FDApy.univariate_functional import UnivariateFunctionalData
    from FDApy.multivariate_functional import MultivariateFunctionalData
    from FDApy.plot import plot








Load the data into Pandas dataframe


.. code-block:: default

    precipitation = pd.read_csv('./data/canadian_precipitation_monthly.csv',
                                index_col=0)
    temperature = pd.read_csv('./data/canadian_temperature_daily.csv',
                              index_col=0)








Create univariate functional data for the precipitation and temperature
dataset. Then, we will combine them to form a multivariate functional
dataset.


.. code-block:: default


    # Create univariate functional data for the precipitation data
    argvals = pd.factorize(precipitation.columns)[0]
    values = np.array(precipitation)
    monthlyPrec = UnivariateFunctionalData(argvals, values)

    # Create univariate functional data for the daily temperature data.
    argvals = pd.factorize(temperature.columns)[0]
    values = np.array(temperature) / 4
    dailyTemp = UnivariateFunctionalData(argvals, values)

    # Create multivariate functional data for the Canadian weather data.
    canadWeather = MultivariateFunctionalData([dailyTemp, monthlyPrec])








Print out an univariate functional data object.


.. code-block:: default


    # Print univariate functional data
    print(dailyTemp)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Univariate Functional data objects with 35 observations of 1-dimensional support
    argvals:
            0, 1, ... , 364 (365 sampling points)
    values:
            array of size (35, 365)




Print out a multivariate functional data object.


.. code-block:: default


    # Print multivariate functional data
    print(canadWeather)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Multivariate Functional data objects with 2 funtions:
    - Univariate Functional data objects with 35 observations of 1-dimensional support
    argvals:
            0, 1, ... , 364 (365 sampling points)
    values:
            array of size (35, 365)
    - Univariate Functional data objects with 35 observations of 1-dimensional support
    argvals:
            0, 1, ... , 11  (12 sampling points)
    values:
            array of size (35, 12)





We can plot the data.


.. code-block:: default


    # Plot the multivariate functional data
    fig, ax = plot(canadWeather,
                   main=['Daily temperature', 'Monthly precipitation'],
                   xlab=['Day', 'Month'],
                   ylab=['Temperature', 'Precipitation'])




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_canadian_weather_001.png
          :alt: Daily temperature
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_canadian_weather_002.png
          :alt: Monthly precipitation
          :class: sphx-glr-multi-img





The attributs of the univariate functional data classes can easily be
accessed.

The sampling points of the data can easily be accessed.


.. code-block:: default


    # Accessing the argvals of the object
    print(monthlyPrec.argvals)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])]




The number of observations within the data are obtained using the function
:func:`~FDApy.univariate_functional.UnivariateFunctional.nObs`.


.. code-block:: default


    # Get the number of observations for the object
    print(monthlyPrec.nObs())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    35




The number of sampling points per observation is given by the function
:func:`~FDApy.univariate_functional.UnivariateFunctional.nObsPoint`.


.. code-block:: default


    # Retrieve the number of sampling points for the object
    print(monthlyPrec.nObsPoint())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [12]




The dimension of the data is given by the function
:func:`~FDApy.univariate_functional.UnivariateFunctional.dimension`.


.. code-block:: default


    # Get the dimension of the domain of the observations
    print(monthlyPrec.dimension())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1




The extraction of observations is also easily done.


.. code-block:: default


    # Extract observations from the object
    print(monthlyPrec[3:6])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Univariate Functional data objects with 3 observations of 1-dimensional support
    argvals:
            0, 1, ... , 11  (12 sampling points)
    values:
            array of size (3, 12)




In a same way, the attributs of the multivariate functional data classes
can also be easily accessed.


.. code-block:: default


    # Number of sampling points for the object
    canadWeather.nObsPoint()

    # Extract functions from MultivariateFunctionalData
    print(canadWeather[0])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Univariate Functional data objects with 35 observations of 1-dimensional support
    argvals:
            0, 1, ... , 364 (365 sampling points)
    values:
            array of size (35, 365)




Compute the mean function for an univariate functional data object.


.. code-block:: default


    # Mean function of the monthly precipitation
    monthlyPrec.mean()

    # Plot the mean function of the monthly precipation
    fig, ax = plot(monthlyPrec.mean_,
                   main='Mean monthly precipitation',
                   xlab='Month',
                   ylab='Precipitation (mm)')




.. image:: /auto_examples/images/sphx_glr_plot_canadian_weather_003.png
    :alt: Mean monthly precipitation
    :class: sphx-glr-single-img





Compute the covariance surface for an univariate functional data object.


.. code-block:: default


    # Covariance function of the monthly precipitation
    monthlyPrec.covariance()

    # Plot the covariance function of the monthly precipitation
    fig, ax = plot(monthlyPrec.covariance_,
                   main='Covariance monthly precipitation',
                   xlab='Month',
                   ylab='Month')




.. image:: /auto_examples/images/sphx_glr_plot_canadian_weather_004.png
    :alt: Covariance monthly precipitation
    :class: sphx-glr-single-img





We can also compute a smoothed estimate of the mean function and the
covariance surface.


.. code-block:: default


    # Smoothing covariance of the daily temperature
    dailyTemp.covariance(smooth=True, method='GAM', bandwidth=20)

    # Plot the smooth covariance function of the daily temperature
    fig, ax = plot(dailyTemp.covariance_,
                   main='Covariance daily temperature',
                   xlab='Day',
                   ylab='Day')




.. image:: /auto_examples/images/sphx_glr_plot_canadian_weather_005.png
    :alt: Covariance daily temperature
    :class: sphx-glr-single-img





Instead of directly computing an estimation of the mean and covariance by
smoothing, we can smooth all the curve in an individual way.


.. code-block:: default


    # Smooth the data
    dailyTempSmooth = dailyTemp.smooth(t0=200, k0=17,
                                       points=dailyTemp.argvals[0],
                                       kernel='gaussian')

    # Plot the smooth data
    fig, ax = plot(dailyTempSmooth,
                   main='Daily temperature',
                   xlab='Day',
                   ylab='Temperature')



.. image:: /auto_examples/images/sphx_glr_plot_canadian_weather_006.png
    :alt: Daily temperature
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  9.186 seconds)


.. _sphx_glr_download_auto_examples_plot_canadian_weather.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_canadian_weather.py <plot_canadian_weather.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_canadian_weather.ipynb <plot_canadian_weather.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
