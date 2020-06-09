.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_multivariate_fpca.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_multivariate_fpca.py:


Multivariate Functional Principal Components Analysis
=====================================================

This notebook shows how to perform an multivariate functional principal
components analysis on an example dataset.


.. code-block:: default


    # Author: Steven Golovkine <steven_golovkine@icloud.com>
    # License: MIT

    # shinx_gallery_thumbnail_number = 2

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from FDApy.univariate_functional import UnivariateFunctionalData
    from FDApy.multivariate_functional import MultivariateFunctionalData
    from FDApy.fpca import MFPCA
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








Estimate the covariance for each of the components of the multivariate
functional data.


.. code-block:: default

    monthlyPrec.covariance()
    dailyTemp.covariance()








Perform a multivariate functional PCA and explore the results.


.. code-block:: default


    # Perform multivariate FPCA
    mfpca = MFPCA(n_components=[0.99, 0.95], method='NumInt')
    mfpca.fit(canadWeather)

    # Plot the results of the FPCA (eigenfunctions)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mfpca.basis_[0])
    plt.title('Eigenfunctions for dailyTemp')
    plt.subplot(1, 2, 2)
    plt.plot(mfpca.basis_[1])
    plt.title('Eigenfunctions for monthlyPrec')
    plt.tight_layout()




.. image:: /auto_examples/images/sphx_glr_plot_multivariate_fpca_001.png
    :alt: Eigenfunctions for dailyTemp, Eigenfunctions for monthlyPrec
    :class: sphx-glr-single-img





Compute the scores of the dailyTemp data into the eigenfunctions basis using
numerical integration.


.. code-block:: default


    # Compute the scores
    canadWeather_proj = mfpca.transform(canadWeather)

    # Plot the projection of the data onto the eigenfunctions
    pd.plotting.scatter_matrix(pd.DataFrame(canadWeather_proj), diagonal='kde')




.. image:: /auto_examples/images/sphx_glr_plot_multivariate_fpca_002.png
    :alt: plot multivariate fpca
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f247920ab00>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476b9db70>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476b51dd8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476b16080>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476ac82e8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476afb550>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2476ab07b8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476a639e8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476a63a58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24769ccef0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f247698e198>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24769c0400>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2476975668>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24769278d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24768dbb38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476892da0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476848fd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24768092b0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f247683a518>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24767ef780>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476bf7978>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476a719b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f247693b278>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2477187c50>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2477143780>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2477127240>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f247712d748>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f247672f828>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24766e4a90>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476697cf8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f247664ff60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f247660e208>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2476642470>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24765f66d8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f24765a9940>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f247655eba8>]],
          dtype=object)



Then, we can test if the reconstruction of the data is good.


.. code-block:: default


    # Test if the reconstruction is good.
    canadWheather_reconst = mfpca.inverse_transform(canadWeather_proj)

    # Plot the reconstructed curves
    fig, ax = plot(canadWheather_reconst,
                   main=['Daily temperature', 'Monthly precipitation'],
                   xlab=['Day', 'Month'],
                   ylab=['Temperature', 'Precipitation'])



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_multivariate_fpca_003.png
          :alt: Daily temperature
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_multivariate_fpca_004.png
          :alt: Monthly precipitation
          :class: sphx-glr-multi-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  2.435 seconds)


.. _sphx_glr_download_auto_examples_plot_multivariate_fpca.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_multivariate_fpca.py <plot_multivariate_fpca.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_multivariate_fpca.ipynb <plot_multivariate_fpca.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
