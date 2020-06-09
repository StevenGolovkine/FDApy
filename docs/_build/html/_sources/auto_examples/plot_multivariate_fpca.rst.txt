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


    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7faddf8007f0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadded4fa58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf1bdda0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddedcc278>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde88e2b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf826518>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7faddf825780>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf16a9b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf16aa20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf725eb8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf7bd160>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf0ba3c8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fadded70630>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadded86898>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf100b00>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde7dcd68>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde792fd0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde750278>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fadde7044e0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde591748>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde5c59b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde4fbc18>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8e94e80>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8e56128>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8e07390>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8dbd5f8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8d70860>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8da6828>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8d59ba8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8d0cf28>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8cce2e8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf1bdcf8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7faddf12f550>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde7e9ba8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadd8e74668>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7fadde3bbc88>]],
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

   **Total running time of the script:** ( 0 minutes  2.076 seconds)


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
