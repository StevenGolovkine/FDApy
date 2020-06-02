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









.. code-block:: default

    print(monthlyPrec.argvals)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])]




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
    plt.show()



.. rst-class:: sphx-glr-script-out


.. code-block:: pytb

    Traceback (most recent call last):
      File "/home/steven/.ve/FDApy/lib/python3.7/site-packages/sphinx_gallery/gen_gallery.py", line 159, in call_memory
        return 0., func()
      File "/home/steven/.ve/FDApy/lib/python3.7/site-packages/sphinx_gallery/gen_rst.py", line 466, in __call__
        exec(self.code, self.fake_main.__dict__)
      File "/home/steven/Documents/workspace/FDApy/examples/plot_multivariate_fpca.py", line 57, in <module>
        mfpca.fit(canadWeather)
      File "/home/steven/Documents/workspace/FDApy/FDApy/fpca.py", line 284, in fit
        self._fit(X)
      File "/home/steven/Documents/workspace/FDApy/FDApy/fpca.py", line 291, in _fit
        self._fit_multi(X, self.n_components, self.method)
      File "/home/steven/Documents/workspace/FDApy/FDApy/fpca.py", line 312, in _fit_multi
        ufpca.append(uni.fit(function))
      File "/home/steven/Documents/workspace/FDApy/FDApy/fpca.py", line 69, in fit
        self._fit(X)
      File "/home/steven/Documents/workspace/FDApy/FDApy/fpca.py", line 75, in _fit
        self._fit_uni(X)
      File "/home/steven/Documents/workspace/FDApy/FDApy/fpca.py", line 110, in _fit_uni
        X.covariance(smooth=True, **self.smoothing_parameters)
      File "/home/steven/Documents/workspace/FDApy/FDApy/univariate_functional.py", line 476, in covariance
        self.mean(smooth, method, **kwargs)
      File "/home/steven/Documents/workspace/FDApy/FDApy/univariate_functional.py", line 414, in mean
        lp.fit(self.argvals, mean_)
      File "/home/steven/Documents/workspace/FDApy/FDApy/local_polynomial.py", line 288, in fit
        bandwidth)])
      File "/home/steven/Documents/workspace/FDApy/FDApy/local_polynomial.py", line 286, in <listcomp>
        for (i, j, h) in zip(x0.T,
      File "/home/steven/Documents/workspace/FDApy/FDApy/local_polynomial.py", line 186, in _loc_poly
        K = _compute_kernel(x=x, x0=x0, h=h, kernel=kernel)
      File "/home/steven/Documents/workspace/FDApy/FDApy/local_polynomial.py", line 127, in _compute_kernel
        if x.ndim != np.size(x0):
    AttributeError: 'list' object has no attribute 'ndim'




Compute the scores of the dailyTemp data into the eigenfunctions basis using
numerical integration.


.. code-block:: default


    # Compute the scores
    canadWeather_proj = mfpca.transform(canadWeather)

    # Plot the projection of the data onto the eigenfunctions
    pd.plotting.scatter_matrix(pd.DataFrame(canadWeather_proj), diagonal='kde')
    plt.show()


Then, we can test if the reconstruction of the data is good.


.. code-block:: default


    # Test if the reconstruction is good.
    canadWheather_reconst = mfpca.inverse_transform(canadWeather_proj)

    # Plot the reconstructed curves
    fig, ax = plot(canadWheather_reconst,
                   main=['Daily temperature', 'Monthly precipitation'],
                   xlab=['Day', 'Month'],
                   ylab=['Temperature', 'Precipitation'])
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.442 seconds)


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
