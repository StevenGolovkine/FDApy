.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_simulation.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_simulation.py:


Functional data simulation
==========================

This notebook shows how to simulate functional data according with different
basis.


.. code-block:: default


    # Author: Steven Golovkine <steven_golovkine@icloud.com>
    # License: MIT

    # shinx_gallery_thumbnail_number = 2

    import numpy as np

    from FDApy.basis import Basis, Brownian, basis_legendre, basis_wiener
    from FDApy.plot import plot








We will define a Legendre polynomial basis using the method
:func:`~FDApy.basis.basis_legendre`.



.. code-block:: default


    argvals = np.linspace(-1, 1, 1000)
    LP = basis_legendre(K=5, argvals=argvals, norm=True)

    # Plot the basis
    fig, ax = plot(LP, main='Legendre basis', xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_001.png
    :alt: Legendre basis
    :class: sphx-glr-single-img





Next, we will define a Wiener basis using the method
:func:`~FDApy.basis.basis_wiener`.



.. code-block:: default


    argvals = np.linspace(-1, 1, 1000)
    WP = basis_wiener(K=5, argvals=argvals, norm=True)

    # Plot the basis
    fig, ax = plot(WP, main='Wiener basis', xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_002.png
    :alt: Wiener basis
    :class: sphx-glr-single-img





Now, we will simulate some curves data according to diverse basis with
different eigenvalues decay.


Legendre basis and exponential eigenvalues decay


.. code-block:: default

    sim = Basis(N=100, M=50, basis='legendre', K=5,
                eigenvalues='exponential', norm=True)
    sim.new()

    # Plot some simulations
    fig, ax = plot(sim.obs_, main='Simulation', xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_003.png
    :alt: Simulation
    :class: sphx-glr-single-img





Legendre basis and linear eigenvalues decay


.. code-block:: default

    sim = Basis(N=100, M=50, basis='legendre', K=5,
                eigenvalues='linear', norm=True)
    sim.new()

    # Plot some simulations
    fig, ax = plot(sim.obs_, main='Simulation', xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_004.png
    :alt: Simulation
    :class: sphx-glr-single-img





Wiener basis and Wiener eigenvalues decay


.. code-block:: default

    sim = Basis(N=100, M=50, basis='wiener', K=5,
                eigenvalues='wiener', norm=True)
    sim.new()

    # Plot some simulations
    fig, ax = plot(sim.obs_, main='Simulation', xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_005.png
    :alt: Simulation
    :class: sphx-glr-single-img





Wiener basis and user-set eigenvalues


.. code-block:: default

    sim = Basis(N=100, M=50, basis='wiener', K=3,
                eigenvalues=[100, 25, 5], norm=True)
    sim.new()

    # Plot some simulations
    fig, ax = plot(sim.obs_, main='Simulation', xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_006.png
    :alt: Simulation
    :class: sphx-glr-single-img





We can also add some noise to the data.

First, we consider homoscedastic noise. Thus, we add realizations of the
random variable :math:`\varepsilon \sim \mathcal{N}(0, \sigma^2)` to the
data.



.. code-block:: default


    # Add some noise to the simulation.
    sim.add_noise(5)

    # Plot the noisy simulations
    fig, ax = plot(sim.noisy_obs_,
                   main='Noisy simulation',
                   xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_007.png
    :alt: Noisy simulation
    :class: sphx-glr-single-img





Second, we may add heteroscedatic noise to the data. In this case, the
quantity added to the data is defined as realisations of the random variable
:math:`\varepsilon \sim \mathcal{N}(0, \sigma^2(t))`.



.. code-block:: default


    # Add some heteroscedastic noise to the simulation
    sim.add_noise(sd_function=lambda x: np.sqrt(np.abs(x) + 1))

    # Plot the heteroscedastic noisy simulations
    fig, ax = plot(sim.noisy_obs_,
                   main='Noisy heteroscedastic simulation',
                   xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_008.png
    :alt: Noisy heteroscedastic simulation
    :class: sphx-glr-single-img





We can also simulate Brownian motion and some of processes derived from it,
such as Geometric Brownian motion and Fractional Brownian motion.


Simulate some standard brownian motions.


.. code-block:: default

    sim = Brownian(N=100, M=50, brownian_type='standard')
    sim.new(x0=0)

    # Plot some simulations
    fig, ax = plot(sim.obs_,
                   main='Standard Brownian motion',
                   xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_009.png
    :alt: Standard Brownian motion
    :class: sphx-glr-single-img





Simulate some geometric brownian motions.


.. code-block:: default

    sim = Brownian(N=100, M=50, brownian_type='geometric')
    sim.new(x0=1, mu=5, sigma=1)

    # Plot some simulations
    fig, ax = plot(sim.obs_,
                   main='Geometric Brownian motion',
                   xlab='Sampling points')




.. image:: /auto_examples/images/sphx_glr_plot_simulation_010.png
    :alt: Geometric Brownian motion
    :class: sphx-glr-single-img





Simulate some fractional brownian motions.


.. code-block:: default

    sim = Brownian(N=100, M=50, brownian_type='fractional')
    sim.new(H=0.7)

    # Plot some simulations
    fig, ax = plot(sim.obs_,
                   main='Fractional Brownian motion',
                   xlab='Sampling points')



.. image:: /auto_examples/images/sphx_glr_plot_simulation_011.png
    :alt: Fractional Brownian motion
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  2.168 seconds)


.. _sphx_glr_download_auto_examples_plot_simulation.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_simulation.py <plot_simulation.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_simulation.ipynb <plot_simulation.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
