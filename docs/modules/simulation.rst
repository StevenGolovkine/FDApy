==========
Simulation
==========

The package contains a full simulation toolbox to generate functional data. This toolbox can be used for the implementation, the test and the comparison of new methodologies.


Simulation
==========

The simulation is based on basis decomposition and allows to configure parameters for simulating different clusters within the data. The package provides three classes to simulate functional data: :class:`Simulation`, :class:`Brownian`, and :class:`KarhunenLoeve`. The :class:`Simulation` class is the abstract class to simulate functional data. New simulation classes can be added by extending this abstract class. The :class:`Brownian` class simulates functional data with different Brownian paths. The :class:`KarhunenLoeve` class simulates functional data using the Karhunen-Loève expansion.

.. autosummary::
    :toctree: autosummary

    FDApy.simulation.Simulation
    FDApy.simulation.Brownian
    FDApy.simulation.KarhunenLoeve


Datasets
========

The :class:`Datasets` class provides a set of functions to simulate functional datasets that have already been used in the literature.

.. autosummary::
    :toctree: autosummary

    FDApy.simulation.Datasets
