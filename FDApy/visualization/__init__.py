"""Visualization module."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["plot"],
    submod_attrs={"plot": ["plot", "plot_multivariate"]},
)
