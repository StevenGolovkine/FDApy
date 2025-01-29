"""Smoothing module."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["local_polynomial", "psplines"],
    submod_attrs={"local_polynomial": ["LocalPolynomial"], "psplines": ["PSplines"]},
)
