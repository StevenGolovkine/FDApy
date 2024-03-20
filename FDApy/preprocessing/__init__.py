"""Preprocessing module."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["dim_reduction", "smoothing"],
    submod_attrs={
        "dim_reduction": [
            "FCPTPA",
            "UFPCA",
            "MFPCA",
        ],
        "smoothing": ["LocalPolynomial", "PSplines"],
    },
)
