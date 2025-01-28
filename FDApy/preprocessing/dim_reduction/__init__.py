"""Dimension reduction module."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["fcp_tpa", "ufpca", "mfpca"],
    submod_attrs={"fcp_tpa": ["FCPTPA"], "ufpca": ["UFPCA"], "mfpca": ["MFPCA"]},
)
