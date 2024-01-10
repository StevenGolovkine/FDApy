"""Miscellaneous module."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["loader", "utils"],
    submod_attrs={"loader": ["read_csv"]},
)
