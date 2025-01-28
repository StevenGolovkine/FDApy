"""Simulation module."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["brownian", "datasets", "karhunen", "simulation"],
    submod_attrs={
        "simulation": ["Simulation"],
        "brownian": ["Brownian"],
        "datasets": ["Datasets"],
        "karhunen": ["KarhunenLoeve"],
    },
)
