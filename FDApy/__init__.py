"""FDApy package."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "clustering",
        "misc",
        "preprocessing",
        "regression",
        "representation",
        "simulation",
        "visualization",
    ],
    submod_attrs={
        "representation": [
            "DenseFunctionalData",
            "IrregularFunctionalData",
            "MultivariateFunctionalData",
        ],
        "misc": ["read_csv"],
    },
)
