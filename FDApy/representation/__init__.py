"""Representation module."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["argvals", "basis", "functional_data", "values"],
    submod_attrs={
        "functional_data": [
            "DenseFunctionalData",
            "IrregularFunctionalData",
            "MultivariateFunctionalData",
        ],
        "argvals": ["DenseArgvals", "IrregularArgvals"],
        "values": ["DenseValues", "IrregularValues"],
        "basis": ["Basis", "MultivariateBasis"],
    },
)
