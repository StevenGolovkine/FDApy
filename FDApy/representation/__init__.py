"""Representation module."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["argvals", "basis", "functional_data", "values"],
    submod_attrs={
        "functional_data": [
            "FunctionalData",
            "GridFunctionalData",
            "BasisFunctionalData",
            "DenseFunctionalData",
            "IrregularFunctionalData",
            "MultivariateFunctionalData",
            "DenseFunctionalDataIterator",
            "IrregularFunctionalDataIterator",
            "BasisFunctionalDataIterator",
        ],
        "argvals": ["DenseArgvals", "IrregularArgvals"],
        "values": ["DenseValues", "IrregularValues"],
        "basis": ["Basis", "MultivariateBasis"],
    },
)
