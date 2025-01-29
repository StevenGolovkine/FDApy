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
        "argvals": ["Argvals", "DenseArgvals", "IrregularArgvals"],
        "values": ["Values", "DenseValues", "IrregularValues"],
        "basis": ["Basis", "MultivariateBasis"],
    },
)
