#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Basis
-----

"""
import numpy as np
import numpy.typing as npt

from scipy.integrate import simpson

from typing import Callable, Optional, List, Union

from .functional_data import DenseFunctionalData, MultivariateFunctionalData
from .functional_data import _tensor_product

from .argvals import DenseArgvals
from .values import DenseValues

from ..misc.basis import _basis_wiener, _basis_legendre, _basis_fourier, _basis_bsplines


#######################################################################
# Definition of the basis (eigenfunctions)
def _simulate_basis(
    name: str,
    argvals: npt.NDArray[np.float64],
    n_functions: int = 5,
    is_normalized: bool = False,
    add_intercept: bool = True,
    **kwargs,
) -> npt.NDArray[np.float64]:
    """Redirect to the right simulation basis function.

    Parameters
    ----------
    name: str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Name of the basis to use.
    argvals: npt.NDArray[np.float64]
        The values on which the basis functions are evaluated.
    n_functions: int, default=5
        Number of functions to compute.
    is_normalized: bool
        Should we normalize the functions?
    add_intercept: bool, default=True
        Should the constant functions be into the basis?
    **kwargs
        degree: int, default=3
            Degree of the B-splines. The default gives cubic splines.

    Returns
    -------
    values: npt.NDArray[np.float64], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions.

    Example
    -------
    >>> _simulate_basis(
    ...     'legendre',
    ...     n_functions=3,
    ...     argvals=np.arange(-1, 1, 0.1),
    ...     norm=True
    ... )

    """
    if not add_intercept:
        n_functions = n_functions + 1

    if name == "legendre":
        values = _basis_legendre(argvals, n_functions)
    elif name == "wiener":
        values = _basis_wiener(argvals, n_functions)
    elif name == "fourier":
        values = _basis_fourier(argvals, n_functions)
    elif name == "bsplines":
        values = _basis_bsplines(
            argvals,
            n_functions,
            degree=kwargs.get("degree", 3),
            domain_min=kwargs.get("domain_min", np.min(argvals)),
            domain_max=kwargs.get("domain_max", np.max(argvals)),
        )
    else:
        raise NotImplementedError(f"Basis {name!r} not implemented!")

    if is_normalized:
        norm2 = np.sqrt(simpson(values * values, argvals))
        values = np.divide(values, norm2[:, np.newaxis])

    if add_intercept:
        return values
    else:
        return values[1:]


def _simulate_basis_multivariate_weighted(
    basis_name: List[str],
    argvals: List[npt.NDArray[np.float64]],
    n_functions: int = 5,
    is_normalized: bool = False,
    runif: Optional[Callable] = np.random.uniform,
    **kwargs,
):
    """Simulate function for multivariate functional data.

    The multivariate eigenfunction basis consists of weighted univariate
    orthonormal bases. This yields an orthonormal basis of multivariate
    functions with `n_functions` elements. For data on two-dimensional domains,
    the univariate basis is constructed as a tensor product of univariate bases
    in each direction. The simulation setup is based on [1]_.

    Parameters
    ----------
    basis_name: List[str]
        Name of the basis to used.
    argvals: List[npt.NDArray[np.float64]]
        The values on which the basis functions are evaluated.
    n_functions: int
        Number of basis functions to used.
    is_normalized: bool
        Should we normalize the functions?
    runif: Optional[Callable], default=np.random.uniform
        Method used to generate uniform distribution. If `None`, all the
        weights are set to :math:`1`.
    **kwargs
        degree: int, default=3
            Degree of the B-splines. The default gives cubic splines.

    Returns
    -------
    List[npt.NDArray[np.float64]], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions.

    References
    ----------
    .. [1] Happ C. & Greven S. (2018) Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains, Journal of the American Statistical Association, 113:522,
        649-659, DOI: 10.1080/01621459.2016.1273115

    """
    # Define weights
    if runif is None:
        alpha = np.repeat(1, len(basis_name))
    else:
        alpha = runif(low=0.2, high=0.8, size=len(basis_name))
    weights = np.sqrt(alpha / np.sum(alpha))

    return [
        weight * _simulate_basis(name, argval, n_functions, is_normalized, **kwargs)
        for name, argval, weight in zip(basis_name, argvals, weights)
    ]


def _simulate_basis_multivariate_split(
    basis_name: List[str],
    argvals: List[npt.NDArray[np.float64]],
    n_functions: int = 5,
    is_normalized: bool = False,
    rchoice: Callable = np.random.choice,
    **kwargs,
):
    """Simulate function for multivariate functional data.

    The basis functions of an underlying big orthonormal basis are split in
    `n_functions` parts, translated and possible reflected. This yields an
    orthonormal basis of multivariate functions with `n_functions` elements.
    For data on two-dimensional domains, the univariate basis is constructed as
    a tensor product of univariate bases in each direction. The simulation
    setup is based on [1]_.

    Parameters
    ----------
    basis_name: List[str]
        Name of the basis to used.
    argvals: List[npt.NDArray[np.float64]]
        The values on which the basis functions are evaluated.
    n_functions: int
        Number of basis functions to used.
    is_normalized: bool
        Should we normalize the functions?
    rchoice: Callable, default=np.random.choice
        Method used to generate binomial distribution.
    **kwargs
        degree: int, default=3
            Degree of the B-splines. The default gives cubic splines.

    Returns
    -------
    List[npt.NDArray[np.float64]], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions.

    References
    ----------
    .. [1] Happ C. & Greven S. (2018) Multivariate Functional Principal
        Component Analysis for Data Observed on Different (Dimensional)
        Domains, Journal of the American Statistical Association, 113:522,
        649-659, DOI: 10.1080/01621459.2016.1273115

    """
    # Create "big" argvals vector and the split points
    x = [argvals[0]]
    split_vals = [0, len(x[0])]
    for idx in np.arange(1, len(argvals)):
        x.append(argvals[idx] - np.min(argvals[idx]) + np.max(x[-1]))
        split_vals.append(split_vals[-1] + len(x[-1]))

    # Simulate the "big" basis
    x_concat = np.concatenate(x)
    values = _simulate_basis(basis_name, x_concat, n_functions, is_normalized, **kwargs)

    flips = rchoice((-1, 1), size=len(argvals))
    return [
        flips[idx] * values[:, split_vals[idx] : split_vals[idx + 1]]
        for idx in np.arange(len(argvals))
    ]


def _simulate_basis_multivariate(
    simulation_type: str,
    n_components: int,
    name: Union[str, List[str]],
    argvals: List[npt.NDArray[np.float64]],
    n_functions: int = 5,
    is_normalized: bool = False,
    **kwargs,
) -> npt.NDArray[np.float64]:
    """Redirect to the right simulation basis function.

    Parameters
    ----------
    simulation_type: str, {'split', 'weighted'}
        Type of the simulation.
    n_components: int
        Number of components to generate.
    name: Union[str, List[str]]
        Basis names to use, {'legendre', 'wiener', 'fourier', 'bsplines'}.
    argvals: npt.NDArray[np.float64]
        The values on which the basis functions are evaluated.
    n_functions: int, default=5
        Number of functions to compute.
    is_normalized: bool
        Should we normalize the functions?
    **kwargs:
        rchoice: Callable, default=np.random.choice
            Method used to generate binomial distribution.
        runif: Callable, default=np.random.uniform
            Method used to generate uniform distribution.
        degree: int, default=3
            Degree of the B-splines. The default gives cubic splines.

    Returns
    -------
    values: List[npt.NDArray[np.float64]], shape=(n_functions, len(argvals))
        An array containing the evaluation of `n_functions` functions.

    Example
    -------
    >>> _simulate_basis_multivariate(
    ...     simulation_type='split',
    ...     n_components=3,
    ...     name='fourier',
    ...     argvals=[
    ...         np.linspace(0, 1, 101),
    ...         np.linspace(-np.pi, np.pi, 101),
    ...         np.linspace(-0.5, 0.5, 51)
    ...     ],
    ...     n_functions=3,
    ...     norm=True
    ... )

    """
    if len(argvals) != n_components:
        raise ValueError(f"`len(argvals)` should be equal to {n_components}.")

    if simulation_type == "split":
        if not isinstance(name, (str, str)):
            raise ValueError(
                "For the `split` simulation type, `basis_name` " "should be a str."
            )
        values = _simulate_basis_multivariate_split(
            name,
            argvals,
            n_functions,
            is_normalized,
            kwargs.pop("rchoice", np.random.choice),
            **kwargs,
        )
    elif simulation_type == "weighted":
        if not isinstance(name, list):
            raise ValueError(
                "For the `weighted` simulation type, `basis_name` " "should be a list."
            )
        if len(name) != n_components:
            raise ValueError(
                "For the `weighted` simulation type, `len(basis_name)` "
                f"should be equal to {n_components}."
            )
        values = _simulate_basis_multivariate_weighted(
            name,
            argvals,
            n_functions,
            is_normalized,
            kwargs.pop("runif", np.random.uniform),
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Simulation {simulation_type!r} not implemented!")
    return values


###############################################################################
# Class Basis


class Basis(DenseFunctionalData):
    r"""Define univariate orthonormal basis.

    Parameters
    ----------
    name: str, {'legendre', 'wiener', 'fourier', 'bsplines'}
        Denotes the basis of functions to use.
    n_functions: int
        Number of functions in the basis.
    dimension: str, {'1D', '2D'}, default='1D'
        Dimension of the basis to simulate. If '2D', the basis is simulated as
        the tensor product of the one dimensional basis of functions by itself.
        The number of functions in the 2D basis will be :math:`n_function^2`.
    argvals: Optional[npt.NDArray[np.float64]]
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j` th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    is_normalized: bool, default=False
        Should we normalize the basis function?
    add_intercept: bool, default=True
        Should the constant functions be into the basis?
    **kwargs
        degree: int, default=3
            Degree of the B-splines. The default gives cubic splines.

    """

    def __init__(
        self,
        name: str,
        n_functions: int = 5,
        dimension: str = "1D",
        argvals: Optional[npt.NDArray[np.float64]] = None,
        is_normalized: bool = False,
        add_intercept: bool = True,
        **kwargs,
    ) -> None:
        """Initialize Basis object."""
        self.name = name
        self.is_normalized = is_normalized
        self.dimension = dimension

        if argvals is None:
            argvals = np.arange(0, 1.01, 0.01)

        values = _simulate_basis(
            name, argvals, n_functions, is_normalized, add_intercept, **kwargs
        )

        if dimension == "1D":
            super().__init__(
                DenseArgvals({"input_dim_0": argvals}), DenseValues(values)
            )
        elif dimension == "2D":
            cut = np.ceil(np.sqrt(n_functions)).astype(int)
            rest = (n_functions / cut + 1).astype(int)

            basis_first_dim = DenseFunctionalData(
                DenseArgvals({"input_dim_0": argvals}), DenseValues(values[:cut])
            )
            basis_second_dim = DenseFunctionalData(
                DenseArgvals({"input_dim_0": argvals}),
                DenseValues(values[1 : (rest + 1)]),
            )
            basis2d = _tensor_product(basis_first_dim, basis_second_dim)
            super().__init__(
                DenseArgvals(basis2d.argvals), DenseValues(basis2d.values[:n_functions])
            )
        else:
            raise ValueError(f"{dimension} is not a valid dimension!")

    @property
    def name(self) -> str:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not isinstance(new_name, str):
            raise TypeError(f"{new_name!r} has to be `str`.")
        self._name = new_name

    @property
    def is_normalized(self) -> bool:
        """Getter for is_normalized."""
        return self._is_normalized

    @is_normalized.setter
    def is_normalized(self, new_is_normalized: bool) -> None:
        self._is_normalized = new_is_normalized

    @property
    def dimension(self) -> str:
        """Getter for dimension."""
        return self._dimension

    @dimension.setter
    def dimension(self, new_dimension: str) -> None:
        self._dimension = new_dimension


###############################################################################
# Class MultivariateBasis
class MultivariateBasis(MultivariateFunctionalData):
    r"""Define multivariate orthonormal basis.

    Parameters
    ----------
    simulation_type: str, {'split', 'weighted'}
        Type of the simulation.
    n_components: int
        Number of components to generate.
    name: Union[str, List[str]]
        Name of the basis to use. One of
        `{'legendre', 'wiener', 'fourier', 'bsplines'}`.
    n_functions: int
        Number of functions in the basis.
    dimension: Optional[List[str]], {'1D', '2D'}, default=None
        Dimension of the basis to simulate. If '2D', the basis is simulated as
        the tensor product of the one dimensional basis of functions by itself.
        The number of functions in the 2D basis will be :math:`n_function^2`.
    argvals: Optional[Dict[str, npt.NDArray[np.float64]]]
        The sampling points of the functional data. Each entry of the
        dictionary represents an input dimension. The shape of the :math:`j` th
        dimension is :math:`(m_j,)` for :math:`0 \leq j \leq p`.
    is_normalized: bool, default=False
        Should we normalize the basis function?
    **kwargs:
        rchoice: Callable, default=np.random.choice
            Method used to generate binomial distribution.
        runif: Callable, default=np.random.uniform
            Method used to generate uniform distribution.
        degree: int, default=3
            Degree of the B-splines. The default gives cubic splines.

    """

    def __init__(
        self,
        simulation_type: str,
        n_components: int,
        name: Union[str, List[str]],
        n_functions: int = 5,
        dimension: Optional[List[str]] = None,
        argvals: Optional[npt.NDArray[np.float64]] = None,
        is_normalized: bool = False,
        **kwargs,
    ) -> None:
        """Initialize Basis object."""
        self.simulation_type = simulation_type
        self.name = name
        self.is_normalized = is_normalized

        if argvals is None:
            argvals = n_components * [np.arange(0, 1.01, 0.01)]
        self.dimension = n_components * ["1D"] if dimension is None else dimension

        values = _simulate_basis_multivariate(
            simulation_type,
            n_components,
            name,
            argvals,
            n_functions,
            is_normalized,
            **kwargs,
        )

        basis_fd = []
        for argval, basis, dim in zip(argvals, values, self.dimension):
            temp = DenseFunctionalData(
                DenseArgvals({"input_dim_0": argval}), DenseValues(basis)
            )
            if dim == "2D":
                temp = _tensor_product(temp, temp)
            basis_fd.append(temp[:n_functions])
        super().__init__(basis_fd)

    @property
    def simulation_type(self) -> str:
        """Getter for simulation_type."""
        return self._simulation_type

    @simulation_type.setter
    def simulation_type(self, new_simulation_type: str) -> None:
        if not isinstance(new_simulation_type, str):
            raise TypeError(f"{new_simulation_type!r} has to be `str`.")
        self._simulation_type = new_simulation_type

    @property
    def name(self) -> Union[str, List[str]]:
        """Getter for name."""
        return self._name

    @name.setter
    def name(self, new_name: Union[str, List[str]]) -> None:
        if isinstance(new_name, str):
            self._name = new_name
        elif isinstance(new_name, list) and all(isinstance(x, str) for x in new_name):
            self._name = new_name
        else:
            raise TypeError(f"{new_name!r} has to be a `str` or `List[str]`.")

    @property
    def is_normalized(self) -> bool:
        """Getter for is_normalized."""
        return self._is_normalized

    @is_normalized.setter
    def is_normalized(self, new_is_normalized: bool) -> None:
        self._is_normalized = new_is_normalized

    @property
    def dimension(self) -> List[str]:
        """Getter for dimension."""
        return self._dimension

    @dimension.setter
    def dimension(self, new_dimension: List[str]) -> None:
        self._dimension = new_dimension
