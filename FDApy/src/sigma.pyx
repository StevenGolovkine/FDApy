# distutils: language=c++
# cython: language_level=3
# cython: linetrace=True


from libcpp.vector cimport vector


def estimate_sigma(values):
    """Perform an estimation of sigma

    Parameters:
    -----------
    values: list
    """
    # Define variables
    cdef int N, M_n, k
    cdef double sigma, squared_diff
    cdef vector[double] obs

    # Get the number of curves
    N = len(values)

    sigma = 0
    for obs in values:
        M_n = len(obs)

        squared_diff = 0
        for k in range(1, M_n):
            squared_diff += (obs[k] - obs[k - 1])**2

        sigma += (squared_diff / (2 * (M_n - 1)))

    return (sigma / N)**0.5


def estimate_sigma_MSE(values, values_estim):
    """Perform an estimation of sigma using MSE

    Parameters
    ----------
    values: list

    values_estim: list
    """
    # Define variables
    cdef int N, M_n, k
    cdef double sigma, squared_diff
    cdef vector[double] obs, obs_estim

    # Get the number of curves
    N = len(values)

    sigma = 0
    for (obs, obs_estim) in zip(values, values_estim):
        M_n = len(obs)

        squared_diff = 0
        for k in range(M_n):
            squared_diff += (obs[k] - obs_estim[k])**2

        sigma += (squared_diff / M_n)

    return (sigma / N)**2
