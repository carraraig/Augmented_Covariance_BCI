"""
Library to implement the useful utilities needed
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy.core.numeric import dot, newaxis, array, concatenate
import scipy
from sklearn.covariance import empirical_covariance

def AIC(U, nb_params, N):
    """
    Function to implement the AIC (Akaike information criterion)
    :param U: Estimator matrix of mean square error (Forward)
    :param nb_params: Number of independent parameters in the model
    :param N: Total time step of the multivariate time series
    :return:
            - AIC
    """
    return np.log(np.abs(np.linalg.det(U))) + 2 * nb_params / N


def AICc(U, nb_params, N):
    """
    Function to implement the AICc (Akaike information criterion biased corrected)
    :param U: Estimator matrix of mean square error (Forward)
    :param nb_params: Number of independent parameters in the model
    :param N: Total time step of the multivariate time series
    :return:
            - AIC
    """
    return np.log(np.abs(np.linalg.det(U))) + 2 * nb_params / (N - nb_params -1)


def BIC(U, nb_params, N):
    """
    Function to implement the BIC (Bayesian information criterion)
    :param U: Estimator matrix of mean square error (Forward)
    :param nb_params: Number of independent parameters in the model
    :param N: Total time step of the multivariate time series
    :return:
            - BIC
    """
    return np.log(np.abs(np.linalg.det(U))) + nb_params * np.log(N) / N


def HQ(U, nb_params, N):
    """
    Function to implement the HQ (Hannan and Quinn criterion)
    :param U: Estimator matrix of mean square error (Forward)
    :param nb_params: Number of independent parameters in the model
    :param N: Total time step of the multivariate time series
    :return:
            - AIC
    """
    return np.log(np.abs(np.linalg.det(U))) + (2 * np.log(np.log(N)) * nb_params) / N


def autocorrelation_element(x, i, normalisation='stable'):
    """
    A function to compute the autocorrelation element with step difference i.
    """
    d, N = x.shape
    gamma_elem = np.zeros((d, d))
    for t in range(N - i):
        gamma_elem += x[:, t+i].reshape(-1, 1) @ x[:, t].reshape(-1, 1).T

    if normalisation == 'stable':
        return gamma_elem / N
    else:
        return gamma_elem / (N - i)


def autocorrelation_element_Euclidean(x, i, normalisation='stable'):
    """
    A function to compute the autocorrelation element with step difference i.
    """
    d, N = x.shape
    gamma_elem = np.zeros((d, d))

    for t in range(N - abs(i)):
        gamma_elem += x[:, t + abs(i)].reshape(-1, 1) @ x[:, t].reshape(-1, 1).T

    if i < 0:
        gamma_elem = gamma_elem.T

    if normalisation == 'stable':
        return gamma_elem / N
    else:
        return gamma_elem / (N - i)


def autocovariance_element_corrected(x, i, normalisation='stable'):
    """
    A function to compute the autocorrelation element with step difference i.
    """
    d, N = x.shape
    gamma_elem = np.zeros((d, d))
    mean = np.zeros((d, 1))
    mean = np.mean(x, axis=1)

    for t in range(N - abs(i)):
        gamma_elem += (x[:, t + abs(i)].reshape(-1, 1) - mean.reshape(-1, 1)) @ (x[:, t].reshape(-1, 1).T - mean.reshape(-1, 1).T)

    if i < 0:
        gamma_elem = gamma_elem.T

    if normalisation == 'stable':
        return gamma_elem / N
    else:
        return gamma_elem / (N - abs(i))


def matrix_diag(matrix, nb_blocks):
    """
    Function to create a block diagonal matrix form a square matrix. The block on the diagonal are the original matrix

    :param matrix: Matrix that we want to have block diagonal
    :param nb_blocks: The number of blocks in the diagonal

    :return:
            - Block diagonal matrix
    """
    dim = matrix.shape[0]
    matrix_d = np.zeros((nb_blocks * dim, nb_blocks * dim))
    for i in range(0, nb_blocks * dim, dim):
        matrix_d[i:i + dim, i:i + dim] = matrix.copy()
    return matrix_d


def vectorization_matrix(matrix):
    """
    Function to vectorize a matrix. the vectorization of a matrix is a linear transformation which converts the matrix
    into a column vector. Specifically, the vectorization of a m × n matrix A, denoted vec(A), is the mn × 1 column
    vector obtained by stacking the columns of the matrix A on top of one another.
    (https://en.wikipedia.org/wiki/Vectorization_(mathematics))

    :param matrix: Matrix to be vectorized
    :return:
        - Return a vector of the matrix
    """
    return matrix.T.reshape(-1, 1)


def is_pos_def(matrix_coeff):
    """
    Function that return True if the matrix is a Positive defined matrix
    :param matrix_coeff: matrix to check, have to be square
    :return:
        True if matrix is a SPD
    """
    if check_symmetric(matrix_coeff) & np.all(scipy.linalg.eigh(matrix_coeff, eigvals_only=True) > 0):
        print("The coefficient Matrix is a SPD matrix")
    else:
        print("The coefficient Matrix is NOT a SPD matrix")
    if not check_symmetric(matrix_coeff):
        print("The coefficient Matrix is not Symmetric")
    if not np.all(np.linalg.eigvals(matrix_coeff) > 0):
        print("Eigenvalue Not Positive")


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def is_pos_def1(x):

    n_epoch, n_elect, t = x.shape
    B_fin = np.zeros((n_epoch, 1))
    for i in np.arange(n_epoch):
        if check_symmetric(x[i]) & np.all(np.linalg.eigvals(x[i]) > 0):
            B_fin[i] = 0
        else:
            B_fin[i] = 1

    return B_fin


def show_coefs(coefs):
    """
    Function to Plot the AR coefficient
    :param coefs: AR coefficient
    :return:
        Return the plot of the AR coefficient
    """
    p = len(coefs)
    fig, ax_c = plt.subplots(1, p)
    norm = colors.CenteredNorm()
    for i in range(p):
        if p == 1:
            ax = ax_c
        else:
            ax = ax_c[i]
        # Adapt the border of the norm if necessary
        adapted_norm = ax.imshow(coefs[i], norm=colors.CenteredNorm()).norm
        im = ax.imshow(coefs[i], cmap='bwr', norm=norm)
        norm.halfrange = max(norm.halfrange, adapted_norm.halfrange)
        fig.colorbar(im, ax=ax)

    plt.show()
    return


def OAS(A, shrinkage):

    N, _ = A.shape

    A = (1-shrinkage) * A + shrinkage * (np.trace(A) / N ) * np.eye(N)

    return A


def correlation(m, y=None, rowvar=1, bias=0, ddof=None):
    """
    Estimate a correlation matrix, given data. Is exactly the covariance but not taking away the mean
    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        form as that of `m`.
    rowvar : int, optional
        If `rowvar` is non-zero (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : int, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        .. versionadded:: 1.5
        If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
        the number of observations; this overrides the value implied by
        ``bias``. The default value is ``None``.
    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    """
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    X = array(m, ndmin=2, dtype=float)
    if X.size == 0:
        # handle empty arrays
        return np.array(m)
    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        axis = 0
        tup = (slice(None), newaxis)
    else:
        axis = 1
        tup = (newaxis, slice(None))


    if y is not None:
        y = array(y, copy=False, ndmin=2, dtype=float)
        X = concatenate((X,y), axis)

    if rowvar:
        N = X.shape[1]
    else:
        N = X.shape[0]

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    fact = float(N - ddof)

    if not rowvar:
        return (dot(X.T, X.conj()) / fact).squeeze()
    else:
        return (dot(X, X.T.conj()) / fact).squeeze()


def covariance_BM(X, lag):
    S = empirical_covariance(X.T)
    dim = X.shape[0]
    T = X.shape[1]
    A_s = np.zeros((dim, dim))

    Num1 = np.sum((((X[:, :] * X[:, :]) @ (X[:, :].T * X[:, :].T)) - (T * (S ** 2))))

    for s in np.arange(T - 1):
        if abs(s / lag) <= 1:
            A_s = A_s + (((X[:, :T - s] * X[:, s:]) @ (X[:, :T - s].T * X[:, s:].T)) - (
                    (T - s) * (S ** 2)))
        else:
            break
    Num2 = np.sum(A_s)

    # Computing the summation
    Num_sum = (Num1 + 2 * Num2) * (1 / (T - 1 - 2 * lag + (lag * (lag + 1)) / T)) * (1 / T)
    Denom_sum = np.sum((S - (np.trace(S) * np.eye(dim)) / dim) ** 2)

    # New shrinkage
    # We do this to prevent shrinking more than "1", which would invert the value of covariances
    Num_sum = min(Num_sum, Denom_sum)
    if Num_sum == 0:
        schri_new_optm = 0
    else:
        schri_new_optm = Num_sum / Denom_sum
    cov_schri_new_optm = (1 - schri_new_optm) * S + schri_new_optm * ((np.trace(S) * np.eye(dim)) / dim)
    return cov_schri_new_optm


def covariance_BM_epoch(X, lag):
    n_matrices, n_channels, n_times = X.shape
    covmats = np.empty((n_matrices, n_channels, n_channels))
    for i in range(n_matrices):
        covmats[i] = covariance_BM(X[i], lag=lag)
    return covmats
