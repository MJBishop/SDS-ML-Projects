"""Mixture model for matrix completion"""
from typing import Tuple

from matplotlib.pyplot import axis
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    #Your code here

    n = X.shape[0]
    k = mixture.p.shape[0]
    soft_temp = np.zeros([n,k], dtype=np.float64)

    for u in range(n):
        # filter vector for > 0
        X_u = X[u]
        filter_vec = X_u > 0
        X_u_C_u = X_u[filter_vec]
        d = X_u_C_u.shape[0]

        for j in range(k):
            mu_j = mixture.mu[j]
            mu_j_C_u = mu_j[filter_vec]

            norm = np.linalg.norm( X_u_C_u - mu_j_C_u )
            norm_sq = norm**2
            soft_temp[u, j] = np.log(mixture.p[j] + 1e-16) - ( norm_sq / ( 2 * mixture.var[j] ) ) - np.log( (2*np.pi*mixture.var[j] )**(d/2.0) + 1e-16 )

    soft_counts_log = soft_temp - logsumexp(soft_temp, axis=1)[:,None]
    soft_counts = np.exp(soft_counts_log)
    soft_counts = np.where( soft_counts == 0, 1e-32, soft_counts)
    log_liklihood = soft_counts*(np.log(soft_counts + 1e-16) - soft_temp)
    return [soft_counts, -log_liklihood.sum()]
    



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    #Your code here

    n,d = X.shape
    k = post.shape[1]
    post_cluster_sum = (post.sum(axis=0)).T

    # p
    p = np.divide( post_cluster_sum, n )

    # mean
    mu = mixture.mu
    for j in range(k):
        for l in range(d):
            sum = 0
            new_mu = 0
            for u in range(n):
                new_mu += post[u, j]*X[u, l]
                if X[u, l] != 0:
                    sum += post[u, j]
            if sum >= 1:
                mu[j, l] = new_mu / sum


    # variance
    temp_numerator = np.zeros([n,k], dtype=np.float64)
    temp_denominator = np.zeros([n,k], dtype=np.float64)
    for u in range(n):
        X_u = X[u]
        filter_vec = X_u > 0
        X_u_C_u = X_u[filter_vec]
        d = X_u_C_u.shape[0]

        for j in range(k):
            mu_j = mu[j]
            mu_j_C_u = mu_j[filter_vec]

            norm = np.linalg.norm( X_u_C_u - mu_j_C_u )
            norm_sq = norm**2
            temp_numerator[u, j] = post[u, j]*norm_sq
            temp_denominator[u, j] = post[u,j]*d
        
    var = np.divide( temp_numerator.sum(axis=0), temp_denominator.sum(axis=0) )
    var = np.where( var < 0.25, 0.25, var)

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    #Your code here

    old_log_likelihood = 0
    finished = False
    first = True
    while not finished:
        post, new_log_likelihood = estep(X, mixture)
        mixture = mstep(X, post, mixture)
        if not first and (new_log_likelihood - old_log_likelihood) <= (10**-6 * abs(new_log_likelihood)):
            finished = True
        elif first:
            first = False
        old_log_likelihood = new_log_likelihood
        
    return mixture, post, new_log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    #Your code here

    n = X.shape[0]
    k = mixture.p.shape[0]
    soft_temp = np.zeros([n,k], dtype=np.float64)

    for u in range(n):
        # filter vector for > 0
        X_u = X[u]
        filter_vec = X_u > 0
        X_u_C_u = X_u[filter_vec]
        d = X_u_C_u.shape[0]

        for j in range(k):
            mu_j = mixture.mu[j]
            mu_j_C_u = mu_j[filter_vec]

            norm = np.linalg.norm( X_u_C_u - mu_j_C_u )
            norm_sq = norm**2
            
            var = mixture.var[j]
            if var == 0:
                var = 1e-32
            exp = ( norm_sq / ( 2 * var ) )
            if (norm_sq == 0):
                exp == 0
            p = mixture.p[j]
            if p == 0:
                p = 1e-32
            soft_temp[u, j] = np.log(p + 1e-16) - exp - (d/2)*np.log( (2*np.pi*var ) + 1e-16)

    soft_counts_log = soft_temp - logsumexp(soft_temp, axis=1)[:,None]
    soft_counts = np.exp(soft_counts_log)
    soft_counts = np.where( soft_counts == 0, 1e-32, soft_counts)

    n, d = X.shape
    X_pred = np.zeros([n, d])
    for u in range(n):
        for l in range(d):
            if X[u, l] == 0:
                X_pred[u, l] = soft_counts[u]@(mixture.mu).T[l]
            else:
                X_pred[u, l] = X[u, l]
    return X_pred
