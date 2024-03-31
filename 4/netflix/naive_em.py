"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    #Your code here

    n, d = X.shape
    k = mixture.p.shape[0]
    soft_temp = np.zeros([n,k])

    for i in range(n):
        for j in range(k):
            norm = np.linalg.norm( X[i] - mixture.mu[j] )
            norm_sq = norm**2
            soft_temp[i, j] = mixture.p[j]*( np.exp( -( norm_sq / ( 2 * mixture.var[j] ) ) ) / ( 2*np.pi*mixture.var[j] )**(d/2) )

    soft_counts = soft_temp/soft_temp.sum(axis=1)[:,None]
    log_likelihood = soft_counts*np.log(np.divide(soft_counts, soft_temp))
    return [soft_counts, -log_likelihood.sum()]


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    #Your code here

    n, d = X.shape
    k = post.shape[1]
    post_cluster_sum = (post.sum(axis=0)).T
    p = np.divide( post_cluster_sum, n )
    mu = np.divide( post.T@X, post_cluster_sum[:,None] )

    # better way with matrix operations?
    temp = np.zeros([n,k])
    for i in range(n):
        for j in range(k):
            norm = np.linalg.norm( X[i] - mu[j] )
            norm_sq = norm**2
            temp[i, j] = post[i, j]*norm_sq
    var = np.divide( temp.sum(axis=0), d*post_cluster_sum)

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
        mixture = mstep(X, post)
        if not first and (new_log_likelihood - old_log_likelihood) <= (10**-6 * abs(new_log_likelihood)):
            finished = True
        elif first:
            first = False
        old_log_likelihood = new_log_likelihood
        
    return mixture, post, new_log_likelihood
