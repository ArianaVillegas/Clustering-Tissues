from math import floor

import numpy as np


def multivariate_normal(x, mean, covariance):
    return (2 * np.pi) ** (-len(x) / 2) * np.linalg.det(covariance) ** (-1 / 2) * np.exp(
        -np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)), (x - mean)) / 2)


def get_near_psd(A):
    C = (A + A.T) / 2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

