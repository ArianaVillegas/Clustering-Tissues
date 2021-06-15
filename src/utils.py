import numpy as np


def multivariate_normal(x, mean, covariance):
    return (2 * np.pi) ** (-len(x)/2) * np.linalg.det(covariance) ** (-1/2) * np.exp(-np.dot(np.dot((x-mean).T, np.linalg.inv(covariance)), (x-mean))/2)
