import numpy as np
from scipy.stats import multivariate_normal

from src.k_means import KMeans


class GMM:
    def __init__(self, k, max_iter=10, tol=1e-1):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.pi = np.array([1 / k for _ in range(k)], dtype=np.float128)
        self.means = None
        self.covariances = None
        self.clusters = None

    def __initialize(self, x):
        k_means = KMeans(k=self.k)
        k_means.fit(x)
        classes, centroids = k_means.predict(x).values()
        self.means = np.array(centroids, dtype=np.float128)
        self.covariances = [np.identity(x.shape[1], dtype=np.float128) for _ in range(self.k)]

    def __e_step(self, x):
        gamma = np.zeros((x.shape[0], self.k), dtype=np.float128)
        for n in range(x.shape[0]):
            accum_prob = sum([self.pi[j] * multivariate_normal.pdf(x[n], self.means[j], self.covariances[j],
                                                                   allow_singular=True) for j in range(self.k)])
            for k in range(self.k):
                gamma[n][k] = np.log(self.pi[k]) + multivariate_normal.logpdf(x[n], self.means[k], self.covariances[k],
                                                                              allow_singular=True)
                gamma[n][k] = 0 if accum_prob == 0 else gamma[n][k] - np.log(accum_prob)
                gamma[n][k] = np.exp(gamma[n][k])
        return gamma

    def __m_step(self, x, gamma, N):
        means = np.zeros((self.k, x.shape[1]))
        for k in range(self.k):
            for n in range(x.shape[0]):
                means[k] += gamma[n][k] * x[n]
        self.means = [1 / N[k] * means[k] for k in range(self.k)]

        covariances = [np.zeros((x.shape[1], x.shape[1])) for _ in range(self.k)]
        for k in range(self.k):
            covariances[k] = np.cov(x.T, aweights=(gamma[:, k]), ddof=0)
        self.covariances = [1 / N[k] * covariances[k] for k in range(self.k)]

        self.pi = [N[k] / x.shape[0] for k in range(self.k)]

    def __log_likelihood(self, x):
        log_likelihood = 0
        for n in range(x.shape[0]):
            sum_k = sum(
                [self.pi[k] * multivariate_normal.pdf(x[n], self.means[k], self.covariances[k], allow_singular=True)
                 for k in range(self.k)])
            log_likelihood += np.log(sum_k) if sum_k != 0 else np.log(1e-320)
        return log_likelihood

    def fit(self, x):
        # 1. Initialization
        self.__initialize(x)

        prev_log_likelihood = -np.infty
        for itr in range(self.max_iter):
            print("itr", itr)
            # 2. E step
            gamma = self.__e_step(x)
            N = np.sum(gamma, axis=0)
            print(N)

            # 3. M step
            self.__m_step(x, gamma, N)

            # 4. Evaluate log likelihood
            log_likelihood = self.__log_likelihood(x)
            if self.tol > abs(prev_log_likelihood - log_likelihood):
                break
            prev_log_likelihood = log_likelihood

    def predict(self, x):
        probs = []
        for n in range(x.shape[0]):
            probs.append([self.pi[k] * multivariate_normal.pdf(x[n], self.means[k], self.covariances[k],
                                                               allow_singular=True) for k in range(self.k)])

        labels = []
        for prob in probs:
            labels.append(prob.index(max(prob)))

        clusters = list(set(labels))
        classes = {i: [] for i in clusters}
        self.clusters = [x[i] for i in clusters]

        for i in range(x.shape[0]):
            if labels[i] > 0:
                classes[labels[i]].append(i)
        return {"classes": classes, "centroids": self.clusters}
