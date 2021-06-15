import numpy as np
from sklearn.neighbors import KDTree

from python.utils import multivariate_normal


def group_classes_1(centroids, dataset):
    classes = {i: [] for i in range(len(centroids))}
    for j in range(dataset.shape[0]):
        dist = [np.linalg.norm(dataset[j] - centroid) for centroid in centroids]
        class_idx = dist.index(min(dist))
        classes[class_idx].append(j)
    return classes, centroids


def group_classes_2(labels, dataset):
    clusters = list(set(labels))
    classes = {i: [] for i in clusters}

    for i in range(dataset.shape[0]):
        if labels[i] > 0:
            classes[labels[i]].append(i)

    return classes, clusters


def k_means(dataset, k=3):
    centroids = np.array([dataset[i] for i in np.random.choice(dataset.shape[0], k)])
    prev = np.zeros((dataset.shape[0], k))
    while np.array_equal(centroids, prev) is False:
        classes = [[] for i in range(k)]
        for j in range(dataset.shape[0]):
            dist = [np.linalg.norm(dataset[j] - centroid) for centroid in centroids]
            class_idx = dist.index(min(dist))
            classes[class_idx].append(j)
        prev = centroids
        for i in range(k):
            centroids[i] = np.mean([dataset[j] for j in classes[i]], axis=0)
    return group_classes_1(centroids, dataset)


def mean_shift(dataset, radio=10):
    centroids = dataset
    prev_centroids = np.zeros((dataset.shape[0], dataset.shape[1]))
    tree = KDTree(dataset)
    while not np.array_equal(centroids, prev_centroids):
        new_centroids = []
        print(centroids.shape)
        for centroid in centroids:
            window = list(tree.query_radius([centroid], r=radio)[0])
            window = np.array([dataset[i] for i in window])
            new_centroid = np.round(np.mean(window, axis=0)).astype(int)
            new_centroids.append(new_centroid)

        unique_centroids = np.unique(np.array(new_centroids), axis=0)
        prev_centroids = centroids
        centroids = unique_centroids
    return group_classes_1(centroids, dataset)


def dbscan(dataset, radio=1, min_pts=4):
    label = np.zeros(dataset.shape[0])
    tree = KDTree(dataset)
    for i in range(dataset.shape[0]):
        if label[i] != 0:
            continue
        neighbors = list(tree.query_radius([dataset[i]], r=radio)[0])
        if len(neighbors) < min_pts:
            label[i] = -1
            continue
        c = np.max(label) + 1
        label[i] = c
        S = neighbors
        while len(S):
            j = S[0]
            S.pop(0)
            if i == j:
                continue
            if label[j] == -1:
                label[j] = c
            if label[j] != 0:
                continue
            neighbors = list(tree.query_radius([dataset[i]], r=radio)[0])
            label[j] = c
            if len(neighbors) < min_pts:
                continue
            S = list(set(S + neighbors))

    labels = list(map(int, label))
    return group_classes_2(labels, dataset)


def gmm(dataset, n_k=3, max_iter=100):
    dataset_split = np.array_split(dataset, n_k)

    pi = [1/n_k for i in range(n_k)]
    means = [np.mean(subset, axis=0) for subset in dataset_split]
    covariances = [np.cov(subset.T) for subset in dataset_split]

    del dataset_split

    for itr in range(max_iter):
        gamma = np.zeros((dataset.shape[0], n_k))
        for n in range(dataset.shape[0]):
            for k in range(n_k):
                gamma[n][k] = pi[k] * multivariate_normal(dataset[n], means[k], covariances[k])
                gamma[n][k] /= sum([pi[j] * multivariate_normal(dataset[n], means[j], covariances[j]) for j in range(n_k)])
        N = np.sum(gamma, axis=0)

        means = np.zeros((n_k, dataset.shape[1]))
        for k in range(n_k):
            for n in range(dataset.shape[0]):
                means[k] += gamma[n][k] * dataset[n]
        means = [1/N[k] * means[k] for k in range(n_k)]

        covariances = [np.zeros((dataset.shape[0], dataset.shape[0])) for k in range(n_k)]
        for k in range(n_k):
            covariances[k] = np.cov(dataset.T, aweights=(gamma[:, k]), ddof=0)
        covariances = [1/N[k] * covariances[k] for k in range(n_k)]

        pi = [N[k]/dataset.shape[0] for k in range(n_k)]

    probs = []
    for n in range(dataset.shape[0]):
        probs.append([multivariate_normal(dataset[n], means[k], covariances[k]) for k in range(n_k)])

    labels = []
    for prob in probs:
        labels.append(prob.index(max(prob)))

    return group_classes_2(labels, dataset)
