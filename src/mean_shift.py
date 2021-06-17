import numpy as np
from sklearn.neighbors import KDTree


class MeanShift:
    def __init__(self, radio):
        self.radio = radio
        self.centroids = []

    def fit(self, x):
        centroids = x
        prev_centroids = np.zeros((x.shape[0], x.shape[1]))
        tree = KDTree(x)
        while not np.array_equal(centroids, prev_centroids):
            new_centroids = []
            for centroid in centroids:
                window = list(tree.query_radius([centroid], r=self.radio)[0])
                window = np.array([x[i] for i in window])
                new_centroid = np.round(np.mean(window, axis=0)).astype(int)
                new_centroids.append(new_centroid)

            unique_centroids = np.unique(np.array(new_centroids), axis=0)
            prev_centroids = centroids
            centroids = unique_centroids
        self.centroids = centroids

    def predict(self, x):
        classes = {i: [] for i in range(len(self.centroids))}
        for j in range(x.shape[0]):
            dist = [np.linalg.norm(x[j] - centroid) for centroid in self.centroids]
            class_idx = dist.index(min(dist))
            classes[class_idx].append(j)
        return {"classes": classes, "centroids": self.centroids}