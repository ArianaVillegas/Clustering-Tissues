import numpy as np
from sklearn.neighbors import KDTree


class DBScan:
    def __init__(self, radio, min_pts):
        self.radio = radio
        self.min_pts = min_pts
        self.centroids = []
        self.labels = []

    def fit(self, x):
        label = np.zeros(x.shape[0])
        tree = KDTree(x)
        for i in range(x.shape[0]):
            if label[i] != 0:
                continue
            neighbors = list(tree.query_radius([x[i]], r=self.radio)[0])
            if len(neighbors) < self.min_pts:
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
                neighbors = list(tree.query_radius([x[i]], r=self.radio)[0])
                label[j] = c
                if len(neighbors) < self.min_pts:
                    continue
                S = list(set(S + neighbors))

        self.labels = list(map(int, label))
        self.centroids = list(set(label))

    def predict(self, x):
        centroids = list(set(self.labels))
        classes = {i: [] for i in centroids}
        self.centroids = [x[i] for i in centroids]

        for i in range(x.shape[0]):
            classes[self.labels[i]].append(i)
        return {"classes": classes, "centroids": self.centroids}
