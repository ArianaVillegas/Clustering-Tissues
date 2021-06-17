import numpy as np


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None

    def __dist(self, data, centers):
        distance = np.sum((np.array(centers) - data[:, None, :]) ** 2, axis=2)
        return distance

    def __kmeans_plus_plus(self, X, pdf_method=True):
        centers = []
        X = np.array(X)

        initial_index = np.random.choice(range(X.shape[0]),)
        centers.append(X[initial_index, :].tolist())

        for i in range(self.k - 1):
            distance = self.__dist(X, np.array(centers))
            if i == 0:
                pdf = distance / np.sum(distance)
                centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf.flatten())]
            else:
                dist_min = np.min(distance, axis=1)
                if pdf_method:
                    pdf = dist_min / np.sum(dist_min)
                    centroid_new = X[np.random.choice(range(X.shape[0]), replace=False, p=pdf)]
                else:
                    index_max = np.argmax(dist_min, axis=0)
                    centroid_new = X[index_max, :]

            centers.append(centroid_new.tolist())

        return np.array(centers)

    def fit(self, x):
        self.centroids = self.__kmeans_plus_plus(x, self.k)
        prev = np.zeros((x.shape[0], self.k))
        while np.array_equal(self.centroids, prev) is False:
            classes = [[] for _ in range(self.k)]
            for j in range(x.shape[0]):
                dist = [np.linalg.norm(x[j] - centroid) for centroid in self.centroids]
                class_idx = dist.index(min(dist))
                classes[class_idx].append(j)
            prev = self.centroids
            for i in range(self.k):
                self.centroids[i] = np.mean([x[j] for j in classes[i]], axis=0)

    def predict(self, x):
        classes = {i: [] for i in range(len(self.centroids))}
        for j in range(x.shape[0]):
            dist = [np.linalg.norm(x[j] - centroid) for centroid in self.centroids]
            class_idx = dist.index(min(dist))
            classes[class_idx].append(j)
        return {"classes": classes, "centroids": self.centroids}
