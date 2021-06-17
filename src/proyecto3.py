import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from clustering import dbscan, mean_shift, gmm, group_classes_2, k_means
from src.dbscan import DBScan
from src.gmm import GMM
from src.k_means import KMeans
from src.mean_shift import MeanShift

df = pd.read_csv('../dataset/dataset_tissue.txt', delimiter=",").T.drop('Unnamed: 0')
y = pd.read_csv('../dataset/clase.txt')

svd = TruncatedSVD(n_components=20)
df_svd = svd.fit_transform(df)

# scaler = MinMaxScaler(feature_range=(0, 10))
# df_svd = scaler.fit_transform(df_svd)

classificator = GMM(k=7)
classificator.fit(df_svd)
classes, clusters = classificator.predict(df_svd).values()
# classes, clusters = k_means(df_svd, k=7)
# classes, clusters = mean_shift(df_svd, radio=100)
# classes, clusters = dbscan(df_svd, radio=100, min_pts=4)
# classes, clusters = gmm(df_svd, n_k=7, max_iter=5)
# gm = KMeans(n_clusters=7, random_state=0, verbose=1).fit(df_svd)
# gm = GaussianMixture(n_components=7, random_state=0, verbose=1).fit(df_svd)
# labels = gm.predict(df_svd)
# classes, clusters = group_classes_2(labels, df_svd)
classes_names = []

for c in classes:
    rep = {x: 0 for x in list(set(y['x']))}
    for elem in classes[c]:
        rep[y.loc[elem, 'x']] += 1
    classes_names.append(max(rep, key=rep.get))

print(len(classes_names))
print(classes_names)

correct = 0
size = 0
for c, label in zip(classes, classes_names):
    for elem in classes[c]:
        if y.loc[elem, 'x'] == label:
            correct += 1
        size += 1


print("Accuracy:", round(correct / size, 4))
