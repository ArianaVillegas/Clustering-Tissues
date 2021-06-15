import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from clustering import dbscan, k_means, mean_shift, gmm

df = pd.read_csv('../dataset/dataset_tissue.txt', delimiter=",").T.drop('Unnamed: 0')
y = pd.read_csv('../dataset/clase.txt')

svd = TruncatedSVD(n_components=20)
df_svd = svd.fit_transform(df)

scaler = StandardScaler()
scaler.fit(df_svd)
scaler.transform(df_svd)

# classes, clusters = k_means(df_svd, k=7)
# classes, clusters = mean_shift(df_svd, radio=100)
# classes, clusters = dbscan(df_svd, radio=100, min_pts=4)
classes, clusters = gmm(df_svd, n_k=7, max_iter=20)
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
