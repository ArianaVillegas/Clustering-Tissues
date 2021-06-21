import pandas as pd

from sklearn.decomposition import TruncatedSVD

from src.ahc import AHC
from src.dbscan import DBScan
from src.gmm import GMM
from src.k_means import KMeans
from src.mean_shift import MeanShift

df = pd.read_csv('dataset/dataset_tissue.txt', delimiter=",").T.drop('Unnamed: 0')
y = pd.read_csv('dataset/clase.txt')

svd = TruncatedSVD(n_components=50)
df_svd = svd.fit_transform(df)

classificator = AHC(k=7)
classificator.fit(df_svd)
classes, clusters = classificator.predict(df_svd).values()
classes_names = []

for c in classes:
    rep = {x: 0 for x in list(set(y['x']))}
    for elem in classes[c]:
        rep[y.loc[elem, 'x']] += 1
    classes_names.append(max(rep, key=rep.get))

print(len(set(classes_names)))
print(set(classes_names))

correct = 0
size = 0
for c, label in zip(classes, classes_names):
    for elem in classes[c]:
        if y.loc[elem, 'x'] == label:
            correct += 1
        size += 1


print("Accuracy:", round(correct / size, 4))
