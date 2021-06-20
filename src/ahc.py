import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy import spatial
from clustering import dbscan, k_means, mean_shift, gmm


df = pd.read_csv('../dataset/dataset_tissue.txt', delimiter=",").T.drop('Unnamed: 0')
y = pd.read_csv('../dataset/clase.txt')



class AHC:
    def  __init__(self, k):
        self.k = k
        self.clusters = {}
        self.clusters_index = {}
        self.centroides = {}
        self.dist = {}
    
    def find_min(self,arr):
        minimos = {}
        for a in arr:
            fila = arr[a]
            key,value =min(fila.items(), key=lambda x: x[1])
            minimos[(a,key)]=value
        key,value =min(minimos.items(), key=lambda x: x[1])  
        return key

    def fit(self, x):
        self.clusters = {i:[x[i]]for i in range(x.shape[0])}
        self.clusters_index = {i:[i]for i in range(x.shape[0])}
        self.centroides = {i:x[i]for i in range(x.shape[0])}
        self.dist = {a:{i:np.linalg.norm(x[a]-x[i])for i in range(x.shape[0])if a!=i} for a in range(x.shape[0])}

    def predict(self, x):
        while len(self.clusters)!=self.k:
            c1, c2= self.find_min(self.dist)
            self.clusters[c1].extend(self.clusters[c2])
            self.clusters_index[c1].extend(self.clusters_index[c2])
            del self.clusters[c2]
            del self.clusters_index[c2]
            del self.centroides[c2]
            del self.dist[c2]
            self.centroides[c1]=np.mean(self.clusters[c1],0)
            for i in self.dist:
                if c1 in self.dist[i]:
                    self.dist[i][c1]=np.linalg.norm(self.centroides[i]-self.centroides[c1])
                del self.dist[i][c2]
            
            print(self.clusters_index)
        return {"classes": self.clusters_index,"centroids":self.centroides}
