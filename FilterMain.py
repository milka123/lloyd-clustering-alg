import pandas as pd

from FilterMethods import *
from OptimalKMethods import *
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'gregor'


#def addRandomSampleNormal(X, mean, cov, count):
#	tmp = np.random.multivariate_normal(mean,cov,count)
#	for row in tmp:
#		X.append([row[0],row[1]])

# X = pd.read_pickle("data2.pkl")[[0, 1, 2]].values
# kdt = spatial.KDTree(X)
# kdtree = create_kdtree(kdt.tree, X)
# print("finish")

N = 500
k = 3
X = pd.read_pickle("bananadata.pkl")[[0, 1]].values
#X = np.matrix()
#addRandomSampleNormal(X, [10,10],[[1,0],[0,1]], 200)
#addRandomSampleNormal(X, [5,10],[[1,0],[0,1]], 200)
#addRandomSampleNormal(X, [10,5],[[1,0],[0,1]], 200)
result, totalDistances = lloyd_clustering_alg(X, k)
mu = [x.val() for x in result]
clusters = {}
for k, cand in enumerate(result):
    indeksi = cand.indexes
    array = [X[index] for index in indeksi]
    clusters[k] = array

colors = ["g", "r", "c", "b", "y"]
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30, 10))

plot_results(colors, mu, clusters, ax1)
f.show()

# totalDistances je dvodimenzionalna lista, totalDistances[i][j] je zbroj udaljenosti svih točaka u klasteru i od svoje
# centroide nakon j koraka algoritma. Od ovog ćemo napraviti grafove
for totalDist in totalDistances:
	totalDist = totalDist/max(totalDist)*100

print("working")
print(totalDistances)

doStuffForOptimalK(X, 7)