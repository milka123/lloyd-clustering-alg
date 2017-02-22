import pandas as pd

from FilterMethods import *
import matplotlib.pyplot as plt

__author__ = 'gregor'

# X = pd.read_pickle("data2.pkl")[[0, 1, 2]].values
# kdt = spatial.KDTree(X)
# kdtree = create_kdtree(kdt.tree, X)
# print("finish")

# N = 500
k = 3
X = pd.read_pickle("data2.pkl")[[0, 1, 2]].values
result = lloyd_clustering_alg(X, k)
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

print("working")
