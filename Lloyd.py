import timeit

import pandas as pd
from sklearn.cluster import KMeans
from scipy import spatial
import matplotlib.pyplot as plt

from LloydAlg import *


def lloyd_alg(X, k, ax1, colors):
    start = timeit.default_timer()
    mu, clusters = find_centers(X, k)
    stop = timeit.default_timer()
    print("Lloyd algorithm time: ", stop - start)
    plot_results(colors, mu, clusters, ax1)
    ax1.set_title("Lloyd")
    print(mu)


def pyt_kmenas(X, k, ax2, colors):
    start = timeit.default_timer()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    mu2 = kmeans.cluster_centers_
    clusters2 = cluster_points(X, mu2)
    stop = timeit.default_timer()
    print("K-means: ", stop - start)
    plot_results(colors, mu2, clusters2, ax2)
    ax2.set_title("K-means")
    print(mu2)


X = pd.read_pickle("data2.pkl")[[0, 1, 2]].values
# X = pd.read_pickle("data.pkl")[[0, 1]].values
# X = init_board_gauss3d(N, k)

kdt = spatial.KDTree(X)

N = 500
k = 3
colors = ["g", "r", "c", "b", "y"]
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30, 10))

lloyd_alg(X, k, ax1, colors)
pyt_kmenas(X, k, ax2, colors)
f.show()

print("done...")
