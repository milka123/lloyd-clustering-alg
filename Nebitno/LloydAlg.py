__author__ = 'gregor'

import numpy as np


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) for i in enumerate(mu)], key=lambda t: t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centers(X, K):
    # Initialize to K random centers
    idx = np.random.randint(len(X), size=K)
    oldmu = X[idx, :]
    idx = np.random.randint(len(X), size=K)
    mu = X[idx, :]
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return (mu, clusters)


# -------------------------------------------------------------------------------------------
def init_board(N):
    X = np.array([(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for i in range(N)])
    return X


def init_board_gauss(N, k):
    n = float(N) / k
    X = []
    for i in range(k):
        c = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        s = np.random.uniform(0.05, 0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a, b])
        X.extend(x)
    X = np.array(X)[:N]
    return X


def init_board_gauss3d(N, k):
    n = float(N) / k
    X = []
    for i in range(k):
        c = (np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        s = np.random.uniform(0.05, 0.5)
        x = []
        while len(x) < n:
            a, b, d = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s), np.random.normal(c[2], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1 and abs(d) < 1:
                x.append([a, b, d])
        X.extend(x)
    X = np.array(X)[:N]
    return X


def plot_results(colors, mu, clusters, ax):
    for col, center, k in zip(colors, mu, [x for x in range(0, 5)]):
        ax.scatter(np.asarray(clusters[k])[:, 1], np.asarray(clusters[k])[:, 2], c=col)
        ax.scatter(center[1], center[2], c="#000000", marker="x", s=250, linewidth='3')
