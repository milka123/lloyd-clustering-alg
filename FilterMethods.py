from scipy import spatial
import numpy as np
from sklearn import svm
from Filter import Node, Candidate

__author__ = 'gregor'


def create_kdtree(tree, X):
    if hasattr(tree.greater, 'idx'):
        right = Node(tree.greater.idx, X, None, None)
    else:
        right = create_kdtree(tree.greater, X)
    if hasattr(tree.less, 'idx'):
        left = Node(tree.less.idx, X, None, None)
    else:
        left = create_kdtree(tree.less, X)
    indexes = np.concatenate((right.indexes, left.indexes))
    return Node(indexes, X, left, right)


# def filtering(tree, candidates):
#     while

def random_candidates(X, k):
    idx = np.random.randint(len(X), size=k)
    return [Candidate(X[i, :]) for i in idx]


def prune_candidates(tree, candidates):
    candidates = tree.prune(candidates)
    if candidates.__len__() == 1:
        candidates[0].add_cell(tree)
        return
    if tree.right and tree.left:
        prune_candidates(tree.right, candidates)
        prune_candidates(tree.left, candidates)
    else:
        x = [x.val().tolist() for x in candidates]
        y = [i[0] for i in enumerate(candidates)]
        clf = svm.SVC()
        clf.fit(x, y)
        for point in tree.data:
            index = clf.predict([point[1]])
            candidates[index[0]].add_point(point[0], point[1])


def lloyd_clustering_alg(X, k):
    kdt = spatial.KDTree(X)
    kdtree = create_kdtree(kdt.tree, X)
    candidates = random_candidates(X, k)
    while True:
        prune_candidates(kdtree, candidates)
        has_conv = True
        for id, can in enumerate(candidates):
            has_conv = can.recalculate(id) and has_conv
        if has_conv:
            break
    prune_candidates(kdtree, candidates)
    return candidates


def plot_results(colors, mu, clusters, ax):
    for col, center, k in zip(colors, mu, [x for x in range(0, 5)]):
        ax.scatter(np.asarray(clusters[k])[:, 0], np.asarray(clusters[k])[:, 1], c=col)
        ax.scatter(center[0], center[1], c="#000000", marker="x", s=250, linewidth='3')
