import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
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
        c = (np.random.uniform(-1, 1), np.random.uniform(-1, 1),  np.random.uniform(-1, 1))
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


def plot_results(colors, mu, ax):
    for col, center, k in zip(colors, mu, [x for x in range(0, 5)]):
        ax.scatter(np.asarray(clusters[k])[:, 0], np.asarray(clusters[k])[:, 1], c=col)
        ax.scatter(center[0], center[1], c="#000000", marker="x", s=250, linewidth='3')


# ******************************TEST DATA*******************************************
# mu = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
# clusters = np.array([[[1.2, 1.3], [1.1, 1.4], [1.03, 0.9], [0.8, 0.7], [0.6, 1.3]],
#                     [[2.5, 2.0], [2.1, 2.2], [1.9, 2.4], [2.1, 1.7], [1.6, 1.9]],
#                     [[3.6, 2.9], [2.9, 2.8], [3.1, 3.4], [3.1, 2.8], [2.7, 3.4]],
#                     [[4.4, 3.9], [4.1, 4.4], [3.8, 3.9], [4.1, 4.5], [3.8, 4.6]],
#                     [[5.1, 5.1], [5.2, 4.8], [4.7, 4.9], [5.1, 4.8], [5.4, 5.1]]])
# ***********************************************************************************

myarray = np.fromfile('BinaryData.dat',dtype=float)



N = 500
k = 3
colors = ["g", "r", "c", "b", "y"]
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30, 10))

X = init_board_gauss3d(N, k)
mu, clusters = find_centers(X, k)

plot_results(colors, mu, ax1)
f.show()

kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
mu2 = kmeans.cluster_centers_
clusters2 = cluster_points(X, mu2)

plot_results(colors, mu, ax2)
f.show()

print(mu)
print(mu2)
print("done...")
