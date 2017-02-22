import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import timeit

def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) for i in range(K) for c in clusters[i]])

def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)
 
def gap_statistic(X, maxk):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1,maxk)
    Wks = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        print("starting {0}".format(k))
        mu, clusters = find_centers(X,k)
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = maxk
        BWkbs = np.zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([np.random.uniform(xmin,xmax),
                          np.random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            mu, clusters = find_centers(Xb,k)
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk)

def doStuffForOptimalK(X, maxk):


    colors = ["g", "r", "c", "b", "y"]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30, 10))
    ax2.set_ylim([-1,1])
    #lloyd_alg(X, k, ax1)
    #pyt_kmenas(X, k, ax2)
    #f.show()
    print("Doing stuff for optimal K...")
    
    # Milinovo
    
    ks, logWks, logWkbs, sk = gap_statistic(X, maxk)
    kArray = []
    for k in ks:
        kArray.append(k)
    #plt.plot(kArray, np.exp(logWks), 'go-')
    
    bSums = []
    for i in range(1,maxk):
        bSum = sum(logWkbs[0:i])
        bSums.append(bSum/i);
    fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax1.scatter(kArray, logWks, s=10, c='b', marker="s", label='logWk')
    #ax1.plot(kArray, logWks, c='b')
    #ax1.scatter(kArray, bSums, s=10, c='r', marker="o", label='second')
    #ax1.plot(kArray, bSums, c='r')
    
    gapk = []
    print(logWks)
    print(bSums)
    for i in range(0,(maxk-1)):
        gapk.append(bSums[i] - logWks[i])
    ax1.scatter(kArray, gapk, s=10, c='g', marker="o", label='gapk')
    ax1.plot(kArray, gapk, c='g')
    
    gapdiffs = []
    for i in range(maxk-2):
        gapdiffs.append(gapk[i] - gapk[i+1] + sk[i+1])
    
    print(gapdiffs)
    ax2.bar(np.arange(maxk-2), gapdiffs, color='b')
    
    
    plt.show()  



# not sure what of this is needed so i pasted all
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

# myarray = np.fromfile('BinaryData.dat', dtype=float)

def lloyd_alg(X, k, ax1):
    start = timeit.default_timer()
    mu, clusters = find_centers(X, k)
    stop = timeit.default_timer()
    print("Lloyd algorithm time: ", stop - start)
    plot_results(colors, mu, clusters, ax1)
    print(mu)


def pyt_kmenas(X, k, ax2):
    start = timeit.default_timer()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    mu2 = kmeans.cluster_centers_
    clusters2 = cluster_points(X, mu2)
    stop = timeit.default_timer()
    print("Lloyd algorithm time: ", stop - start)
    plot_results(colors, mu2, clusters2, ax2)
    print(mu2)
