__author__ = 'gregor'
from sklearn import svm

from Nebitno.LloydAlg import *


class Node:
    def __init__(self, indexes, data, left, right):
        self.indexes = indexes.tolist()
        self.count = indexes.__len__()
        self.dim = data[0].__len__()
        self.data = []
        self.wghtCent = self.calculate_wght(data, indexes)
        self.centroid = self.wghtCent / self.count
        self.left = left
        self.right = right

    def calculate_wght(self, data, indexes):
        init = np.ones(self.dim)
        for i in indexes:
            init = init + data[i]
            self.data.append((i, data[i]))
        return init

    def closest_to_centroid(self, candidateSet):
        z_star = candidateSet[0]
        dist = np.linalg.norm(z_star.val() - self.centroid)
        for z in candidateSet:
            if dist > np.linalg.norm(z.val() - self.centroid):
                z_star = z
                dist = np.linalg.norm(z_star.val() - self.centroid)
        return z_star

    def prune(self, candidates):
        z_star = self.closest_to_centroid(candidates)
        pruned = set()
        for z in candidates:
            if z != z_star:
                x = [z.val().tolist(), z_star.val().tolist()]
                y = [0, 1]
                clf = svm.SVC()
                clf.fit(x, y)
                flag = False
                for i in self.data:
                    is_closer = clf.predict([i[1]])
                    if is_closer == 0:
                        flag = True
                        break
                if not flag:
                    pruned.add(z)
        return list(set(candidates) - pruned)


class Candidate:
    def __init__(self, point):
        self.point = point
        self.count = 1
        self.wght_cent = np.copy(point)
        self.indexes = set()

    def add_point(self, index, point):
        self.indexes.add(index)
        self.wght_cent += point
        self.count += 1

    def add_cell(self, node):
        self.indexes |= set(node.indexes)
        self.wght_cent += node.wghtCent
        self.count += node.count

    def recalculate(self, id):
        point = self.point
        self.point = self.wght_cent / self.count
        self.count = 1
        self.wght_cent = np.copy(self.point)
        self.indexes = set()
        print("ID:", id, "  changed: ", np.linalg.norm(point - self.point))
        if np.linalg.norm(point - self.point) < 10E-3:
            return True
        return False

    def __str__(self):
        return self.point.__str__()

    def val(self):
        return self.point
