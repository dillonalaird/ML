import random
import numpy as np
import scipy.spatial.distance as dist

class KMeans(object):
    def __init__(self, data, K, centers = None):
        self.K = K
        self.N = data.shape[0]
        # data cluster holds tuples (data point, cluster assigmnet)
        self.data_cluster = map(lambda x: [x, 0], data)

        if centers == None:
            # Pick K centers at random. We need this to be numpy so we can use copy
            self.centers = np.array(random.sample(data, self.K))
        else:
            # Set centers to be centers passed in
            self.centers = centers

    def infer(self):
        self.assign_at_it = []

        it = 0
        while True:
            prev_assign = map(lambda x: x[1], self.data_cluster)

            # This order is taken from marco's post on catalyst
            self.assign_centers()
            self.assign_at_it.append(np.copy(self.data_cluster))

            it += 1
            if it == 20:
                break

            self.calc_centers()

            if self.assignment_no_change(prev_assign):
                break

        return self.data_cluster

    def assignment_no_change(self, prev_assign):
        assign = map(lambda x: x[1], self.data_cluster)
        return prev_assign == assign

    def assign_centers(self):
        for i in xrange(self.N):
            self.data_cluster[i][1] = np.argmin(map( \
                    lambda x: dist.euclidean(x, self.data_cluster[i][0]), self.centers))

    def calc_centers(self):
        for i in xrange(self.K):
            self.centers[i] = np.average(map(lambda x: x[0], filter( \
                    lambda x: x[1] == i, self.data_cluster)), axis=0)

# KMeans++ algorithm for initializing cluster centers
def kmeanspp(data, K):
    centers = [None] * K
    # sample the first cluster center
    centers[0] = random.sample(data, 1)[0]

    for j in xrange(1, K):
        D = map(lambda x: np.min( \
                [dist.euclidean(x, centers[jj]) for jj in xrange(j)]), data)
        # sample from distances to get next cluster center
        centers[j] = data[np.random.choice(data.shape[0], 1, p=(D/sum(D)))[0]]

    return np.array(centers)
