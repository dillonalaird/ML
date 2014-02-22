import numpy as np
import random as rand
import scipy.spatial.distance as dist

class KMeansFast(object):
    def __init__(self, X, K, centers=None):
        self.K = K
        self.N = X.shape[0]
        # Each element of data cluster holds [data point, cluster assignment]
        self.X_cluster = np.array([[x, 0] for x in X])

        if centers == None:
            self.centers = np.array(rand.sample(X, self.K))
        else:
            self.centers = centers

    def fit(self, itr=20):
        for i in xrange(itr):
            prev_assign = np.array([x[1] for x in self.X_cluster])

            self.assign_centers()
            self.calc_centers()

            curr_assign = np.array([x[1] for x in self.X_cluster])

            if np.array_equal(prev_assign, curr_assign):
                break

        return self.X_cluster

    def assign_centers(self):
        for i in xrange(self.N):
            self.X_cluster[i][1] = np.argmin([dist.euclidean(self.X_cluster[i][0], c) \
                    for c in self.centers])

    def calc_centers(self):
        for i in xrange(self.K):
            self.centers[i] = np.average([self.X_cluster[:,1] == i])
