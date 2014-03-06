import numpy as np

class ALSMatrixFactorization(object):
    def __init__(self, X, k, l):
        self.l = l
        self.k = k
        self.N = X.shape[0]
        self.M = X.shape[1]
        self.X = X
        self.L = np.ones((self.N, k))
        self.R = np.ones((self.M, k))

    def factorize(self, iterations=50):
        self.fLR_vec = []
        for i in xrange(iterations):
            self.updateL()
            self.updateR()
            self.fLR_vec.append(self.fLR())

        return self.L, self.R

    # these for loops can be run in parallel (individually)
    def updateL(self):
        for i in xrange(self.N):
            # this is from Yu, Hsieh, Si and Dhillon's paper on Matrix Factorization
            R_i = np.array([self.R[j,:] for j,v in enumerate(self.X[i,:]) if v != 0])
            self.L[i,:] = np.linalg.solve((R_i.T.dot(R_i)) + self.l * np.eye(self.k), \
                    self.R.T.dot(self.X[i,:].T))

    def updateR(self):
        for i in xrange(self.M):
            # this is from Yu, Hsieh, Si and Dhillon's paper on Matrix Factorization
            L_i = np.array([self.L[j,:] for j,v in enumerate(self.X[:,i]) if v != 0])
            self.R[i,:] = np.linalg.solve((L_i.T.dot(L_i)) + self.l * np.eye(self.k), \
                    self.L.T.dot(self.X[:,i]))

    def fLR(self):
        rmse = 0.
        for i in xrange(self.X.shape[0]):
            for j in xrange(self.X.shape[1]):
                if self.X[i,j] != 0:
                    rmse += (self.X[i,j] - self.L[i,:].dot(self.R[j,:]))**2

        return 0.5 * rmse + 0.5 * self.l * (np.linalg.norm(self.L, 'fro')**2 + \
                np.linalg.norm(self.R, 'fro')**2)

def RMSE(L, R, X):
    n = 0
    rmse = 0.
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            if X[i,j] != 0:
                n += 1
                rmse += (X[i,j] - L[i,:].dot(R[j,:]))**2

    return np.sqrt(rmse / float(n))
