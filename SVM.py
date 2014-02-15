import numpy as np

class SVM(object):

    def __init__(self):
        pass

    def learn(self, eta, C, X, Y):
        self.w = np.zeros(X.shape[1])
        self.w0 = 0

        N = len(Y)
        pred_vec = []
        for i in xrange(N):
            X_i_w = X[i].dot(self.w)

            ind = self.ind(Y[i], X_i_w, self.w0)
            pred_vec.append(self.pred(X[i]))
            self.w += eta * (C * ind * Y[i] * X[i] - 2 * self.w)
            self.w0 += eta * C * ind * Y[i]

        return pred_vec

    def ind(self, y, xw, w0):
        return 1 if y * (xw + w0) <= 1 else 0

    def pred(self, x):
        return 1 if self.w.dot(x) + self.w0 > 0 else -1

    def run(self, data):
        return map(lambda x: 1 if x > 0 else -1, data.dot(self.w) + self.w0)

