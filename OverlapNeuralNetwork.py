from __future__ import division

import numpy.random as rand
import numpy as np


class ONN(object):
    """
    Overlap Neural Network makes a doubly overlapped neural network meaning it
    makes two neural networks that overlap for some number of neurons.
    """

    def __init__(self, n_input, n_hidden, n_overlap=1, eta=0.01):
        self.w_ih1 = rand.random(size=(n_input+1, n_hidden-n_overlap))
        self.w_ih2 = rand.random(size=(n_input+1, n_hidden-n_overlap))
        self.w_ov  = rand.random(size=(n_input+1, n_overlap))
        self.w_ho  = rand.random(size=((n_hidden-n_overlap)*2+n_overlap+1,1))
        self.eta = eta

    def forward_propogate(self, X):
        X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        H1 = self.sigmoid(np.dot(self.w_ih1.T, X.T)).T
        H2 = self.sigmoid(np.dot(self.w_ih2.T, X.T)).T
        HS = self.sigmoid(np.dot(self.w_ov.T, X.T)).T
        H = np.concatenate((np.ones((X.shape[0],1)), H1, H2, HS), axis=1)
        return self.sigmoid(np.dot(self.w_ho.T, H.T)).T

    def backward_propogate(self, X, Y, it=100):
        X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        for _ in xrange(it):
            for x, y in zip(X, Y):
                h1 = self.sigmoid(np.dot(self.w_ih1.T, x))
                h2 = self.sigmoid(np.dot(self.w_ih2.T, x))
                ov = self.sigmoid(np.dot(self.w_ov.T, x))
                h = np.concatenate((np.ones(1,), h1, h2, ov))
                o  = self.sigmoid(np.dot(h, self.w_ho))

                delta_o = o*(1-o)*(y-o)
                delta_h1 = h1*(1-h1)*np.sum(np.dot(self.w_ho[1:,:], delta_o))
                delta_h2 = h2*(1-h2)*np.sum(np.dot(self.w_ho[1:,:], delta_o))
                delta_ov = ov*(1-ov)*np.sum(np.dot(self.w_ho[1:,:], delta_o))

                self.w_ih1 += self.eta*np.outer(x[:,np.newaxis], delta_h1)
                self.w_ih2 += self.eta*np.outer(x[:,np.newaxis], delta_h2)
                self.w_ov  += self.eta*np.outer(x[:,np.newaxis], delta_ov)
                self.w_ho  += self.eta*np.outer(h[:,np.newaxis], delta_o)

    def sigmoid(self, wX):
        return 1/(1 + np.exp(-1*wX))

def f(x1, x2):
    if x1**2 + x2 > 1:
        return 1
    return 0


def precision(Y_true, Y_pred):
    Y_true = Y_true.flatten()
    pred_arr = Y_pred == 0
    pred = len(Y_pred[pred_arr])
    if pred == 0: return 0
    corr = len(Y_pred[Y_pred[pred_arr] == Y_true[pred_arr]])
    return corr / pred


if __name__ == '__main__':
    N = 1000
    n_input = 2
    n_hidden = 10
    n_overlap = 2

    X = rand.random(size=(N,2))
    Y = np.array([f(x[0], x[1]) for x in X]).reshape(N,1)

    nn = ONN(n_input, n_hidden, n_overlap)
    nn.backward_propogate(X, Y)

    Y_pred = nn.forward_propogate(X)
    Y_pred = np.array(map(lambda x: 1 if x > 0.5 else 0, Y_pred))
    print Y.flatten()
    print Y_pred
    print precision(Y, Y_pred)
