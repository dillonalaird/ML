import numpy.random as rand
import numpy as np

class NN(object):
    def __init__(self, n_input, n_hidden, n_output, eta=0.01):
        self.w_ih = rand.random(size=(n_input+1, n_hidden))*10
        self.w_ho = rand.random(size=(n_hidden+1, n_output))*10
        self.eta = eta

    def forward_propogate(self, X):
        X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        H = self.sigmoid(np.dot(self.w_ih.T, X.T)).T
        H = np.concatenate((np.ones((H.shape[0],1)), H), axis=1)
        return self.sigmoid(np.dot(self.w_ho.T, H.T)).T

    def forward_propogate_single(self, x):
        x = np.concatenate((np.ones((1,)), x))
        h = self.sigmoid(np.dot(self.w_ih.T, x))
        h = np.concatenate((np.ones((1,)), h))
        return self.sigmoid(np.dot(h, self.w_ho))

    def backward_propogate(self, X, Y, it=100):
        X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        for i in xrange(it):
            for x, y in zip(X, Y):
                h = self.sigmoid(np.dot(self.w_ih.T, x))
                o = self.sigmoid(np.dot(np.concatenate((np.ones(1,), h)), self.w_ho))

                delta_o = o*(1-o)*(y-o)
                delta_h = h*(1-h)*np.sum(np.dot(self.w_ho[1:,:], delta_o))

                self.w_ih += self.eta*np.outer(x[:,np.newaxis], delta_h)
                self.w_ho += self.eta*np.outer(np.concatenate((np.ones(1,), h))
                                               [:,np.newaxis], delta_o)

    def predict(self, wX):
        sig = self.sigmoid(wX)
        return np.array([1 if x > 0.5 else 0 for x in sig])

    def sigmoid(self, wX):
        return 1/(1 + np.exp(-1*wX))
