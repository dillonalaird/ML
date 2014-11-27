from __future__ import division

import numpy.random as rand
import numpy as np


class DNN(object):
    def __init__(self, layers=[], eta=0.01):
        self.eta = eta
        self.layers = []
        for i in xrange(len(layers)-1):
            self.layers.append(rand.random(size=(layers[i]+1,layers[i+1])))

    def forward_propogate(self, X):
        for layer in self.layers:
            X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
            X = self.sigmoid(np.dot(layer.T, X.T)).T

        return X

    def backward_propogate(self, X, Y, it=100):
        for _ in xrange(it):
            for (x,y) in zip(X, Y):
                outputs = [x]
                for layer in self.layers:
                    x = np.concatenate((np.ones(1,), x))
                    x = self.sigmoid(np.dot(x, layer))
                    outputs.append(x)

                deltas = [0 for _ in xrange(len(outputs)-1)]
                N = len(deltas) - 1
                
                deltas[N] = outputs[N+1]*(y - outputs[N+1])*(1 - outputs[N+1])
                for n in reversed(xrange(N)):
                    deltas[n] = outputs[n+1]*(1 - outputs[n+1])*np.sum(np.dot(
                                self.layers[n+1][1:,:], deltas[n+1]))

                for n in xrange(N + 1):
                    self.layers[n] += self.eta*np.outer(np.concatenate((np.ones(1,),
                                      outputs[n]))[:,np.newaxis], deltas[n])

    def predict(self, wX):
        sig = self.sigmoid(wX)
        return np.array([1 if x > 0.5 else 0 for x in sig])

    def sigmoid(self, wX):
        return 1/(1 + np.exp(-1*wX))


def f(x1, x2):
    y = x1 + x2
    if y > 1:
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
    N = 100
    X = rand.random(size=(N,2))
    Y = np.array([f(x[0], x[1]) for x in X]).reshape(N,1)

    dnn = DNN([2,10,10,1])
    dnn.backward_propogate(X, Y, it=500)
    
    Y_pred = dnn.forward_propogate(X)
    Y_pred = np.array(map(lambda x: 1 if x > 0.5 else 0, Y_pred))
    print Y.flatten()
    print Y_pred
    print precision(Y, Y_pred)
    
