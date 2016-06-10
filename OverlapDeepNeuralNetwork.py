from __future__ import division

import numpy.random as rand
import numpy as np


class ODNN(object):
    """
    Creates a deep overlapping neural network. It will make two symmetric left
    and right neural networks that share some number of overlap nodes. For
    example if we pass ODNN the following parameters

    layers=[2,4,4] and overlap=[2,2,2] we would get an input layer with two
    nodes A1 and A2

    input   :       A1 A2
    hidden1 : B1 B2 B3 B4 B5 B6

    Here the left neural network consists of A1 and A2 for input and B1, B2, B3
    and B4 for it's hidden layers when the right neural network consists of A1
    and A2 for input and B3, B4, B5 and B6 for hidden layers. B3 and B4 are
    shared between the networks and are determined by the overlap parameter.
    The full network would look like the following

                left   overlap  right
              _________________________
    input   : |       | A1 A2 |       |
    hidden1 : | B1 B2 | B3 B4 | B5 B6 |
    hidden2 : | C1 C2 | C3 C4 | C5 C6 |
    output  : |       |  D1   |       |
              -------------------------
    """

    def __init__(self, layers=[], overlap=[], eta=0.01):
        # error checking
        if len(layers) != len(overlap):
            raise RuntimeError("The length of layers must be the same as overlap.")
        for l, o in zip(layers, overlap):
            if o > l:
                raise RuntimeError("You can't have more overlap units than regular units for a specific layer.")

        self.eta = eta
        self.layers_left = []
        self.layers_right = []
        self.overlap = []

        # hidden layers
        for i in xrange(len(layers)-1):
            self.layers_left.append(rand.random(size=(layers[i]+1, layers[i+1]-overlap[i+1])))
            self.layers_right.append(rand.random(size=(layers[i]+1, layers[i+1]-overlap[i+1])))
            self.overlap.append(rand.random(size=(overlap[i]+1, overlap[i+1])))

        # last layer
        self.layer_out = rand.random(size=(2*(layers[len(layers)-1]-overlap[len(overlap)-1])
            +overlap[len(overlap)-1]+1,1))

    def forward_propogate(self, X):
        HL = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        HR = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        HO = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

        i = 0
        for layer_left, layer_right, overlap in zip(self.layers_left, 
                                    self.layers_right, self.overlap):
            HL = self.sigmoid(np.dot(layer_left.T, HL.T)).T
            HR = self.sigmoid(np.dot(layer_right.T, HR.T)).T
            HO = self.sigmoid(np.dot(overlap.T, HO.T)).T
            
            if i != len(self.overlap) - 1:
                HL = np.concatenate((np.ones((HL.shape[0],1)), HL, HO), axis=1)
                HR = np.concatenate((np.ones((HR.shape[0],1)), HR, HO), axis=1)
                HO = np.concatenate((np.ones((HO.shape[0],1)), HO), axis=1)
            i += 1

        H = np.concatenate((np.ones((HO.shape[0],1)), HL, HR, HO), axis=1)
        return self.sigmoid(np.dot(self.layer_out.T, H.T)).T

    def backward_propogate(self, X, Y, it=100):
        for _ in xrange(it):
            for x, y in zip(X, Y):
                outputs_l = [x]
                outputs_r = [x]
                outputs_o = [x]

                hl = np.concatenate((np.ones(1,), x))
                hr = np.concatenate((np.ones(1,), x))
                ho = np.concatenate((np.ones(1,), x))

                i = 0
                for layer_left, layer_right, overlap in zip(self.layers_left,
                                            self.layers_right, self.overlap):
                    hl = self.sigmoid(np.dot(layer_left.T, hl.T)).T
                    hr = self.sigmoid(np.dot(layer_right.T, hr.T)).T
                    ho = self.sigmoid(np.dot(overlap.T, ho.T)).T

                    if i != len(self.overlap) - 1:
                        outputs_l.append(hl)
                        outputs_r.append(hr)
                        outputs_o.append(ho)

                        hl = np.concatenate((np.ones(1,), hl, ho), axis=1)
                        hr = np.concatenate((np.ones(1,), hr, ho), axis=1)
                        ho = np.concatenate((np.ones(1,), ho), axis=1)

                    i += 1

                h = np.concatenate((np.ones(1,), hl, hr, ho))
                o = self.sigmoid(np.dot(self.layer_out.T, h.T))

                outputs_l.append(o)
                outputs_r.append(o)
                outputs_o.append(o)

                deltas_l = [0 for _ in xrange(len(outputs_l)-1)]
                deltas_r = [0 for _ in xrange(len(outputs_r)-1)]
                deltas_o = [0 for _ in xrange(len(outputs_o)-1)]
                N = len(deltas_o) - 1

                deltas_l[N] = outputs_l[N+1]*(y - outputs_l[N+1])*(1 - outputs_l[N+1])
                deltas_r[N] = outputs_r[N+1]*(y - outputs_r[N+1])*(1 - outputs_r[N+1])
                deltas_o[N] = outputs_o[N+1]*(y - outputs_o[N+1])*(1 - outputs_o[N+1])

                deltas_l[N-1] = outputs_l[N]*(1 - outputs_l[N])*np.sum(np.dot(
                              self.layer_out, deltas_l[N]))
                deltas_r[N-1] = outputs_r[N]*(1 - outputs_r[N])*np.sum(np.dot(
                              self.layer_out, deltas_r[N]))
                deltas_o[N-1] = outputs_o[N]*(1 - outputs_o[N])*np.sum(np.dot(
                              self.layer_out, deltas_o[N]))

                for n in reversed(xrange(N - 1)):
                    deltas_l[n] = outputs_l[n+1]*(1 - outputs_l[n+1])*np.sum(np.dot(
                                  self.layers_left[n+1][1:,:], deltas_l[n+1]))
                    deltas_r[n] = outputs_r[n+1]*(1 - outputs_r[n+1])*np.sum(np.dot(
                                  self.layers_right[n+1][1:,:], deltas_r[n+1]))
                    deltas_o[n] = outputs_o[n+1]*(1 - outputs_o[n+1])*np.sum(np.dot(
                                  self.overlap[n+1][1:,:], deltas_o[n+1]))
                    
                for n in xrange(N):
                        self.layers_left[n]  += self.eta*np.outer(np.concatenate((np.ones(1,),
                                                outputs_l[n]))[:,np.newaxis], deltas_l[n])
                        self.layers_right[n] += self.eta*np.outer(np.concatenate((np.ones(1,),
                                                outputs_r[n]))[:,np.newaxis], deltas_r[n])
                        self.overlap[n]      += self.eta*np.outer(np.concatenate((np.ones(1,),
                                                outputs_o[n]))[:,np.newaxis], deltas_o[n])

                o = np.concatenate((np.ones(1,), outputs_l[N], outputs_r[N], outputs_o[N]))
                self.layer_out += self.eta*np.outer(o[:,np.newaxis], deltas_o[N])
                
    def predict(self, wX):
        pass

    def sigmoid(self, wX):
        return 1/(1 + np.exp(-1*wX))


def f(x1, x2):
    if x1 + x2 > 1:
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
    X = rand.random(size=(N,2))
    Y = np.array([f(x[0], x[1]) for x in X]).reshape(N,1)

    # 2 overlap for the input because input is fed to all networks
    odnn = ODNN([2, 4, 4], [2, 2, 2])
    odnn.backward_propogate(X, Y)
    Y_pred = odnn.forward_propogate(X)
    Y_pred = np.array(map(lambda x: 1 if x > 0.5 else 0, Y_pred))
    print Y.flatten()
    print Y_pred
    print precision(Y, Y_pred)
