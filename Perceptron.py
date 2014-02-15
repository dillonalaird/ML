import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self):
        pass

    def learn(self, kernel, X, Y):
        N = len(Y)
        pred_vec = []
        self.kernel = kernel
        self.mistakes = []
        for i in xrange(N):
            yhat = self.calc_yhat(X[i])
            pred_vec.append(yhat)

            if yhat != Y[i]:
                self.mistakes.append((Y[i], X[i]))

        # return vector of predictions
        return pred_vec

    def calc_yhat(self, x):
        yhat = 0
        for xy in self.mistakes:
            # xy[0] = Y and xy[1] = X
            yhat += xy[0] * self.kernel(x, xy[1])

        return 1 if yhat > 0 else -1

    def run(self, data):
        return map(lambda x: self.calc_yhat(x), data)

