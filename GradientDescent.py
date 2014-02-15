import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_l2(X, Y, eta, l, e = None, it = None):
    # initialize w to zeros
    w = np.zeros(X.shape[1])
    N = X.shape[0]

    # I only used X.T, not sure if this is more efficient to store X.T here than calculate
    # the transpose for every iteration
    X_T = X.T.copy()
    log_loss_vec = [np.inf]
    w_0_vec = []

    while True:
        # calculate exp(X.dot(w)) = exp(w_0 + sum(w_i * x_i)) for all data points j in N
        X_w = X.dot(w)
        exp_y_hat = np.exp(X_w)

        # this is for fixing w[0]
        w_0_old = w[0]

        # calculate X_i[Y - P(Y=1 | X, w)] summed over all data points j in N
        xy_diff_sum = X_T.dot(Y - (exp_y_hat / (1 + exp_y_hat)))
        prob_y1 = exp_y_hat / (1 + exp_y_hat)

        w = w + eta * (-l * w + (xy_diff_sum / N))
        # this is to correct the regularization for w_0
        w[0] += eta * l * w_0_old

        ll = log_loss(Y, X_w, N, w, l)
        log_loss_vec.append(ll)
        delta = np.abs(ll - log_loss_vec[len(log_loss_vec) - 2])

        if e != None:
            if delta < e: break

        """
        print("log loss = ", ll)
        l2 = np.sum(w[1:]**2)
        print("l2 norm = ", l2)
        """

        if it != None:
            it -= 1
            if it == 0: break

    x = [i for i in xrange(len(log_loss_vec) - 1)]
    log_loss_vec.remove(np.inf)
    plt.plot(x, log_loss_vec)
    plt.title('iterations vs log loss')
    plt.xlabel('iterations')
    plt.ylabel('log loss')
    plt.show()

    return w

def stochastic_gradient_descent_l2(X, Y, eta, l, it = 10):
    D = X.shape[1]
    w = np.zeros(D)
    N = X.shape[0]

    log_loss_vec = []

    while True:
        for i in xrange(N):
            X_i_w = X[i].dot(w)
            exp_y_hat = np.exp(X_i_w)

            w_0_old = w[0]

            xy_diff_sum = X[i].T.dot(Y[i] - (exp_y_hat / (1 + exp_y_hat)))
            w = w + eta * (-l * w + (xy_diff_sum))
            w[0] += eta * l * w_0_old

            ll = log_loss(Y, X.dot(w), N, w, l)
        """
        log_loss_vec.append(ll)
        print("log loss = ", ll)
        l2 = np.sum(w[1:]**2)
        print("l2 norm = ", l2)
        """

        it -= 1
        if it == 0: break

    x = [i for i in xrange(len(log_loss_vec))]
    plt.plot(x, log_loss_vec)
    plt.title('iterations vs log loss')
    plt.xlabel('iterations')
    plt.ylabel('log loss')
    plt.show()

    return w


def log_loss(Y, X_w, N, w, l):
    # leave out the w_0 term
    l2_norm = (l / 2) * np.sum(w[1:]**2)
    return -((Y.dot(X_w) - np.sum(np.log(1 + np.exp(X_w)))) / N) + l2_norm

def prob_y1(X_i, w):
    return np.exp(X_i.dot(w)) / (1 + np.exp(X_i.dot(w)))

def sse(X, Y, w):
    s = 0
    for i in xrange(Y.shape[0]):
        prob = prob_y1(X[i], w)
        s += (Y[i] - 1) if prob > 0.5 else Y[i]

    return s

def precision(X, Y, w, cl):
    correct = 0
    predict = 0
    for i in xrange(Y.shape[0]):
        prob = prob_y1(X[i], w)
        pred = 1 if prob > 0.5 else 0
        if pred == cl:
            predict += 1
            if pred == Y[i]:
                correct += 1

    return correct / float(predict) if predict != 0.0 else 0.0

def recall(X, Y, w, cl):
    correct = 0
    actual = 0
    for i in xrange(Y.shape[0]):
        prob = prob_y1(X[i], w)
        pred = 1 if prob > 0.5 else 0
        if Y[i] == cl:
            actual += 1
            if pred == Y[i]:
                correct += 1

    return correct / float(actual)

