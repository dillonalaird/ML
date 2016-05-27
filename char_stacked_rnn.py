from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import theano
import theano.tensor as T


class CharStackedRNN(object):
    """
    An implemenation of a minimal character-level vanilla stacked recurrent
    neural network (RNN) based off of Andrej Karpathy's min-char-rnn.py using
    Theano.

    Notes
    -----
    A stacked RNN uses 1 or more hidden layers feeding each subsequent hidden
    layer into the next.

    Forward Pass:
    .. math::

        h_t^{(0)} = \tanh(W_{ih}^{(0)} u_t + W_{hh}^{(0)} h_{t-1}^{(0)} + b_h^{(0)})
        h_t^{(1)} = \tanh(W_{ih}^{(1)} h_t^{(0)} + W_{hh}^{(1)} h_{t-1}^{(1)} + b_h^{(1)})
        y_t = softmax(W_{hy} h_t{(1)} + b_y)
    """

    def __init__(self): pass

    @staticmethod
    def make_1ofk(ixs, l, w):
        ret = np.zeros((l, w), dtype=theano.config.floatX)
        for i in xrange(l):
            ret[i,ixs[i]] = 1
        return ret


    def _init_params(self, f_input, n_seq, n_hiddens, learning_rate):
        """
        Initializes internal parameters.

        Parameters
        ----------
        f_input : str
            A file name containing the input text.
        n_seq : int
            The size of each sequence to train on.
        n_hiddens : list of int
            The number of nodes in each hidden layer.
        learning_rate : float
            The learning rate for training the model.

        Notes
        -----
        self.Wih represents the input weights for each layer. For layer 0, this
        is multiplied by the input vector. For layers greater than 0, this is
        multiplied by the previous hidden layer output. For example, a stacked
        RNN with 2 hidden layers would have

                                 u_t                   _________________________
                                  |                    |
                                  V                    V
        h_t0 = tanh(dot(Wih[0], h_lm1) + dot(Whh[0], h_t0m1) + bh[0])
         |_________________________                    _________________________
                                  |                    |
                                  V                    V
        h_t1 = tanh(dot(Wih[1], h_lm1) + dot(Whh[1], h_t1m1) + bh[1])

        Here h_t0 represents h_t for the first hidden layer (input to hidden 0)
        and h_t0m1 is short for h_t0 "minus 1" for h_t0 from the previous
        iteration. Similarly h_t1 is h_t for the second hidden layer (hidden 0
        to hidden 1).
        """

        self.data  = open(f_input, 'r').read()
        chars      = list(set(self.data))
        self.n_vocab       = len(chars)
        self.N             = len(self.data)
        self.n_hiddens     = n_hiddens
        self.n_seq         = n_seq
        self.learning_rate = learning_rate

        self.char_to_ix = {ch:i for i,ch in enumerate(chars)}
        self.ix_to_char = {i:ch for i,ch in enumerate(chars)}

        # input and hidden layers
        self.Wih = {}
        self.Whh = {}
        self.bh  = {}
        layers = [self.n_vocab] + self.n_hiddens
        for l in xrange(len(layers) - 1):
            Wih_init = np.random.uniform(size=(layers[l+1], layers[l]),   low=-.01, high=.01).astype(theano.config.floatX)
            Whh_init = np.random.uniform(size=(layers[l+1], layers[l+1]), low=-.01, high=.01).astype(theano.config.floatX)
            bh_init  = np.zeros((layers[l+1],)).astype(theano.config.floatX)
            Wih = theano.shared(name='Wih'+str(l), value=Wih_init)
            Whh = theano.shared(name='Whh'+str(l), value=Whh_init)
            bh  = theano.shared(name='bh'+str(l), value=bh_init)
            self.Wih[l] = Wih
            self.Whh[l] = Whh
            self.bh[l]  = bh

        Why_init = np.random.uniform(size=(self.n_vocab, layers[-1]), low=-.01, high=.01).astype(theano.config.floatX)
        by_init  = np.zeros((self.n_vocab,)).astype(theano.config.floatX)

        self.Why = theano.shared(name='Why', value=Why_init)
        self.by  = theano.shared(name='by',  value=by_init)

        # keep all the parameters in one place
        self.params = [self.Wih[l] for l in xrange(len(self.Wih))] + \
                      [self.Whh[l] for l in xrange(len(self.Whh))] + \
                      [self.bh[l]  for l in xrange(len(self.bh))]  + \
                      [self.Why, self.by]

    def hidden_layer(self, h_lm1, h_tm1, l):
        return T.tanh(T.dot(self.Wih[l], h_lm1) + T.dot(self.Whh[l], h_tm1) + self.bh[l])

    def train(self, f_input, n_seq, n_hiddens, learning_rate, periodic_print=100):
        """
        Train the stacked RNN.

        Parameters
        ----------
        f_input : str
            A file name containing the input text.
        n_seq : int
            The size of each sequence to train on.
        n_hiddens : list of int
            The number of nodes in each hidden layer.
        learning_rate : float
            The learning rate for training the model.
        periodic_print : int, optional
            How often to print sample and loss in terms of iterations.
        """

        self._init_params(f_input, n_seq, n_hiddens, learning_rate)

        u   = T.matrix(name='u')
        u0  = T.vector(name='u0')
        t   = T.matrix(name='t')
        h0s = [T.vector(name='h0'+str(l)) for l in xrange(len(n_hiddens))]
        k   = T.iscalar(name='k')
        lr  = T.scalar(name='lr')
        rng = RandomStreams()

        # ======================================================================
        #                          TRAINING FUNCTIONS
        # ======================================================================
        def step(u_t, *hs):
            hs_out = []
            h_t = u_t
            for l,h in enumerate(hs):
                h_t = self.hidden_layer(h_t, h, l)
                hs_out.append(h_t)

            y_t = [T.dot(self.Why, h_t) + self.by]
            return hs_out + y_t

        results, _ = theano.scan(step,
                                 sequences=u,
                                 outputs_info=h0s + [None])
        y  = results[-1]
        hs = [h[-1] for h in results[:len(results)-1]]

        prob  = T.nnet.softmax(y)
        loss  = T.nnet.categorical_crossentropy(prob, t).mean()
        grads = T.grad(loss, self.params)
        updates = OrderedDict([(param, param - lr*T.clip(grad, -5, 5))
            for param, grad in zip(self.params, grads)])

        train = theano.function([u, t, lr] + h0s, [loss] + hs, updates=updates)

        # ======================================================================
        #                          SAMPLING FUNCTIONS
        # ======================================================================
        def step_sample(u_t, *hs):
            hs_out = []
            h_t = u_t
            for l,h in enumerate(hs):
                h_t = self.hidden_layer(h_t, h, l)
                hs_out.append(h_t)

            p_t = T.nnet.softmax(T.dot(self.Why, h_t) + self.by)[0]
            ix  = rng.choice(size=(1,), a=T.arange(u_t.shape[0]), p=p_t)
            u_t = T.zeros((u_t.shape[0],))
            u_t = [T.set_subtensor(u_t[ix], 1)]
            return u_t + hs_out

        results_sample, updates_sample = theano.scan(step_sample,
                                                     outputs_info=[u0] + h0s,
                                                     n_steps=k)
        sample = theano.function([u0, k] + h0s, results_sample[0], updates=updates_sample)

        # ======================================================================
        #                          MAIN TRAINING LOOP
        # ======================================================================
        n, p = 0, 0
        smooth_loss = -np.log(1.0/self.n_vocab)*n_seq
        while True:
            if p + self.n_seq + 1 >= self.N or n == 0:
                hprevs = [np.zeros((l,)) for l in n_hiddens]
                p = 0

            inputs_ix  = [self.char_to_ix[ch] for ch in self.data[p:p+self.n_seq]]
            targets_ix = [self.char_to_ix[ch] for ch in self.data[p+1:p+self.n_seq+1]]
            inputs  = CharStackedRNN.make_1ofk(inputs_ix, self.n_seq, self.n_vocab)
            targets = CharStackedRNN.make_1ofk(targets_ix, self.n_seq, self.n_vocab)

            if n % periodic_print == 0:
                seed = CharStackedRNN.make_1ofk(inputs_ix, 1, self.n_vocab)[0]
                out  = sample(seed, 200, *hprevs)
                txt  = ''.join(self.ix_to_char[np.where(v == 1)[0][0]] for v in out)
                print '----\n {} \n----'.format(txt)
                print 'iter: {}, loss: {}'.format(n, smooth_loss)

            loss_h = train(inputs, targets, self.learning_rate, *hprevs)
            loss   = loss_h[0]
            hprevs = loss_h[1:]

            smooth_loss = 0.999*smooth_loss + 0.001*(self.n_seq*loss)

            n += 1
            p += self.n_seq
