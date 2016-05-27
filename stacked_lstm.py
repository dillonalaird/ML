from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import cPickle as pkl
import theano
import theano.tensor as T


#mode = theano.compile.mode.Mode(linker='py', optimizer='fast_compile')
mode = theano.compile.mode.Mode(linker='cvm', optimizer='fast_run')


class SLSTM:
    @staticmethod
    def init(n_input, n_hiddens):
        np.random.seed(123)
        layers = [n_input] + n_hiddens
        WSLSTM = {}
        for l in xrange(len(layers) - 1):
            WSLSTM[l] = {'W': {}, 'U': {}, 'b': {}}
            WSLSTM[l]['W']['input']  = theano.shared(np.random.uniform(size=(layers[l+1], layers[l]),
                                                                       low=-.01, high=.01).astype(theano.config.floatX))
            WSLSTM[l]['W']['forget'] = theano.shared(np.random.uniform(size=(layers[l+1], layers[l]),
                                                                       low=-.01, high=.01).astype(theano.config.floatX))
            WSLSTM[l]['W']['output'] = theano.shared(np.random.uniform(size=(layers[l+1], layers[l]),
                                                                       low=-.01, high=.01).astype(theano.config.floatX))
            WSLSTM[l]['W']['state']  = theano.shared(np.random.uniform(size=(layers[l+1], layers[l]),
                                                                       low=-.01, high=.01).astype(theano.config.floatX))

            WSLSTM[l]['U']['input']  = theano.shared(np.random.uniform(size=(layers[l+1], layers[l+1]),
                                                                       low=-.01, high=.01).astype(theano.config.floatX))
            WSLSTM[l]['U']['forget'] = theano.shared(np.random.uniform(size=(layers[l+1], layers[l+1]),
                                                                       low=-.01, high=.01).astype(theano.config.floatX))
            WSLSTM[l]['U']['output'] = theano.shared(np.random.uniform(size=(layers[l+1], layers[l+1]),
                                                                       low=-.01, high=.01).astype(theano.config.floatX))
            WSLSTM[l]['U']['state']  = theano.shared(np.random.uniform(size=(layers[l+1], layers[l+1]),
                                                                       low=-.01, high=.01).astype(theano.config.floatX))
            WSLSTM[l]['b']['input']  = theano.shared(np.zeros((layers[l+1],)).astype(theano.config.floatX))
            WSLSTM[l]['b']['forget'] = theano.shared(np.zeros((layers[l+1],)).astype(theano.config.floatX))
            WSLSTM[l]['b']['output'] = theano.shared(np.zeros((layers[l+1],)).astype(theano.config.floatX))
            WSLSTM[l]['b']['state']  = theano.shared(np.zeros((layers[l+1],)).astype(theano.config.floatX))

        return WSLSTM

    @staticmethod
    def train(data, WSLSTM, OUT, learning_rate, h_inits, c_inits,
              temperature=1., periodic_print=100, save_file='lstm.model'):
        x   = T.matrix(name='x',  dtype=theano.config.floatX)
        t   = T.matrix(name='t',  dtype=theano.config.floatX)
        x0  = T.vector(name='x0', dtype=theano.config.floatX)
        lr  = T.scalar(name='lr', dtype=theano.config.floatX)
        k   = T.iscalar(name='k')
        rng = RandomStreams()

        layers = [l for l in xrange(len(WSLSTM.keys()))]
        hc0s = [[T.vector(name='h0'+str(l), dtype=theano.config.floatX),
                 T.vector(name='c0'+str(l), dtype=theano.config.floatX)]
                for l in layers]
        hc0s = [item for sublist in hc0s for item in sublist]

        params = [[WSLSTM[l][w]['input'], WSLSTM[l][w]['forget'],
                   WSLSTM[l][w]['output'], WSLSTM[l][w]['state']]
                  for w in ['W', 'U', 'b'] for l in layers] + \
                 [[OUT['W'], OUT['b']]]
        params = [item for sublist in params for item in sublist]

        # hs_cs holds the previous hidden and state vectors in pairs for each
        # layer. For example if we had two layers then we'd have
        #   hs_cs = [h1_tm1, c1_tm1, h2_tm1, c2_tm1]
        # where h1 and c1 are the hidden and state vectors for layer one and h2
        # and c2 are the hidden and state vectors for layer two.
        def step(x_t, *hs_cs):
            hs_cs_out = []
            for l in layers:
                h_tm1 = hs_cs[2*l+0]
                c_tm1 = hs_cs[2*l+1]

                i_t = T.nnet.sigmoid(T.dot(WSLSTM[l]['W']['input'], x_t)  +
                        T.dot(WSLSTM[l]['U']['input'], h_tm1)  + WSLSTM[l]['b']['input'])
                f_t = T.nnet.sigmoid(T.dot(WSLSTM[l]['W']['forget'], x_t) +
                        T.dot(WSLSTM[l]['U']['forget'], h_tm1) + WSLSTM[l]['b']['forget'])
                o_t = T.nnet.sigmoid(T.dot(WSLSTM[l]['W']['output'], x_t) +
                        T.dot(WSLSTM[l]['U']['output'], h_tm1) + WSLSTM[l]['b']['output'])
                c_t = T.tanh(T.dot(WSLSTM[l]['W']['state'], x_t) +
                        T.dot(WSLSTM[l]['U']['state'], h_tm1) + WSLSTM[l]['b']['state'])

                c_t = f_t*c_tm1 + i_t*c_t
                h_t = o_t*T.tanh(c_t)

                hs_cs_out.append(h_t)
                hs_cs_out.append(c_t)

                x_t = h_t

            y_t = [T.dot(OUT['W'], h_t) + OUT['b']]
            return hs_cs_out + y_t

        results, _ = theano.scan(step,
                                 sequences=x,
                                 outputs_info=hc0s + [None])
        y = results[-1]
        hs_cs = [hc[-1] for hc in results[:len(results)-1]]

        prob  = T.nnet.softmax(y)
        loss  = T.nnet.categorical_crossentropy(prob, t).mean()
        grads = T.grad(loss, params)
        updates = OrderedDict([(param, param - lr*grad)
            for param, grad in zip(params, grads)])
        train = theano.function([x, t, lr] + hc0s,
                                [loss] + hs_cs,
                                updates=updates,
                                allow_input_downcast=True)

        def step_sample(x_t, *hs_cs):
            hs_cs_out = []
            for l in layers:
                h_tm1 = hs_cs[2*l+0]
                c_tm1 = hs_cs[2*l+1]

                i_t = T.nnet.sigmoid(T.dot(WSLSTM[l]['W']['input'], x_t)  +
                        T.dot(WSLSTM[l]['U']['input'], h_tm1)  + WSLSTM[l]['b']['input'])
                f_t = T.nnet.sigmoid(T.dot(WSLSTM[l]['W']['forget'], x_t) +
                        T.dot(WSLSTM[l]['U']['forget'], h_tm1) + WSLSTM[l]['b']['forget'])
                o_t = T.nnet.sigmoid(T.dot(WSLSTM[l]['W']['output'], x_t) +
                        T.dot(WSLSTM[l]['U']['output'], h_tm1) + WSLSTM[l]['b']['output'])
                c_t = T.tanh(T.dot(WSLSTM[l]['W']['state'], x_t) +
                        T.dot(WSLSTM[l]['U']['state'], h_tm1) + WSLSTM[l]['b']['state'])

                c_t = f_t*c_tm1 + i_t*c_t
                h_t = o_t*T.tanh(c_t)

                hs_cs_out.append(h_t)
                hs_cs_out.append(c_t)

                x_t = h_t

            p_t = T.dot(OUT['W'], h_t) + OUT['b']
            p_t = T.nnet.softmax(p_t/temperature)[0]
            ix  = rng.choice(size=(1,), a=T.arange(p_t.shape[0]), p=p_t)
            x_t = T.zeros((p_t.shape[0],))
            x_t = [T.set_subtensor(x_t[ix], 1.)]
            return x_t + hs_cs_out

        results_sample, updates_sample = theano.scan(step_sample,
                                                     outputs_info=[x0] + hc0s,
                                                     n_steps=k)
        sample = theano.function([x0, k] + hc0s,
                                 results_sample[0],
                                 updates=updates_sample,
                                 allow_input_downcast=True)


        n = 0
        smooth_loss = -np.log(1.0/data.n_vocab)*data.n_seq
        while True:
            hcprevs = []
            for h_init, c_init in zip(h_inits, c_inits):
                hcprevs.append(h_init)
                hcprevs.append(c_init)

            for x_t, y_t in data:
                loss_hcprevs = train(x_t, y_t, learning_rate, *hcprevs)
                loss = loss_hcprevs[0]
                hcprevs = loss_hcprevs[1:]
                smooth_loss = 0.999*smooth_loss + 0.001*(data.n_seq*loss)

                if n % periodic_print == 0:
                    seed = x_t[0]
                    out  = sample(seed, 200, *hcprevs)
                    txt  = data.one_hot2text(out)
                    print '----\n {} \n----'.format(txt)
                    print 'iter: {}, loss: {}'.format(n, smooth_loss)

                    with open(save_file, 'wb') as f:
                        pkl.dump({'WSLSTM': WSLSTM, 'OUT': OUT}, f, -1)

                n += 1
