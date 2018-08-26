import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
import config

class GRU_theano_batch:

    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.batch_size = config.batch_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(word_dim)

        # next time we will change this!
        ML = np.zeros((word_dim,word_dim))

        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.ML = theano.shared(name='ML', value=ML.astype(theano.config.floatX))

        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c, ML = self.E, self.V, self.U, self.W, self.b, self.c, self.ML
        batch_size = self.batch_size

        mx = T.imatrix('mx')
        my = T.imatrix('my')

        x = T.ivector('x')
        y = T.ivector('y')

        bx = T.ivectors(batch_size)
        by = T.ivectors(batch_size)

        for i in np.arange(batch_size):
            bx[i] = mx[i,:]
            by[i] = my[i,:]

        prediction = T.ivectors(batch_size)
        bce = T.dvectors(batch_size)
        bout = T.dvectors(batch_size)

        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            print "print type of x_t: ", type(x_t)
            # Word embedding layer
            x_e = E[:,x_t]

            # weight for MLE
            weight = ML[:,x_t]

            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t2) + c + weight)[0]

            return [o_t, s_t1, s_t2]

        for bs in np.arange(batch_size):
            # o will be the output vector for each word in vocabulary
            [bout[bs], s, s2], updates = theano.scan(
                forward_prop_step,
                sequences=bx[bs],
                truncate_gradient=self.bptt_truncate,
                outputs_info=[None,
                              dict(initial=T.zeros(self.hidden_dim)),
                              dict(initial=T.zeros(self.hidden_dim))])
            #index prediction
            prediction[bs] = T.argmax(bout[bs], axis=1)
            bce[bs] = T.sum(T.nnet.categorical_crossentropy(bout[bs], by[bs]))

        cost = T.mean(bce) + 0.01*(T.sum(E**2) + T.sum(V**2) + T.sum(U**2) + T.sum(W**2) + T.sum(b**2) + T.sum(c**2))

        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # for minibatch, it goes like this:
        # loop through all samples in batch and get sample derivative
        # accumulative all sample derivative to get batch derivative
        # update all parameters using batch derivative

        # Assign functions
        self.predict_prob = theano.function([mx], bout)
        self.predict_class = theano.function([mx], prediction)
        self.optimization_error = theano.function([mx,my], cost)
        self.cross_entropy_loss = theano.function([mx,my], T.mean(bce))
        self.bptt = theano.function([mx,my], [dE, dU, dW, db, dV, dc])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        #rmsprop
        self.batch_step = theano.function(
            [mx,my,learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

#    def calculate_total_loss(self, X, Y):
#        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
#
    def calculate_loss(self, X, Y):
        # we need to calculate loss on a very large dataset, that is,
        # the dataset with different sequences
        # in order to do so, we cannot use theano function
        # instead we will loop through training (or validation or testing)
        # dataset in teh same way as we did in training, and accumulate
        # error from there
        # the loss calculation should potentially be map reduced

        return np.mean([self.optimization_error(x,y) for x,y in zip(X,Y)])
#

def train_with_batch(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9,
    callback_every=10000, callback=None):
    batch_size = 4
    num_examples_seen = 0
    return model

def save_model_parameters_batch(model, outfile):
    np.savez(outfile,
        E=model.E.get_value(),
        U=model.U.get_value(),
        W=model.W.get_value(),
        V=model.V.get_value(),
        b=model.b.get_value(),
        c=model.c.get_value(),
        ML=model.ML.get_value()
        )
    print "Saved model parameters to %s." % outfile

def load_model_parameters_batch(path, modelClass=GRU_theano_batch):
    npzfile = np.load(path)
    E, U, W, V, b, c, ML = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"], npzfile["ML"]
    hidden_dim, word_dim = E.shape[0], E.shape[1]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    model.ML.set_value(ML)
    return model

#print "total time take ", (time.time()-t1), "  seconds"
#print "done!!"
