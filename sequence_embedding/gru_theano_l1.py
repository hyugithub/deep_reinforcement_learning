import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
import config

# raw data is passed in as gx and by
# for each call, again we have two indices
# pass two arguments from function input, arg1 being starting position
# arg2 being length of sequence
# now we need to turn the slice of data into an array of vectors

class gru_theano_l1:

    def __init__(self, gx, gy, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.batch_size = config.batch_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(word_dim)

        # next time we will change this!
        ML = np.zeros((word_dim,word_dim))

        # load data into shared memory
        self.gx = theano.shared(name='gx', value=gx.astype(theano.config.floatX))
        self.gy = theano.shared(name='gy', value=gy.astype(theano.config.floatX))


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

#        mx = T.imatrix('mx')
#        my = T.imatrix('my')

        start = T.iscalar('start')
        batch_len = T.iscalar('batch_len')

#        x = T.ivector('x')
#        y = T.ivector('y')

#        bx = T.ivectors(batch_size)
#        by = T.ivectors(batch_size)

        bx = T.cast(self.gx[start:start+batch_len*batch_size], dtype='int32')
        by = T.cast(self.gy[start:start+batch_len*batch_size], dtype='int32')

#        for i in np.arange(batch_size):
#            bx[i] = T.cast(self.gx[start+i*batch_len:start+(i+1)*batch_len], dtype='int32')
#            by[i] = T.cast(self.gy[start+i*batch_len:start+(i+1)*batch_len], dtype='int32')

        prediction = T.ivector()
        #bce = T.vector()
#        bout = T.matrix()

        def forward_prop_step(x_t, s_t1_prev):
            # we want to save # of scan calls by batching them
            # this makes the forward step complicated
            # x_t now should be a vector of size batch_size which we got from reshape
            # element of x_t should be integers smaller than self.hidden_dim
            # s_t1_prev should be a matrix of dimension hidden x batch_size

            # Word embedding layer
            #this line does not have to change as long as x_t is a tensor vector
            # but x_e now becomes a matrix
            x_e = E[:,x_t]
#            print "are we here?"
            # weight for MLE
            weight = ML[:,x_t]

            # GRU Layer 1
            # now U[0] need to multiply x_e as a matrix, not as a vector
            # this should be fine since the batch is embedded in the final dimension
            # of the dot

#            U_reshape = U.dot(x_e)
#            UW_reshape = W[0:2].dot(s_t1_prev) + U_reshape[0:2]
#
#            z_t1 = T.nnet.hard_sigmoid(UW_reshape[0]
#                                       + T.reshape(b[0], (b[0].shape[0],1)))
#            r_t1 = T.nnet.hard_sigmoid(UW_reshape[1]
#                                       + T.reshape(b[1],(b[1].shape[0],1)))
#            c_t1 = T.tanh(U_reshape[2] + W[2].dot(s_t1_prev*r_t1) + T.reshape(b[2],(b[2].shape[0],1)))
            U_reshape = U.dot(x_e)
            UW_reshape = W[0:2].dot(s_t1_prev) + U_reshape[0:2]

            zr_t1 = T.nnet.hard_sigmoid(UW_reshape + T.reshape(b[0:2],(2, self.hidden_dim, 1)))
#            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e)
#                                       + W[0].dot(s_t1_prev)
#                                       + T.reshape(b[0], (b[0].shape[0],1)))
#            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e)
#                                       + W[1].dot(s_t1_prev)
#                                       + T.reshape(b[1],(b[1].shape[0],1)))
#            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev*r_t1) + T.reshape(b[2],(b[2].shape[0],1)))
            c_t1 = T.tanh(U_reshape[2] + W[2].dot(s_t1_prev*zr_t1[1]) + T.reshape(b[2],(self.hidden_dim,1)))
#            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            s_t1 = (T.ones_like(zr_t1[0]) - zr_t1[0]) * c_t1 + zr_t1[0] * s_t1_prev

            # GRU Layer 2 -- disabled
#            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
#            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
#            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
#            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            #o_t = T.nnet.softmax(V.dot(s_t1) + c + weight)[0]

            # note that softmax is applied on axis=1, we need to comment out old
            # code and do transpose
            tt = V.dot(s_t1) + T.reshape(c, (self.word_dim,1)) + weight
            o_t = T.nnet.softmax(tt.T).T

#            print "how many dimension does o_t have? ", o_t.ndim
            # IMPORTANT NOTE: dimension of o_t is vocabulary size x batch size
            # dimension of s_t1 is hidden dimension x batch_size
            return [o_t, s_t1]

        def forward_prop_step3(x_t, y_t, s_t1_prev):
            x_e = E[:,x_t]
            weight = ML[:,x_t]
            U_reshape = U.dot(x_e)
            UW_reshape = W[0:2].dot(s_t1_prev) + U_reshape[0:2]

            zr_t1 = T.nnet.hard_sigmoid(UW_reshape + T.reshape(b[0:2],(2, self.hidden_dim, 1)))
            c_t1 = T.tanh(U_reshape[2] + W[2].dot(s_t1_prev*zr_t1[1]) + T.reshape(b[2],(self.hidden_dim,1)))
            s_t1 = (T.ones_like(zr_t1[0]) - zr_t1[0]) * c_t1 + zr_t1[0] * s_t1_prev          

            tt = V.dot(s_t1) + T.reshape(c, (self.word_dim,1)) + weight
            #o_t is vocab x batch_size
#            o_t = T.nnet.softmax(tt.T).T

            o_t_clip = T.nnet.softmax(tt.T).T

            o_t = theano.gradient.grad_clip(T.clip(o_t_clip,1e-8,.99999999), -10.0, 10.0)
            # the index below is a list of probabilities
            # they need to be logged and summed up
            return [o_t[y_t, T.arange(self.batch_size)], s_t1]

        def forward_prop_step2(x_t, s_t1_prev):
            x_e = E[:,x_t]
            weight = ML[:,x_t]
            U_reshape = U.dot(x_e)
            UW_reshape = W[0:2].dot(s_t1_prev) + U_reshape[0:2]

            zr_t1 = T.nnet.hard_sigmoid(UW_reshape + T.reshape(b[0:2],(2, self.hidden_dim, 1)))
#            z_t1 = zr_t1[0]
#            r_t1 = zr_t1[1]
#            return [z_t1, r_t1]
            c_t1 = T.tanh(U_reshape[2] + W[2].dot(s_t1_prev*zr_t1[1]) + T.reshape(b[2],(self.hidden_dim,1)))

            s_t1 = (T.ones_like(zr_t1[0]) - zr_t1[0]) * c_t1 + zr_t1[0] * s_t1_prev
            tt = V.dot(s_t1) + T.reshape(c, (self.word_dim,1)) + weight
            o_t = T.nnet.softmax(tt.T).T

#            print "how many dimension does o_t have? ", o_t.ndim
            # IMPORTANT NOTE: dimension of o_t is vocabulary size x batch size
            # dimension of s_t1 is hidden dimension x batch_size
            return [o_t, s_t1]
#            return T.argmax(o_t, axis=0)

        x_debug = T.ivector()
        s_t_debug = T.dmatrix()
#        [d1,d2] =
        self.debug2 = theano.function([x_debug, s_t_debug], forward_prop_step2(x_debug, s_t_debug), on_unused_input='warn', name="TDF DEBUG2")

#        print self.debug2(np.zeros(batch_size,dtype='int32'), np.zeros(self.hidden_dim))

        # every step forward_prop_step takes a vector of batch_size
        # output bout takes three dimensions:
        # sequence length x vocabulary size x batch
        [bout, s], updates = theano.scan(forward_prop_step,
                                            sequences=bx.reshape((bx.shape[0]/batch_size, batch_size)),
                                            truncate_gradient=self.bptt_truncate,
                                            outputs_info=[None,
                                                          dict(initial=T.zeros((self.hidden_dim,self.batch_size)))
                                                          ],
                                         name="TDF FORWARD SCAN"
                                          )

        # at this point, we have by and bout
        # by is a vector ordered by batch_size x sequence
        # bout is sequence x vocabulary size x batch

        #index prediction
        prediction = T.argmax(bout, axis=1)

        # first we need bout to be like batch_size x sequence x vocab size
        # then we flatten it
        # FIX LATER !!! no need to shuffle things if we switch by to a different order
        bout2 = bout.dimshuffle([2,0,1])
#        bout3 = bout2.reshape((bout2.shape[0]*bout2.shape[1], bout2.shape[2]))
        bce = T.nnet.categorical_crossentropy(bout2.reshape((bout2.shape[0]*bout2.shape[1], self.word_dim)), by)

#        cost = T.mean(bce)*batch_len + 0.01*(T.sum(E**2) + T.sum(V**2) + T.sum(U**2) + T.sum(W**2) + T.sum(b**2) + T.sum(c**2))
#        cost = T.mean(bce)*batch_len
#        print "dimensions of cost: ", cost.ndim

        #alternative models to get here
        [bout3, _], updates = theano.scan(forward_prop_step3,
                                            sequences=[bx.reshape((bx.shape[0]/batch_size, batch_size)),by.reshape((by.shape[0]/batch_size, batch_size))],
                                            truncate_gradient=self.bptt_truncate,
                                            outputs_info=[None,
                                                          dict(initial=T.zeros((self.hidden_dim,self.batch_size)))
                                                          ],
                                         name="TDF FORWARD SCAN ALT"
                                          )
        # bout3 is sequence x batch
        cost = T.sum(-T.log(bout3)) + 0.01*(T.sum(E**2) + T.sum(V**2) + T.sum(U**2) + T.sum(W**2) + T.sum(b**2) + T.sum(c**2))

#        cost_tmp = T.sum(-T.log(bout3))
#        cost = theano.gradient.grad_clip(cost_tmp,-10.0,10.0)
        self.cost = cost

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
        self.predict_prob = theano.function([start,batch_len], bout, name="TDF PREDICT PROB")
        self.predict_class = theano.function([start,batch_len], prediction, name="TDF PREDICT CLASS")
        self.optimization_error = theano.function([start,batch_len], [cost,bout3], name="TDF OPT ERROR")
        self.cross_entropy_loss = theano.function([start,batch_len], T.mean(bce), name="TDF CROSS ENTROPY LOSS")
        self.bptt = theano.function([start,batch_len], [dE, dU, dW, db, dV, dc], name="TDF BPTT")

        self.debug = theano.function([start,batch_len], bout3, name="TDF DEBUG")

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
            [start,batch_len,learning_rate, theano.In(decay, value=0.9)],
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
                    ],
            name="TDF BATCH STEP"
            )


        def forward_prop_step_scalar(x_t, s_t1_prev):
            # scalar (original version)
            x_e = E[:,x_t]
            weight = ML[:,x_t]
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev*r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
#            o_t = T.nnet.softmax(V.dot(s_t1) + c + weight)
 
            tt = T.nnet.softmax(V.dot(s_t1) + c + weight)
            o_t = theano.gradient.grad_clip(T.clip(tt,1e-8,.99999999), -10.0, 10.0)
#            print "how many dimension does o_t have? ", o_t.ndim
            # IMPORTANT NOTE: dimension of o_t is vocabulary size x batch size
            # dimension of s_t1 is hidden dimension x batch_size
            return [o_t, s_t1]

        #calculate cross entropy on one sample
        tx = T.ivector()
        ty = T.ivector()
        [tout, _], _ = theano.scan(forward_prop_step_scalar,
                                    sequences=tx,
                                    truncate_gradient=self.bptt_truncate,
                                    outputs_info=[None,
                                                  dict(initial=T.zeros(self.hidden_dim))
                                                  ]
                                  , name="TDF SAMPLE FORWARD"
                                  )
        sce = T.sum(T.nnet.categorical_crossentropy(T.reshape(tout, (tout.shape[0]*tout.shape[1], tout.shape[2])),ty))
        self.example_loss = theano.function([tx,ty], sce, on_unused_input='warn', name="TDF SAMPLE LOSS")
#        self.example_predict_prob = theano.function([tx,ty],tout)
        self.example_prediction = theano.function([tx,ty],[tout, T.argmax(tout, axis=2), sce], name="TDF SAMPLE PREDICTION")
        self.sample_scalar_forward = theano.function([tx],tout, name="TDF SCALAR FORWARD")
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

#def save_model_parameters_batch2(model, outfile):
#    np.savez(outfile,
#        E=model.E.get_value(),
#        U=model.U.get_value(),
#        W=model.W.get_value(),
#        V=model.V.get_value(),
#        b=model.b.get_value(),
#        c=model.c.get_value(),
#        ML=model.ML.get_value()
#        )
#    print "Saved model parameters to %s." % outfile
#
#def load_model_parameters_batch2(path, modelClass=gru_theano_l1):
#    npzfile = np.load(path)
#    E, U, W, V, b, c, ML = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"], npzfile["ML"]
#    hidden_dim, word_dim = E.shape[0], E.shape[1]
#    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
#    sys.stdout.flush()
#    model = modelClass(word_dim, hidden_dim=hidden_dim)
#    model.E.set_value(E)
#    model.U.set_value(U)
#    model.W.set_value(W)
#    model.V.set_value(V)
#    model.b.set_value(b)
#    model.c.set_value(c)
#    model.ML.set_value(ML)
#    return model

#print "total time take ", (time.time()-t1), "  seconds"
#print "done!!"
