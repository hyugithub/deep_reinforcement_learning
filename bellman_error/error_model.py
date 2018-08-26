#the purpose of this file is to generate a simple functional
# approximator to V(s,t)

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import time

class error_model_linear:
    def __init__(self):
        self.weights = tf.Variable(np.random.uniform(size=[dim_state_space]), dtype=tf.float32)
        self.bias = tf.Variable(np.random.uniform(size=1), dtype=tf.float32)
        self.boundary_error = tf.constant(0.0, dtype=tf.float32)
        self.bellman_error = tf.constant(0.0, dtype=tf.float32)
        self.value_lhs = tf.constant(0.0, dtype=tf.float32)
        self.value_rhs_1 = tf.constant(0.0, dtype=tf.float32)
        self.value_rhs_2 = tf.constant(0.0, dtype=tf.float32)
        
        #TODO: tune thse two parameters
        self.lambda_s0 = 1e-3
        self.lambda_t0 = 1e-3

        self.state_lhs = tf.placeholder(tf.float32, [batch_size, dim_state_space])
        #V(s-a(p),t-1)
        self.state_rhs_1 = tf.placeholder(tf.float32, [batch_size, num_product, dim_state_space])
        #V(s,t-1)
        self.state_rhs_2 = tf.placeholder(tf.float32, [batch_size, dim_state_space])
        self.mask = tf.placeholder(tf.float32, [batch_size, num_product])        
        self.loss = self.generate_bellman_error() + self.generate_boundary_error(self.state_lhs)
        self.train_step =tf.train.AdagradOptimizer(0.3).minimize(self.loss)
        
    #we also need to define boundary conditions for V(s,0) = 0
    def generate_boundary_error(self,s):
        weights_t0 = tf.multiply(self.weights, tf.constant(np.concatenate([np.ones(num_nights), np.zeros(1)]), dtype=tf.float32))
        v_t0 = tf.reduce_sum(tf.multiply(self.state_lhs, weights_t0), axis=1) + self.bias
                
        boundary_t0 = tf.multiply(tf.reduce_mean(tf.multiply(v_t0,v_t0)), tf.constant(self.lambda_t0, dtype=tf.float32))
        
        #V(0,t) = 0
        weights_s0 = tf.multiply(self.weights, tf.constant(np.concatenate([np.zeros(num_nights), np.ones(1)]), dtype=tf.float32))
        v_s0 = tf.reduce_sum(tf.multiply(self.state_lhs, weights_s0), axis=1) + self.bias
                
        boundary_s0 = tf.multiply(tf.reduce_mean(tf.multiply(v_s0,v_s0)), tf.constant(self.lambda_s0, dtype=tf.float32))
        self.boundary_error = boundary_t0 + boundary_s0        
        return self.boundary_error

    def generate_bellman_error(self):
        #define LHS
        value_lhs = tf.reduce_sum(tf.multiply(self.state_lhs, self.weights), axis=1) + self.bias
        
        #V(s,t-1) as a matrix of batch x 1
        value_rhs_2 = tf.reduce_sum(tf.multiply(self.state_rhs_2, self.weights), axis=1) + self.bias
        
        # this is a long definition for the sum max calculation done in multiple steps
        #V(s-a(p),t-1) for every p, dimension is batch x product
        value_rhs_1 = tf.reduce_sum(tf.multiply(self.state_rhs_1, self.weights), axis=2) + self.bias
        #V(s-a(p),t-1) - V(s,t-1) + r(p)
        value_rhs_1 = value_rhs_1 - tf.reshape(value_rhs_2, [batch_size,-1]) + tf.constant(product_revenue, dtype=tf.float32)
        # max(x,0)
        value_rhs_1 = tf.maximum(value_rhs_1, tf.constant(0.0))
        #we need the self.mask here because certain products are unsellable given
        #a certain state. To implement this logic, we do two things:
        # 1. setting self.mask = 0 for such state/product combination
        # 2. in data preparation setting that state to 0 
        #in this way, no error should come up in approximator
        #and no impact on gradient estimator
        value_rhs_1 = tf.multiply(value_rhs_1, self.mask)
        #prob*max
        value_rhs_1 = tf.multiply(value_rhs_1
                                , tf.constant(product_prob
                                              , dtype=tf.float32))
        #sum (prob*max)
        value_rhs_1 = tf.reduce_sum(value_rhs_1, axis=1)
        #V(s,t-1) + sum pr*max(*)
        print(value_rhs_1)
        print(value_rhs_2)
        value_rhs = value_rhs_1 + value_rhs_2
        
        bellman_error = value_lhs-value_rhs
        self.bellman_error = tf.reduce_mean(tf.multiply(bellman_error,bellman_error))
        self.value_lhs = value_lhs 
        self.value_rhs_1 = value_rhs_1
        self.value_rhs_2 = value_rhs_2
        return self.bellman_error, self.value_lhs, self.value_rhs_1, self.value_rhs_2, self.weights, self.bias
        #EOF
    def train(self, session, data_lhs, data_rhs_1, data_rhs_2, data_mask):
        session.run(self.train_step
                , feed_dict={self.state_lhs: data_lhs,
                             self.state_rhs_1: data_rhs_1,
                             self.state_rhs_2: data_rhs_2,
                             self.mask: data_mask
                             })        
    #EOC
