#the purpose of this file is to test feasibility of a bellman
#error. 

import numpy as np
import tensorflow as tf

np.random.seed(4321)
dx1 = np.random.rand(7)
dx2 = np.random.rand(7)

dw1 = np.random.rand(7)
dw2 = np.random.rand(7)

print(np.maximum(np.sum(np.multiply(dx1,dw1))
                , np.sum(np.multiply(dx1,dw1))))

x1 = tf.placeholder(tf.float32, [7])
x2 = tf.placeholder(tf.float32, [7])
W1 = tf.Variable(dw1, dtype=tf.float32)
W2 = tf.Variable(dw2,dtype=tf.float32)
f1 = tf.reduce_sum(tf.multiply(x1, W1))
f2 = tf.reduce_sum(tf.multiply(x2, W2))

f3 = tf.maximum(f1,f2)

gw1 = tf.gradients(f3,W1)
gw2 = tf.gradients(f3,W2)

# sample code to optimize gradients
#params = tf.trainable_variables()
#gradients = tf.gradients(train_loss, params)
#clipped_gradients, _ = tf.clip_by_global_norm(gradients
#                                              , max_gradient_norm)



#tf.global_variables_initializer()

with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    result = sess.run([f3,gw1,gw2], feed_dict={x1:dx1,x2:dx2})
    print(result)
