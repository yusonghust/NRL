""" Non-negative matrix factorization (tensorflow)"""

import numpy as np
import tensorflow as tf

class NMF:
    """Compute Non-negative Matrix Factorization (NMF)"""
    def __init__(self, max_iter=200,display_step=1, initW=False):

        self.max_iter = max_iter
        self.display_step = display_step

    def NMF(self, X, r_components, max_iter, display_step, initW, givenW ):
        m,n=np.shape(X)
        tf.reset_default_graph()
        V = tf.placeholder(tf.float32)

        initializer = tf.random_uniform_initializer(1e-7,1)
        if initW is False:
            W =  tf.get_variable(name="W", shape=[m, r_components], initializer=initializer)
            H =  tf.get_variable("H", [r_components, n], initializer=initializer)
        else:
            W =  tf.constant(givenW, shape=[m, r_components], name="W")
            H =  tf.get_variable("H", [r_components, n], initializer=initializer)

        WH =tf.matmul(W, H)
        cost = tf.reduce_mean(tf.square(V - WH))

        """Compute Non-negative Matrix Factorization with Multiplicative Update"""
        Wt = tf.transpose(W)
        H_new = H * tf.clip_by_value(tf.matmul(Wt, V),1e-32,1e32) / tf.clip_by_value(tf.matmul(tf.clip_by_value(tf.matmul(Wt, W),1e-32,1e32), H),1e-32,1e32)
        # H_new = tf.clip_by_value(H * tf.clip_by_value(tf.matmul(Wt, V),1e-32,1e32) / tf.clip_by_value(tf.matmul(tf.clip_by_value(tf.matmul(Wt, W),1e-32,1e32), H),1e-32,1e32),1e-32,1.001)
        H_update = H.assign(H_new)

        if initW is False:
            Ht = tf.transpose(H)
            W_new = W * tf.clip_by_value(tf.matmul(V, Ht),1e-32,1e32)/ tf.clip_by_value(tf.matmul(W, tf.clip_by_value(tf.matmul(H, Ht),1e-32,1)),1e-32,1e32)
            # W_new = tf.clip_by_value(W * tf.clip_by_value(tf.matmul(V, Ht),1e-32,1e32)/ tf.clip_by_value(tf.matmul(W, tf.clip_by_value(tf.matmul(H, Ht),1e-32,1)),1e-22,1e32),1e-22,1.001)
            W_update = W.assign(W_new)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for idx in range(max_iter):
                if initW is False:
                    W_=sess.run(W_update, feed_dict={V:X})
                    H_=sess.run(H_update, feed_dict={V:X})
                else:
                    H_=sess.run(H_update, feed_dict={V:X})

                if (idx % display_step) == 0:
                    # print(sess.run([dW,dH],feed_dict={V:X}))
                    costValue = sess.run(cost,feed_dict={V:X})
                    print("|Epoch:","{:4d}".format(idx), " Cost=","{:.5f}".format(costValue))

        return W_, H_

    def fit_transform(self, X,r_components, initW, givenW):
        """Transform input data to W, H matrices which are the non-negative matrices."""
        W, H =  self.NMF(X=X, r_components = r_components,
                    max_iter = self.max_iter, display_step = self.display_step,
                    initW=initW, givenW=givenW  )
        return W, H

    def inverse_transform(self, W, H):
        """Transform data back to its original space."""
        return np.matmul(W,H)