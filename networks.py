# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:56:22 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

autoencoder-axuiliary-networks
"""
import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

class DiagonalGaussian():
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        
    def log_probability(self, x):
        return -0.5*tf.reduce_sum(np.log(2.0*np.pi) + self.logvar + ((x-self.mu)**2)
                                  /tf.exp(self.logvar), axis = 1)
        
    def sample(self):
        eps = tf.random_normal(tf.shape(self.mu), 0, 1, dtype = tf.float32)
        return self.mu + tf.exp(0.5 * self.logvar)*eps
    
    def repeat(self, n):
        mu = tf.reshape(tf.tile(tf.expand_dims(self.mu, 1), [1, n, 1]), shape = (-1, self.mu.shape[-1]))
        var = tf.reshape(tf.tile(tf.expand_dims(self.logvar, 1), [1, n, 1]), shape = (-1, self.logvar.shape[-1]))
        return DiagonalGaussian(mu, var)
    
    def kl_div(p, q):
        return 0.5*tf.reduce_sum(q.logvar - p.logvar - 1.0 + (tf.exp(p.logvar) + (p.mu - q.mu)**2)/(tf.exp(q.logvar)), axis = 1)

        

