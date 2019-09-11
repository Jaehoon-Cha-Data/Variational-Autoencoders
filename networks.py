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

class MADE(object):
    def __init__(self, n_input, n_hidden,):
        self.n_input = n_input      # n_latent
        self.n_hidden = n_hidden    # should equal to before latent nodes
        self.network_weights = self.Initialize_weights()
        
        self.W_mask, self.V_mask = self.Generate_mask()
        
    def Initialize_weights(self):
        all_weights = dict()
        all_weights['W'] = tf.Variable(xavier_init(self.n_input + self.n_hidden, self.n_hidden), name = "MMDW")
        all_weights['b'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32), name = "MMDB")
        all_weights['V_s'] = tf.Variable(xavier_init(self.n_hidden, self.n_input), name = "MMDVs")
        all_weights['V_m'] = tf.Variable(xavier_init(self.n_hidden, self.n_input), name = "MMDVm")
        all_weights['c_s'] = tf.Variable(tf.ones([self.n_input], dtype = tf.float32) * 2.0, name = "MMCs")
        all_weights['c_m'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32), name = "MMCm")
        return all_weights


    def Generate_mask(self):
        max_masks = np.random.randint(low = 1, high = self.n_input, size = self.n_hidden)
        W_mask = np.fromfunction(lambda d, k: max_masks[k] >= d+1, 
                                 (self.n_input+self.n_hidden, self.n_hidden),
                                 dtype = int).astype(np.float32)
        W_mask = tf.Variable(W_mask, trainable=False)

        V_mask = np.fromfunction(lambda k, d: d + 1 > max_masks[k], 
                                 (self.n_hidden, self.n_input),
                                 dtype=int).astype(np.float32)
        V_mask = tf.Variable(V_mask, trainable=False)

        return W_mask, V_mask
    
    def Apply_mask(self):
        self.network_weights['W'] = tf.multiply(self.network_weights['W'], self.W_mask)
        self.network_weights['V_s'] = tf.multiply(self.network_weights['V_s'], self.V_mask)
        self.network_weights['V_m'] = tf.multiply(self.network_weights['V_m'], self.V_mask)
        
    def Forward(self, z, h):
        self.Apply_mask()
        tf.concat([z, h], axis = 1) 
        
        x = tf.add(tf.matmul(tf.concat([z, h], axis = 1), self.network_weights['W']), self.network_weights['b'])
        x = tf.nn.relu(x)
        
        m = tf.add(tf.matmul(x, self.network_weights['V_m']), self.network_weights['c_m'])
        s = tf.add(tf.matmul(x, self.network_weights['V_s']), self.network_weights['c_s'])
        return m, s



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

        

class Gaussian():
    def __init__(self, mu, precision):
        self.mu = mu
        self.precision = precision #(batch_size, lat_dim, lat_dim)
        self.L = tf.linalg.cholesky(tf.linalg.inv(precision))
        self.dim = tf.cast(tf.shape(self.mu)[1], tf.float32)

    def log_probability(self, x):
        return -0.5 * (self.dim * np.log(2.0*np.pi)
                       + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis = 1)
                       +tf.reduce_sum(tf.reduce_sum(tf.matmul(tf.matmul(tf.expand_dims((x - self.mu), 1), self.precision),
                                  (tf.expand_dims((x - self.mu), -1))), axis = 2), axis =1))
                         
    def sample(self):
        eps = tf.random_normal(tf.shape(self.mu), 0, 1, dtype = tf.float32)
        return self.mu + tf.squeeze(tf.matmul(self.L, tf.expand_dims(eps, -1)), -1)
        


class Transformation():
    def __init__(self, mu):
        self.mu = mu
        self.min = tf.reduce_min(mu, 0, keepdims = True)
        self.max = tf.reduce_max(mu, 0, keepdims = True)
   
          
    def interpolate(self):
        return self.mu + tf.random_uniform(tf.shape(self.mu),-0.05,0.05)

    
    def normalized(self):
        zero = tf.constant([1e-10], dtype = tf.float32)
        return tf.divide(2.0 * tf.subtract(self.mu, self.min),
                                    tf.add(tf.subtract(self.max,self.min),zero)) - 1.0

    def diff(self, x):
        return tf.reduce_sum((x-self.mu)**2, axis = 1)
