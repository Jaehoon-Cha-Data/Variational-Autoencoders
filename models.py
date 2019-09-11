# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:12:35 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

autoencoder-generative model
"""
import tensorflow as tf
from tensorflow.contrib import slim
import networks
import numpy as np
tfd = tf.contrib.distributions
tfb = tfd.bijectors


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)



'''
Autoencoder
'''
class AE(object):
    def __init__(self, n_input, n_output, archi):
        self.n_input = n_input
        self.n_output = n_output
        self.archi = archi
        
        self.x = tf.placeholder(tf.float32, shape=(None, 784),  name = "input")
        
    def Encoder(self, inputs, name):
        with tf.name_scope(name):        
            layer_1 = slim.layers.linear(inputs, num_outputs = self.archi['h1'], activation_fn = tf.nn.softplus, scope = 'encoder1')
            layer_2 = slim.layers.linear(layer_1, num_outputs = self.archi['h2'], activation_fn = tf.nn.softplus, scope = 'encoder2')
            return layer_2
    
    def To_latent(self, inputs, name):
        with tf.name_scope(name):        
            encoder_out = slim.layers.linear(inputs, num_outputs = self.archi['latent'])
            return encoder_out
         
    def Decoder(self, inputs, name):
        with tf.name_scope(name):   
            layer_1 = slim.layers.linear(inputs, num_outputs = self.archi['h2'], activation_fn = tf.nn.softplus, scope = 'decoder1')
            layer_2 = slim.layers.linear(layer_1, num_outputs = self.archi['h1'], activation_fn = tf.nn.softplus, scope = 'decoder2')
            decoder_out = slim.layers.linear(layer_2, num_outputs = self.n_output, activation_fn = tf.nn.sigmoid, scope = 'decoder_out')
            return decoder_out

    def Forward(self):
        self.encoder_out = self.Encoder(self.x, 'encoder')
        self.z = self.To_latent(self.encoder_out, 'latent')
        self.reconstr = self.Decoder(self.z, 'decoder')
        
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.x*tf.math.log(tf.clip_by_value(self.reconstr, 1e-10, 1.0))
                                            + (1-self.x)*tf.math.log(tf.clip_by_value(1-self.reconstr,1e-10,1.0)), 1))       
        def Summaray():
            tf.summary.scalar('loss', self.loss)
        
        Summaray()
        
        return self.z, self.reconstr, self.loss

'''
Variational Autoencoder
'''
class VAE(AE):
    def __init__(self, n_input, n_output, archi):
        super().__init__(n_input, n_output, archi)
        self.prior = networks.DiagonalGaussian(mu = tf.zeros(1,1), logvar = tf.zeros(1,1))
    
    def To_latent(self, inputs, name):
        with tf.name_scope(name):
            mu = slim.layers.linear(inputs, num_outputs = self.archi['latent'], scope = 'mean')
            logvar = slim.layers.linear(inputs, num_outputs = self.archi['latent'], scope = 'logvar')
            return networks.DiagonalGaussian(mu, logvar)
    
    def Forward(self):
        self.encoder_out = self.Encoder(self.x, 'encoder')
        
        self.encode_dist = self.To_latent(self.encoder_out, 'encoder_dist')
        
        with tf.name_scope('latent'):
            self.z = self.encode_dist.sample()
        
        self.reconstr = self.Decoder(self.z, 'decoder')
        
        
        with tf.name_scope('loss'):
            self.reconst_loss = tf.reduce_mean(-tf.reduce_sum(self.x*tf.math.log(tf.clip_by_value(self.reconstr, 1e-10, 1.0))
                                                + (1-self.x)*tf.math.log(tf.clip_by_value(1-self.reconstr,1e-10,1.0)), 1))
    
            self.kl_loss = tf.reduce_mean(-self.prior.log_probability(self.z))
            
            self.lat_loss = tf.reduce_mean(self.encode_dist.log_probability(self.z))
    
            self.loss = tf.reduce_mean(self.reconst_loss + self.kl_loss + self.lat_loss)
            
        def Summaray():
            tf.summary.scalar('reconst_loss', self.reconst_loss)
            tf.summary.scalar('kl_loss', self.kl_loss)
            tf.summary.scalar('lat_loss', self.lat_loss)
            tf.summary.scalar('loss', self.loss)

        Summaray()
        
        return self.z, self.reconstr, self.loss



