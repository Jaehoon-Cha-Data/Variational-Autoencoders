# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:22:52 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

run
"""
from models import AE, VAE
import tensorflow as tf
import numpy as np
import os
from mnist import Mnist
import argparse
from collections import OrderedDict
np.random.seed(0)
tf.set_random_seed(0)

def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Error')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'VAE')
    parser.add_argument('--datasets', type = str, default = 'MNIST')
    parser.add_argument('--epochs', type = int, default = 5)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--base_lr', type = float, default = 0.0005)
    parser.add_argument('--n_input', type = int, default = 784)
    parser.add_argument('--n_output', type = int, default = 784)
    parser.add_argument('--archi', type = dict, default = {'h1':500,
                                                           'h2':500,
                                                           'latent':2})
    
    args = parser.parse_args()
    
    config = OrderedDict([
            ('model_name', args.model_name),
            ('datasets', args.datasets),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('base_lr', args.base_lr),
            ('n_input', args.n_input),
            ('n_output', args.n_output),
            ('archi', args.archi)])
    
    return config
    
config = parse_args()

mnist = Mnist()
n_samples = mnist.num_examples 


if config['model_name'] == 'AE':
    print('Run AE')
    model = AE(config['n_input'], config['n_output'], config['archi'])   
elif config['model_name'] == 'VAE':
    print('Run VAE')
    model = VAE(config['n_input'], config['n_output'], config['archi'])


mother_folder = config['model_name']
try:
    os.mkdir(mother_folder)
except OSError:
    pass    


latent, reconstr, loss = model.Forward()

lr = config['base_lr']

folder_name = os.path.join(mother_folder, config['model_name']+'_'+config['datasets']+f'_{lr}')

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            
summ = tf.summary.merge_all()
    
writer_save_name = os.path.join('log', folder_name)
writer = tf.summary.FileWriter(writer_save_name)

with tf.Session() as sess:    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    
    model_save_name = os.path.join(folder_name, config['model_name']+f'_{lr}'+'.ckpt')
    
    try:
        os.mkdir(folder_name)
    except OSError:
        pass    
    
    iteration = 0
    iter_per_epoch = n_samples/config['batch_size'] 
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for _ in range(int(iter_per_epoch)):
            epoch_x, epoch_y = mnist.next_train_batch(config['batch_size'])
            _, c, s = sess.run([optimizer, loss, summ], feed_dict = {model.x: epoch_x})
            writer.add_summary(s, global_step=iteration)
            epoch_loss += c/iter_per_epoch
            iteration+=1
        print('Epoch', epoch, 'completedd out of', config['epochs'], 'loss:', epoch_loss)
        
        if epoch % 50 == 0:
            saver.save(sess, model_save_name, global_step=epoch)    
