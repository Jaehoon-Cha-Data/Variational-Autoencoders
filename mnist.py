# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:46:28 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

dataset class
"""
import tensorflow as tf
import numpy as np


    
class Mnist(object):
    def __init__(self, one_hot = True):
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        
        self._train_images = self.train_images
        self._train_labels = self.train_labels

        self._test_images = self.test_images
        self._test_labels = self.test_labels
        
        self.normalize_images()
        
        self.num_examples = train_images.shape[0]
        self.num_test_examples = test_images.shape[0]
        
        self.epochs_completed = 0
        self.index_in_epoch = 0
        
        if one_hot:
            self.one_hot_coding()
      
    def train_images(self):
        return self.train_images
        
    def train_labels(self):
        return self.train_labels

    def test_images(self):
        return self.test_images

    def test_labels(self):
        return self.test_labels

        
    def normalize_images(self):
        self.train_images = self.train_images/255.
        self.train_images = self.train_images.astype(np.float32)
        self.train_images = self.train_images.reshape(-1,784)
        
        self.test_images = self.test_images/255.
        self.test_images = self.test_images.astype(np.float32)
        self.test_images = self.test_images.reshape(-1,784)
        
    def one_hot_coding(self):
        I10 = np.eye(10)
        self.train_labels = I10[self.train_labels]
        self.test_labels = I10[self.test_labels]
        
    def next_train_batch(self, batch_size, shuffle=True):
        start = self.index_in_epoch
        if start == 0:
            perm0 = np.arange(self.num_examples)
            np.random.shuffle(perm0)
            self._train_images = self.train_images[perm0]
            self._train_labels = self.train_labels[perm0]
        if start + batch_size > self.num_examples:
            rand_index = np.random.choice(self.num_examples, size = (batch_size), replace = False)
            epoch_x, epoch_y = self.train_images[rand_index], self.train_labels[rand_index]
            self.index_in_epoch = 0
            return epoch_x, epoch_y
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            epoch_x, epoch_y = self._train_images[start:end], self._train_labels[start:end]
            return epoch_x, epoch_y

    def next_test_batch(self, batch_size, shuffle=True):
        start = self.index_in_epoch
        if start == 0:
            perm0 = np.arange(self.num_test_examples)
            np.random.shuffle(perm0)
            self._test_images = self.test_images[perm0]
            self._test_labels = self.test_labels[perm0]
        if start + batch_size > self.num_test_examples:
            rand_index = np.random.choice(self.num_test_examples, size = (batch_size), replace = False)
            epoch_x, epoch_y = self.test_images[rand_index], self.test_labels[rand_index]
            self.index_in_epoch = 0
            return epoch_x, epoch_y
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            epoch_x, epoch_y = self._test_images[start:end], self._test_labels[start:end]
            return epoch_x, epoch_y
        
