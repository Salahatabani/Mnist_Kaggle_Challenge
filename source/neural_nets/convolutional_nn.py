#!/usr/bin/python
import os, sys, math, random, pdb
import numpy as np
sys.path.insert(0, os.path.abspath("../data_processing/"))
from dataio import *
import matplotlib.pyplot as plt
import tensorflow as tf


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def apply_cnn(Xtr, Ytr, Xte, Ground_Truth=None, drop_pr=0.5, n_training=1000):

	x = tf.placeholder(tf.float32, shape=[None, 784]) #placeholder for data points
	y_ = tf.placeholder(tf.float32, shape=[None, 10]) #placeholder for labels
	x_image = tf.reshape(x, [-1,28,28,1]) #transform image to 2D 28*28 pixels matrix
	#Hidden Layer 1: 
	W_conv1 = weight_variable([5, 5, 1, 32]) #weights: 5*5 filter and 32 features
	b_conv1 = bias_variable([32]) #bais
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #convolution with relu transfer
	h_pool1 = max_pool_2x2(h_conv1) #do pooling

	#Hidden Layer 2: 
	W_conv2 = weight_variable([5, 5, 32, 64]) #weights: 5*5 patch and 64 output feature for each one of the 32 features
	b_conv2 = bias_variable([64]) #bais
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #convolution with relu transfer
	h_pool2 = max_pool_2x2(h_conv2)#do pooling

	#Hidden layer 3: (fully connected layer)
	W_fc1 = weight_variable([7 * 7 * 64, 1024]) #weights: map neurons to 1024 output
	b_fc1 = bias_variable([1024]) #bais
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #fully connected layer (i.e mat mul) with relu transfer
	keep_prob = tf.placeholder(tf.float32) #dropout probability holder
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) #apply drop out on fully connected layer

	#Hidden layer 4: softmax
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(
    	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) #we use the cross entropy loss
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #we use adam method to minimize the loss
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	
	#train
	for i in range(n_training):
		X_batch, Y_batch = sample_seqential_batch(Xtr, Ytr, 100, i)
		#X_batch, Y_batch = sample_random_batch(Xtr, Ytr, 100)

		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x:X_batch, y_: Y_batch, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
			if Ground_Truth !=None:
				print("test accuracy %g"%accuracy.eval(feed_dict={
					x: Xte, y_: Ground_Truth, keep_prob: 1.0}))
			print("\n")
		train_step.run(feed_dict={x: X_batch, y_: Y_batch, keep_prob: 1-drop_pr})

	#return accuracy score on test set
	predictions = vectorize_labels(sess.run(tf.argmax(y_conv,1), feed_dict={x: Xte, keep_prob: 1.0})) 
	confidences = np.max(sess.run(tf.nn.softmax(y_conv), feed_dict={x: Xte, keep_prob: 1.0}),1) #compute the probability associated with each prediction
	acc=None
	if Ground_Truth !=None:
		acc = accuracy.eval(feed_dict={x: Xte, y_: Ground_Truth, keep_prob: 1.0})
	return predictions, confidences, acc

def predict_testlabels(Xtr, Ytr, Xte, train_iterations):
	Xtr, Ytr = hallucinate_data(Xtr, Ytr)
	print("train size (n,d)=("+str(np.shape(Xtr))+")")
	predictions, confidences, accuracy = apply_cnn(Xtr, Ytr, Xte, None, dropout_probability, train_iterations)
	save_predictions(predictions)
	plot_lowestconfidence_images(Xte, predictions, confidences, n=10)
	return predictions, confidences #predictions are the predicted labels, confidences are the probabilities of every predicted label

def predict_onvalidation(Xtr, Ytr, train_iterations):
	Xtr, Ytr, Xvl, Ground_Truth = split_rnd(Xtr, Ytr)
	Xtr, Ytr = hallucinate_data(Xtr, Ytr)
	print("train size (n,d)=("+str(np.shape(Xtr))+")")
	predictions, confidences, accuracy = apply_cnn(Xtr, Ytr, Xvl, Ground_Truth, dropout_probability, train_iterations)
	print("test accuracy "+str(accuracy))
	plot_missclassified_images(Xvl, predictions, Ground_Truth, n=10)
	plot_wellclassified_images(Xvl, predictions, Ground_Truth, n=10)
	plot_lowestconfidence_images(Xvl, predictions, confidences, n=10)
	return predictions, confidences, accuracy #predictions are the predicted labels, confidences are the probabilities of every predicted label, accuracy is the accuracy of the validation set



if __name__ == '__main__':
	#load data
	train_iterations = 10000
	dropout_probability = 0.5
	Xtr, Ytr, Xte = load_data()
	predictions, confidences = predict_testlabels(Xtr, Ytr, Xte, train_iterations)
	#predictions, confidences, accuracy = predict_onvalidation(Xtr, Ytr, train_iterations)
	log_likelihood = np.sum(np.log(confidences))/np.size(confidences)
	#pdb.set_trace()
	print("log likelihood prediction (confidence of predictions):"+str(log_likelihood))
	

	

