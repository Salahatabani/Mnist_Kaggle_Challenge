#!/usr/bin/env python
import os, sys, math, random, pdb
import numpy as np
from scipy import io
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time




def load_data(): 
	#returns Xtr, Ytr, Xte
	#Xtr is a numpy matrix representing the training data n*d
	#Ytr is a numpy matrix representing the training labels n*10 (a label is a vector of dimention 10 that contains 1 on the dimention that corresponds to the right digit)
	#Xte is a numpy matrix representing the test data m*d
	data_dir=os.path.abspath("../../dataset/")
	train_file = os.path.join(data_dir, 'train.mat')
	test_file= os.path.join(data_dir, 'test.mat')
	
	tr=io.loadmat(train_file); Xtr, Ytr = tr['x'], tr['y']
	Xte = io.loadmat(test_file)['x']

	return Xtr, Ytr, Xte


def split_rnd(Xtr, Ytr, ratio=0.8):
	#splits the data (randomly) into training set and validation set.
	#ration is a number between 0 and 1. ratio is the size of training set. 1-ratio is the size of the validation set
	#returns [Xtr, Ytr, Xvl, Yvl]
	[n, d] = np.shape(Xtr) #n is the number of training points. d is the dimention. 
	shuffle = np.random.permutation(n) #randomly shuffle the data
	Xtr = Xtr[shuffle, :]
	Ytr = Ytr[shuffle, :] 

	Xvl, Yvl, Xtr, Ytr = Xtr[int(ratio*n)+1:, :],Ytr[int(ratio*n)+1:, :] , Xtr[:int(ratio*n), :], Ytr[:int(ratio*n), :]

	return Xtr, Ytr, Xvl, Yvl


def compute_accuracy(Predicted_labels, Ground_Truth):
	return np.sum(np.nonzero(Ground_Truth)[1]==np.nonzero(Predicted_labels)[1])/(np.shape(Ground_Truth)[0]*1.0)

def normalize_and_center(X):
	scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
	return scaler.transform(X) #Center the data around the mean and normalize it

def plot_images(X):
	xdim=28; ydim=28 
	n, d = X.shape
	f, axarr = plt.subplots(1, n, sharey=True)
	f.set_figwidth(10 * n)
	f.set_figheight(n)
    dskdskd
	
	if n > 1:
		for i in range(n):
			axarr[i].imshow(X[i, :].reshape(ydim, xdim).T, cmap=plt.cm.binary_r)
	else:
		axarr.imshow(X[0, :].reshape(ydim, xdim).T, cmap=plt.cm.binary_r)
	
	plt.show()
	



if __name__ == '__main__':
	#pdb.set_trace()
	Xtr, Ytr, Xte = load_data()
	Xtr, Ytr, Xvl, Yvl = split_rnd(Xtr, Ytr)


