#!/usr/bin/env python
import os, sys, math, random, pdb
import numpy as np
from scipy import io


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


if __name__ == '__main__':
	#pdb.set_trace()
	Xtr, Ytr, Xte = load_data()
	Xtr, Ytr, Xvl, Yvl = split_rnd(Xtr, Ytr)


