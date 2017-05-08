#!/usr/bin/env python
import os, sys, math, random, pdb
import numpy as np
from scipy import io
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import scipy.ndimage as scim
import scipy


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

def save_predictions(Yte):
	#Ytr is a numpy matrix representing the test labels n*10 (a label is a vector of dimention 10 that contains 1 on the dimention that corresponds to the right digit)
	data_dir=os.path.abspath("../../output/")
	prediction_file= os.path.join(data_dir, 'prediction.csv')

	#io.savemat(prediction_file, {'y': Yte})
	Yte = devectorize_labels(Yte)
	f = open(prediction_file, 'w')
	f.write('id,digit\n')
	for i in range(np.size(Yte)):
		f.write(str(i)+","+str(Yte[i])+"\n")
	f.close()




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

def sample_random_batch(X, Y, size):
	#X is a numpy matrix representing the data n*d
	#Y is a numpy matrix representing the labels n*10
	#size (int) is the desired number of the datapoints of the batch
	#return a batch of data points and labels of the size selected
	[n, d] = np.shape(X) #n is the number of training points. d is the dimention. 
	shuffle = np.random.permutation(n) #randomly shuffle the data
	X = X[shuffle, :]
	Y = Y[shuffle, :] 
	return X[:size, :], Y[:size, :]

def sample_seqential_batch(X, Y, size, slice_number):
	#X is a numpy matrix representing the data n*d
	#Y is a numpy matrix representing the labels n*10
	#size (int) is the desired number of the datapoints of the batch
	#return a batch of data points and labels of the size selected
	[n, d] = np.shape(X) #n is the number of training points. d is the dimention. 
	start = (size*slice_number)%n
	end = min(start+size, n)
	return X[start:end, :], Y[start:end, :]

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
	
	if n > 1:
		for i in range(n):
			axarr[i].imshow(X[i, :].reshape(ydim, xdim), cmap=plt.cm.binary_r)
	else:
		axarr.imshow(X[0, :].reshape(ydim, xdim), cmap=plt.cm.binary_r)
	plt.show()

def hallucinate_data(X, Y, factor=5): #will return a dataset = of size factor* numper of data points  
	#rotate randomly every data point factor times
	#X is a numpy matrix representing the data n*d
	#Y is a numpy matrix representing the labels n*10
	#return X and Y where dim X is n_new*10 and dim Y is n_new*10. n_new=n*factor
	if factor > 1:
		hallucinated_X = np.apply_along_axis(affine_transformation, 1, X)
		hallucinated_Y = np.copy(Y)
		for i in range(factor-2):
			hallucinated_X = np.concatenate((hallucinated_X, np.apply_along_axis(affine_transformation, 1, X)), 0)
			hallucinated_Y = np.concatenate((hallucinated_Y, Y),0)
		X=np.concatenate((X, hallucinated_X),0)
		Y=np.concatenate((Y, hallucinated_Y), 0)

		#suffle data and return
		[n, d] = np.shape(X) #n is the number of training points. d is the dimention. 
		shuffle = np.random.permutation(n) #randomly shuffle the data
		return X[shuffle, :], Y[shuffle, :]
	else:
		return X, Y

def vectorize_labels(Y): 
	#Y: n vector where each value is an int between 0 and 9 representing the label
	# return a matrix n*10 n*10 (a label is a vector of dimention 10 that contains 1 on the dimention that corresponds to the right digit)
	indptr = range(len(Y)+1)
	ones = np.ones(len(Y))
	matrix = scipy.sparse.csr_matrix((ones, Y, indptr))
	return np.array(matrix.todense())

def devectorize_labels(Y): #inverste transformation of vectorize_labels(Y)
	return np.argmax(Y,1)

def plot_missclassified_images(X, Y_pred, Y_labels, n=10):
	#n is the number of images to be displayed. if n is bigger than the number of missclassified images, then all the missclassified images will be displayed
	truth=devectorize_labels(Y_labels)
	predictions=devectorize_labels(Y_pred)
	iderr=np.where(predictions!=truth)[0]
	Xerr, wrong_labels, correct_labels = X[iderr,:], predictions[iderr], truth[iderr]
	n=min(n, np.shape(iderr)[0])
	if n>0:
		print("plotting missclassified images: ")
		print("uncorrect labels : "+str(wrong_labels[:n]))
		print("correct   labels : "+str(correct_labels[:n]))
		plot_images(Xerr[:n,:])

def plot_wellclassified_images(X, Y_pred, Y_labels, n=10):
	#n is the number of images to be displayed
	truth=devectorize_labels(Y_labels)
	predictions=devectorize_labels(Y_pred)
	iderr=np.where(predictions==truth)[0]
	Xerr = X[iderr,:]
	n=min(n, np.shape(iderr)[0])
	print("plotting well classified images: ")
	plot_images(Xerr[:n,:])

def plot_lowestconfidence_images(X, Y_pred, Y_confidences, n=10):
	predictions=devectorize_labels(Y_pred)
	n=min(n, np.size(predictions))
	lc_ids=Y_confidences.argsort()[:n]
	print("plotting least confident images :")
	print("predictions with the lowest confidence are : "+str(lc_ids))
	print("their predicted labels are:                : "+str(predictions[lc_ids]))
	print("their confidences (probabilities) are      : "+str(Y_confidences[lc_ids]))
	plot_images(X[lc_ids,:])


def affine_transformation(vec): #applay a rotation and a translation
	angle_window = 45 #generate uniformly an angle between [-angle_window, angle_window] (in degrees)
	trans_windon = 3  #generate uniformly a pixel translation [-trans_windon, trans_windon]
	d = np.size(vec)
	xdim=int(np.sqrt(d)); ydim=int(np.sqrt(d))
	angle = int(np.random.rand()*(angle_window*2)+1)-angle_window #sample uniformly an angle 
	translation = (int(np.random.rand()*((trans_windon*2)+1))-trans_windon, int(np.random.rand()*((trans_windon*2)+1))-trans_windon) #sample a translation for each dimension
	im = vec.reshape(ydim, xdim)
	trans_im = scim.interpolation.shift(im, translation)
	rot_im = scim.interpolation.rotate(trans_im, angle, reshape=False)
	#trans_vec = trans_im.reshape((xdim*ydim,))
	
	rot_vec = rot_im.reshape((xdim*ydim,))
	
	return rot_vec

if __name__ == '__main__':
	Xtr, Ytr, Xte = load_data()
	Xtr, Ytr, Xvl, Yvl = split_rnd(Xtr, Ytr)

	hallucinate_data(Xtr, Ytr, 4)
	print("done")


