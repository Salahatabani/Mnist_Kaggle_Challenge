#!/usr/bin/python
import os, sys, math, random, pdb
import numpy as np
sys.path.insert(0, os.path.abspath("../data_processing/"))
from dataio import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
import scipy
import matplotlib.pyplot as plt


def fit(Xtr, Ytr, do_bagging=False): #returns a function that takes Xte and returns Yte
	#Xtr, Ytr, Xvl, Yvl = split_rnd(Xtr, Ytr)
	hidden_layers = (100, 20)
	regularization = 1
	#---Boosting params----
	num_estimators, data_samples, max_features = 10, 1.0, 1.0
	

	print ("Classifier: \n hidden layers --> "+str(hidden_layers)+"\n regularization --> "+str(regularization)+"\n bagging --> "+str(do_bagging)+ 
		"\n bagging_classifiers --> "+str(num_estimators)+"\n bagging_data_size --> "+str(data_samples)+" \n features_sampling --> "+str(max_features))
	
	clf = MLPClassifier(solver='lbfgs', alpha=regularization, hidden_layer_sizes=hidden_layers, 
		max_iter=1000000, learning_rate='invscaling')

	if do_bagging:
		clf = BaggingClassifier(clf, max_samples=data_samples, max_features=max_features, n_estimators=num_estimators)

	
	Ytr_f=np.nonzero(Ytr)[1]
	clf.fit(Xtr, Ytr_f)
	#print ("Loss on training set is : "+str(clf.loss_))
	print ("Training accuracy is "+str(clf.score(Xtr, Ytr_f)))
	return clf




def predict(clf, X):
	Yte_f = clf.predict(X)
	indptr = range(len(Yte_f)+1)
	ones = np.ones(len(Yte_f))
	matrix = scipy.sparse.csr_matrix((ones, Yte_f, indptr))
	Yte = matrix.todense()
	#pdb.set_trace()
	return Yte



if __name__ == '__main__':
	Xtr, Ytr, Xte = load_data()
	#pdb.set_trace()

	#Xtr = normalize_and_center(Xtr) #Center the data around the mean and normalize it
	#pdb.set_trace()
	Xtr, Ytr, Xvl, Ground_Truth = split_rnd(Xtr, Ytr)
	#scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(Xtr); Xtr = scaler.transform(Xtr); Xvl = scaler.transform(Xvl)
	#Xtr = normalize_and_center(Xtr)
	#Xvl = normalize_and_center(Xvl)

	neural_net = fit(Xtr, Ytr, do_bagging=True)
	Yvl = predict(neural_net, Xvl)
	
	validation_accuracy = compute_accuracy(Yvl, Ground_Truth)
	print("Validation accracy is : "+str(validation_accuracy))
	pdb.set_trace()
	print("end")
	