File descriptions
------
train.mat - the training dataset; it contains a dictionary with two entries : "x" and "y". The value for entry "x" is a 5000x784 matrix representing 5000 grayscale images with size (28 pixels )x(28 pixels). The value for "y" entry is a 5000x10 matrix, containing 5000 one-hot vector representing the label of a corresponding image in "x" entry. For example, digit 3 will have one-hot vector [0,0,0,1,0,0,0,0,0,0].

test.mat - the test set; it contains a dictionary with only one entry "x". This only entry has the size of 800x784, representing 800 images.

trainX.csv - the training dataset images (same as the data contained in train.mat, just different format); it has 4000 rows and 784 columns, representing 4000 28x28 grayscale images. 

trainY.csv - the training dataset labels (same as the data contained in train.mat, just in different format); it contains 4000 rows and 10 columns, each row is a one-hot vector of the corresponding digit. For example, label for digit 3 will be represented in a row like this: "0,0,0,1,0,0,0,0,0,0".

testX.csv - the test dataset images (same as the data contained in test.mat, just in different format); it contains 800 rows and 784 columns, representing 800 28x28 grayscale images.

sample_solution.csv - a sample submission file in the correct format
