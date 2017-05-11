# digit_classification

Libraries and dependencies
----------
This code runs in python 3. Follow instructions on how to Install Python 3 here. https://www.python.org/downloads/s

This code uses Anaconda. Follow instructions on how to Install anconda here: https://docs.continuum.io/anaconda/install. The python3 version of Anaconda is needed.

This code also uses Tensor Flow. Tensor Flow can be installed with Anaconda. Follow instructions here: https://www.tensorflow.org/install/install_mac#installing_with_anaconda

Folder structure 
----------
dataset: contains the training and test digits data

source: contains the source code

output: contains predictions output by the system and some digit plots

Code Execution
----------
go to source/neural_nets
run >> python convolutional_nn.py

Accuracy
----------
100% accuracy is acheived. convolutional_nn.py won the Kaggle challenge of the Cornell Machine Learning class competition.

Methodology
---------
Preprocessing: 
Geometric transformation for the digits: we applied random rotations and translations. Five new images were generated from each image, so the new training set consisted of 24,000 images

Model
Convolutional Neural Network with :
  -Layer 1: 5*5 patch, 32 feature maps, Relu transfer function. Then pooling 2*2
  -Layer 2: 5*5 patch, 64 feature maps, Relu transfer function. Then pooling 2*2
  -Layer 3: fully connected layer with 1024 output (with 0.5 dropout probability)
  -Layer 4: Softmax
