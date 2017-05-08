import numpy as np
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import svm

pathX = "C:\\Users\\Salah\Desktop\Cornell\\Spring 2017\\CS 5780 Machine Learning\\Projects\\Extra Credit Project\\trainX.csv"
pathY = "C:\\Users\\Salah\\Desktop\\Cornell\\Spring 2017\\CS 5780 Machine Learning\\Projects\\Extra Credit Project\\trainY.csv"


"Import the data and split it to training and validation"
trainX = np.genfromtxt(pathX, dtype=float, delimiter=',', names=None) 
trainY = np.genfromtxt(pathY, dtype=int, delimiter=',', names=None) 

mapp = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(10,1)
trainY= np.dot(trainY,mapp).flatten(-1)

n,d = trainX.shape
i = np.random.permutation(n)
trainX = trainX[i,:]
trainY = trainY[i]
cutoff = int(np.ceil(0.7 * n))
# indices of training samples
xTr = trainX[:cutoff,:]
yTr = trainY[:cutoff]
# indices of testing samples
xValid = trainX[cutoff:,:]
yValid = trainY[cutoff:]


" This part uses Histogram of Oriented Gradients to capture the pattern in neighboring pixels"

k1 = 128
d1 = 6
o1 = 8
xTr_HOG = np.zeros((xTr.shape[0],k1))
for i in range(xTr.shape[0]):
    xTr_HOG[i,:], _ = hog(xTr[i,:].reshape(28,28),orientations=o1,pixels_per_cell=(d1, d1),cells_per_block=(1, 1), visualise=True)

xValid_HOG = np.zeros((xValid.shape[0],k1))
for i in range(xValid.shape[0]):
    xValid_HOG[i,:], _ = hog(xValid[i,:].reshape(28,28),orientations=o1,pixels_per_cell=(d1, d1),cells_per_block=(1, 1), visualise=True)
    

"concatenating HOG to the original data improves the performance"
xTr2 = np.concatenate((xTr,xTr_HOG),axis = 1)
xValid2 = np.concatenate((xValid,xValid_HOG),axis = 1)

"The best gamma we got was 0.0215172413793, C = 4"

CList = (np.linspace(3,4,2))
gamaList =  (np.linspace(0.019,0.032,15))
ErrorMatrix=np.zeros((len(CList),len(gamaList)))

for i in range(len(CList)):
    for j in range(len(gamaList)): 
        clf = svm.SVC(gamma=gamaList[j], C=CList[i])
        clf.fit(xTr2, yTr)
        predsV = clf.predict(xValid2)
        ErrorMatrix[i,j]=np.mean(predsV!=yValid)

bestIndex = np.unravel_index(np.argmin(ErrorMatrix),ErrorMatrix.shape)
bestC = CList[bestIndex[0]]
bestgama = gamaList[bestIndex[1]]
a = sklearn.metrics.confusion_matrix(yValid, predsV)
