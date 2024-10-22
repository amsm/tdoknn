#%%
import random
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas.plotting

from IPython.display import display
import time
import sklearn #useless, better to import specific classes, when needed

#%%
#_.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-.
#Application: Classifying Iris Species
from sklearn.datasets import load_iris

iTimeWhenLoadingStarted = time.time()
bunchIrisDataset = load_iris()
iTimeWhenLoadingCompleted = time.time()
iSecondsTheLoadingTook = iTimeWhenLoadingCompleted-iTimeWhenLoadingStarted
strFormat = "Loading started at {}, ended at {}, took {} second(s)".format(iTimeWhenLoadingStarted, iTimeWhenLoadingCompleted, iSecondsTheLoadingTook)
print (strFormat)

#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
dictKeys = bunchIrisDataset.keys()
strFormat = "Keys in loaded \"Bunch\" corresponding to the iris dataset:\n" + str(dictKeys)
print(strFormat)

strDescriptionTruncatedTo200Chars = bunchIrisDataset['DESCR'][:200]
print(strDescriptionTruncatedTo200Chars)

#Targets are the classes to predict (all the different iris species, in this case), as numbers 0 1 2 with corresponding names
#numpy.ndarray shape is tuple 1 (3,) => ['setosa' 'versicolor' 'virginica']
ndarrayTargetNamesForClassesToPredict = bunchIrisDataset['target_names']
strFormat = "Target classes names:" + str(ndarrayTargetNamesForClassesToPredict)
print(strFormat)

#Features are the attributes that define each sample in the dataset (sepal length, sepal width, petal length, petal width in the case
listFeatureNames = bunchIrisDataset['feature_names'] #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
strFormat = "Feature names:\n" + str(listFeatureNames)
print(strFormat)

#numpy.ndarray
ndarrayIrisData = bunchIrisDataset['data'] # 150x4 matrix
typeOfIrisData = type(ndarrayIrisData) #<class 'numpy.ndarray'>
strFormat = "Type of the iris data structure: " + str(typeOfIrisData)
print(strFormat)

#shape of the data, in this case 150 sampless, each with 4 features, so (150, 4)
tuple2ShapeOfData = ndarrayIrisData.shape #tuple 2 (150, 4)
strFormat = "Shape of available data: " + str(tuple2ShapeOfData)
print(strFormat)

#random [5, 10] n first samples, just to observe
iHowMany = random.randint(5, 10)
randomSamples = ndarrayIrisData[:iHowMany]
strFormat = "First %d samples: "%(iHowMany)
strFormat += str(bunchIrisDataset['data'][:iHowMany])
print(strFormat)

#0 for setosa, 1 for versicolor, 2 for virginica
#samples / data points sorted by label: do NOT just truncate the date when splitting test/train => shuffle first!
#[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0, 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1, 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
ndarrayTargetClassForEachDataPoint = bunchIrisDataset["target"] #numpy.ndarray (150,)
#numpy.ndarray
typeOfTargetClassesNdarray = type(ndarrayTargetClassForEachDataPoint)
strFormat = "Target classes represented as type {}".format(typeOfTargetClassesNdarray)
print(strFormat)

tuple1ShapeOfPossibleClassificationTargets = ndarrayTargetClassForEachDataPoint.shape #the dot notation instead of the dict key notation, when working with Bunch objects
strFormat = "Shape of target classes array: {}".format(tuple1ShapeOfPossibleClassificationTargets)
print(strFormat)

strFormat = "Printing the target classes array: {}".format(ndarrayTargetClassForEachDataPoint)
print(strFormat)

#_.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-.
#%%
ndarray2DTheData = ndarrayIrisData #= bunchIrisDataset['data']
ndarray1DLabelsForData = ndarrayTargetClassForEachDataPoint #= bunchIrisDataset['target']
"""
train_test_split
Split arrays or matrices into random train and test subsets
Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) 
and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner.

use an int from the random_state named param, to get the same results, call after call, for better understanding
the 4-tuple components are all numpy.ndarray

75% will be for training
25% will be for testing
150*0.75 = 112.5 ~ 112
150*0.25 = 37.5
150-112 = 38
There will be 112 samples for training + 38 samples for testing
"""
#import sklearn #NOT ENOUGH to do sklearn.model_selection.train_test_split
from sklearn import model_selection
tupleTrainAndTestsSets = model_selection.train_test_split(
    ndarray2DTheData,
    ndarray1DLabelsForData,
    random_state=0, #Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
    shuffle=True
)

# unpacking of the return of train_test_split
X_train, X_test, y_train, y_test = tupleTrainAndTestsSets

"""
#X_train # coleção das amostras selecionadas para treino do classificador
print(X_train.shape) # (112, 4)
#X_test # coleção das amostras reservadas para teste ao treino, quando o treino terminar
print(X_test.shape) # (38, 4)

#y_train # coleção, PARALELA a X_train, das targets para cada amostra em X_train
print(y_train.shape) # (112,)
y_test # coleção, PARALELA a X_test, das targets para cada amostra em X_test
#print(y_test.shape) # (38,)

X_train = [(4,5,3,4), (3,5,4,3)]
y_train = [0, 2]
"""

"""
aux function to print the values and the shape of a numpy.array
"""
def printAboutNumpyNdarray (
    pA:numpy.ndarray,
    pStrTitle:str,
    piMaxNumEls:int=10
):
    strFormat =\
        "{}\nshape: {} ; values: {}".\
        format(
            pStrTitle,
            pA.shape,
            pA[:piMaxNumEls]
        )
    print(strFormat)
#def printAboutNumpyNdarray

"""
Example of X_train (10 samples long)
[
[5.8 2.6 4.  1.2]
[6.8 3.  5.5 2.1]
[4.7 3.2 1.3 0.2]
[6.9 3.1 5.1 2.3]
[5.  3.5 1.6 0.6]
[5.4 3.7 1.5 0.2]
[5.  2.  3.5 1. ]
[6.5 3.  5.5 1.8]
[6.7 3.3 5.7 2.5]
]
"""

printAboutNumpyNdarray(X_train, "X_train") #(112, 4)
printAboutNumpyNdarray(y_train, "y_train") #(112,)
printAboutNumpyNdarray(X_test, "X_test") #(38,4)4
printAboutNumpyNdarray(y_test, "y_test") #(38,)

#_.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-.
#%%
#import pandas as pd
dfIrisDatasetDataFrame = pd.DataFrame(
    data = X_train, #the 2D data as provided in the tuple returned by train_test_split
    #index = index to use for resulting frame. Will default to RangeIndex if no indexing information part of input data and no index provided.
    #columns = y_train, #the 1D numerical classes as provided in the tuple returned by train_test_split (x and y axes would display numbers)
    columns=listFeatureNames, #as provided by bunchIrisDataset['feature_names'] also accessible as bunchIrisDataset.feature_names
)
print("This is a Panda's DataFrame (caution: the print may ommit data):")
print(dfIrisDatasetDataFrame)

"""
head
This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.
For negative values of n, this function returns all rows except the last n rows, equivalent to df[:-n].
"""
head = dfIrisDatasetDataFrame.head()
display(head) #read about display vs print

"""
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.9               3.0                4.2               1.5
1                5.8               2.6                4.0               1.2
2                6.8               3.0                5.5               2.1
3                4.7               3.2                1.3               0.2
4                6.9               3.1                5.1               2.3
"""

#TODO
#add a classification column

#an array of colors can be directly used as value for the named param c for pandas' scatter_matrix, IF IT IS THE SAME SIZE of the data points array
dictColorsForEachTargetClassAsDesired={
    0:"#FF0000",
    1:"#00FF00",
    2:"#0000FF"
} #RGB color model

def createListOfColorsForEachDataPoint(
    pndarrayDataPoints:numpy.ndarray, #typically X_train
    pndarrayCorrespondingTargetClasses:numpy.ndarray, #typically y_train
    pdictCorrespondencesBetweenEachTargetClassAndDesiredColor: dict #typically a dictionary corresponding NUMERICAL target classes to colors as strings
)->list:
    listRet = []
    listTargetClassesForWhichColorWasProvided = pdictCorrespondencesBetweenEachTargetClassAndDesiredColor.keys()
    iHowManyDataPoints = pndarrayDataPoints.shape[0] #(112@0, 4)
    iHowManyTargetClasses = pndarrayCorrespondingTargetClasses.shape[0]
    bCheckSameSize:bool = iHowManyDataPoints == iHowManyTargetClasses
    if (bCheckSameSize): # garantir o paralelismo!
        for targetClass in pndarrayCorrespondingTargetClasses:
            strColorForCurrentTargetClass = pdictCorrespondencesBetweenEachTargetClassAndDesiredColor[targetClass]
            listRet.append(strColorForCurrentTargetClass)
        #for
    #if
    return listRet  # ("#FF0000", ... , "#00FF00")
#def

listColorsToUsePerSampleInData = createListOfColorsForEachDataPoint(X_train, y_train, dictColorsForEachTargetClassAsDesired)

#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html
pd.plotting.scatter_matrix(
    frame=dfIrisDatasetDataFrame,
    alpha=0.7, #amount of opacity per plot (0 - can NOT see the plots ; 1 - makes it difficult to observe overlaps)
    diagonal="hist", #kde or hist #Pick between 'kde' and 'hist' for either Kernel Density Estimation or Histogram plot in the diagonal
    figsize=(15,15), #in inches
    marker='o', #not just aesthetics - different markers might support color differently
    hist_kwds = {"bins":40}, #number of bars per histogram
    #c=y_train, #(112,) controls the color of the plots (received by kwargs) : will use matplotlib get_plot_backend("matplotlib")
    c=listColorsToUsePerSampleInData,
    s=40, #scale of each plot
)
plt.show() #without this, the plot might NOT appear

print ("END VISUALIZATION")

#_.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-.
#Building a ML model around the KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
#KNeighborsClassifier data type
#build the model => we get an object capable of classifying new samples
myModelBasedOnKNN = KNeighborsClassifier(
    metric="minkowski", #The distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric
    n_neighbors=1
)

#build the model from the training data, using the knn object
#classifiers can have many, many params, most for speed optimizations and very particupar cases
#often, the default assumed values are good to go
#the result is KNeighborsClassifier data type also (same as knn)
knnResult = myModelBasedOnKNN.fit(
    X_train, #the data selected by "train_test_split" for being the testing samples
    y_train #targets / correct classifications
)
print("knnResult:")
display (knnResult)

#gen and classify a new (random) iris
"""
Min  Max   Mean    SD   Class Correlation
sepal length:   4.3  7.9   5.84   0.83    0.7826
sepal width:    2.0  4.4   3.05   0.43   -0.4194
petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
"""
def randomIris():
    randomSepalLength = random.randint(4, 8)+random.random() #~4 to ~8 cm of sepal length
    randomSepalWidth = random.randint(1, 5)+random.random() #~1 to ~5 cm of sepal width
    randomPetalLength = random.randint(0, 8)+random.random() #~0 to ~8 cm of petal length
    randomPetalWidth = random.randint(1, 3)++random.random() #~1 to ~3 cm of petal width
    listSomeRandomIris = [
        randomSepalLength,
        randomSepalWidth,
        randomPetalLength,
        randomPetalWidth
    ]
    npaRandomIris = np.array(listSomeRandomIris)
    return npaRandomIris
#def randomIris

npaRandomIris = randomIris()
print (npaRandomIris) #e.g. [1.88201295 0.57775978 0.4055968  0.74002187]
print (npaRandomIris.shape) #(4,)
"""
[1.88201295 0.57775978 0.4055968  0.74002187]
(4,) #number of features
"""
npa1RandomSample4Features = np.array(
    [npaRandomIris] #1 sample, 4 features
)
print (npa1RandomSample4Features)
print (npa1RandomSample4Features.shape) #(1,4)

npa2RandomSample4Features = np.array(
    [
        [5, 2.9, 1, 0.2]
    ]
)

iPredictionClassTargetNumber1 = myModelBasedOnKNN.predict(npa1RandomSample4Features)
iPredictionClassTargetNumber2 = myModelBasedOnKNN.predict(npa2RandomSample4Features)

#ndarrayTargetNamesForClassesToPredict = bunchIrisDataset['target_names']
strPredictionClassName1 = ndarrayTargetNamesForClassesToPredict[iPredictionClassTargetNumber1]
strFormat = "New iris {} shaped {} predicted to be of class {} = {}".format(
    npa1RandomSample4Features,
    npa1RandomSample4Features.shape,
    iPredictionClassTargetNumber2,
    strPredictionClassName1
)
print(strFormat)

strPredictionClassName2 = ndarrayTargetNamesForClassesToPredict[iPredictionClassTargetNumber2]
strFormat = "New iris {} shaped {} predicted to be of class {} = {}".format(
    npa2RandomSample4Features,
    npa2RandomSample4Features.shape,
    iPredictionClassTargetNumber2,
    strPredictionClassName2
)
print(strFormat)

def predictRandomNIris (pN=25):
    for i in range(pN):
        npaRandomIris4F = randomIris()
        npaRandomSample1x4F = np.array(
            [npaRandomIris4F]
        )
        iPredictionClass = myModelBasedOnKNN.predict(npaRandomSample1x4F)
        strPredictionClass = ndarrayTargetNamesForClassesToPredict[iPredictionClass]

        strFormat = "New random iris {} shaped {} predicted to be of class {} = {}".format(
            npaRandomSample1x4F,
            npaRandomSample1x4F.shape,
            iPredictionClass,
            strPredictionClass
        )
        print(strFormat)
    #for
#def predictRandomNIris

predictRandomNIris()

#_.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-._.-~^~-.
#Evaluate the ML model around the KNN algorithm

#the test data was NOT used to build the model
#BUT the correct target classes / classifications are known
y_pred = myModelBasedOnKNN.predict(
    X_test #38 samples x 4 features
)
#ndarray1DLabelsForData = ndarrayTargetClassForEachDataPoint #= bunchIrisDataset['target']
print("Know classes:      ", ndarray1DLabelsForData)
print("Predicted classes: ", y_pred)
#Predictions:  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]
npaMatches = y_pred == y_test # Broadcasting
print("Matches: ", npaMatches)

"""
Matches:  [ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True
  True False]
Model accuracy / average matches:  0.9736842105263158
"""

#mean works because True==1 and False==0
fAverageMatches = np.mean(npaMatches)
print("Model accuracy / average matches: ", fAverageMatches)
#Model accuracy "manually computed" / average matches:  0.9736842105263158

fModelScoreComputedByKnn = myModelBasedOnKNN.score(X_test, y_test)
print("Model score by knn.score ", fModelScoreComputedByKnn)
#Model score by knn.score  0.9736842105263158