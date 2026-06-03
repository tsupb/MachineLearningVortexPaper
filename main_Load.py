import os
from PIL import Image
import numpy
import time
from matplotlib import pyplot
import scipy.ndimage as ndi

from sklearn import preprocessing

from CattoPolFFT import CattoPolFFT

from pickle import dump, load

#CHANGE:
vTestName = "TEST SAMPLES" #Sample directory name
vLoadLimiter = 1000 #If number of loaded samples per class should be limited (to test influence of sample size) (-1 for unlimited)
vUseImagePreprocessor = True #If the ImagePreprocessor should be used (normalises and centers density)
vAdvancedPreprocessor = True #Transforms into polar coordinates, fourier transforms, cuts complex phase windows
vComposition = "process-III-phase" #The preprocess (step) used for this model, choose "process-I" (normal real space states), "process-II" (state in polar coordinates), "process-III-phase" (phase information after FFT), "process-III-amplitude" (amplitude information after FFT)

vClassifierName = "AdaBoost" #Choose classifier, options are ["AdaBoost", "Nearest Neighbor", "LinearSVC", "Decision Tree", "Random Forest"]

vLoad = "TRAIN SAMPLES" #Name of sample file used for model to be loaded

#Make sure the following parameters match those used when training the model
#Parameters for the view window in k-Space
pKxMin = 3
vKxRange = 10
vKyRange = 15

vDiscreteness = 31 #Discreteness samples (i.e. color depth)

#DO NOT CHANGE:
vLoadName = f"{vTestName}-{vClassifierName}-{vComposition}-KV_{pKxMin}_{vKxRange}_{vKyRange}-DIS_{vDiscreteness}-{vLoadLimiter}"

vTestDirectory = "./samples/" + vTestName
vDirectories = os.walk(vTestDirectory)

#vAverageSuccess = 0
#vLowestSuccess = 1
#vHighestSuccess = 0
vSuccesses = []

vTrainTime = 0
vPredictTime = 0

def sortFilesNum(pFiles):
    return sorted(pFiles, key=lambda pFileName: int(os.path.splitext(pFileName)[0]))

def ImagePreprocessor(pImage):
    vImage = numpy.asarray(pImage)
    #Normalise
    vImage = vImage/numpy.max(vImage)

    #Center
    vCenterx, vCentery = ndi.center_of_mass(vImage)
    vSizex = len(vImage) - 1
    vSizey = len(vImage[0]) - 1
    vImage = numpy.roll(vImage, vSizex / 2 - vCenterx, axis = 0)
    vImage = numpy.roll(vImage, vSizey / 2 - vCentery, axis = 1)

    #Coordinate Switch + FFT
    if vAdvancedPreprocessor:
        vImage = CattoPolFFT(vImage, pKxMin, vKxRange, vKyRange, vComposition)
        vImage = numpy.round(vImage * vDiscreteness)

    return vImage

vDirectoryNames = [os.path.split(vDirectory[0])[1] for vDirectory in vDirectories]
vDirectoryNames.pop(0)

vTypesNumber = len(vDirectoryNames)

print(f"Found {vTypesNumber} types")

vTypeSamples = {}

for vType in vDirectoryNames:
    print("- \"" + vType + "\":")

    vTypeSamples[vType] = []

    vTypeDirectory = vTestDirectory + "/" + vType

    vTypeFileNames = os.listdir(vTestDirectory + "/" + vType)
    vTypeFileNames = sortFilesNum(vTypeFileNames)

    print(f"    Found {len(vTypeFileNames)} files")

    if (vLoadLimiter >= 0 and len(vTypeFileNames) > vLoadLimiter):
        print(f"    Limiting to {vLoadLimiter} files")

        vTypeFileNames = vTypeFileNames[0:vLoadLimiter]

    for vID, vFileName in enumerate(vTypeFileNames):
        vImage = Image.open(vTypeDirectory + "/" + vFileName)
        if vUseImagePreprocessor:
            vImage = ImagePreprocessor(vImage)
            print(f"\r    Processing: {vID / len(vTypeFileNames) * 100:.1f}%", end="")
        else :
            vImage = numpy.asarray(vImage)

        vTypeSamples[vType].append(vImage.reshape(-1))

    if vUseImagePreprocessor:
        print(f"\r    Processing: {1 * 100:.1f}%")

print(f"With Image Preprocessor: {vUseImagePreprocessor}")

vTypeData = []
vImageData = []

for vType in vDirectoryNames:
    vTypeData.extend(numpy.full(len(vTypeSamples[vType]), vType))
    vImageData.extend(vTypeSamples[vType])

vTrainTime = 0
vPredictTime = 0

#Load classifier directly from file
vFile = open(f"modelSaves/{vLoadName}.pkl", "rb")
oClassifier = load(vFile)
vFile.close()

vPreprocessor = preprocessing.LabelEncoder()

vOverviewAverage = numpy.zeros([len(vDirectoryNames), len(vDirectoryNames)])

vXTest = vImageData
vYTest = vTypeData

#Test
vPredictStart = time.time()
vPredictions = oClassifier.predict(vXTest)
vPredictEnd = time.time()

vPredictTime = vPredictTime + (vPredictEnd - vPredictStart)

#reapply preprocessor
vPreprocessor.fit_transform(vYTest)
vDecodedPredictions = vPreprocessor.inverse_transform(vPredictions)

#print statistics
vResultComparison = zip(vYTest, vDecodedPredictions)

vOverview = {}
vOverviewMatrix = numpy.zeros((len(vDirectoryNames), len(vDirectoryNames)))

for vTypeReal in vDirectoryNames:
    vOverview[vTypeReal] = {}
    for vTypePredicted in vDirectoryNames:
        vOverview[vTypeReal][vTypePredicted] = 0

for vTypeReal, vTypePredicted in vResultComparison:
    vOverview[vTypeReal][vTypePredicted] += 1
    vOverviewMatrix[vDirectoryNames.index(vTypeReal), vDirectoryNames.index(vTypePredicted)] += 1

vOverviewMatrix = vOverviewMatrix / numpy.sum(vOverviewMatrix)
vSuccess = numpy.trace(vOverviewMatrix)

print(f"result: {vOverview} ({vSuccess:.3f})")
#print(vOverviewMatrix)

#Plot
pyplot.matshow(vOverviewMatrix)
pyplot.xlabel("predicted type")
pyplot.ylabel("real type")
pyplot.xticks(ticks = range(len(vDirectoryNames)), labels = vDirectoryNames)
pyplot.yticks(ticks = range(len(vDirectoryNames)), labels = vDirectoryNames)
pyplot.gca().invert_yaxis()
pyplot.gca().xaxis.set_ticks_position('bottom')
#pyplot.colorbar()
vColorThreshold = numpy.max(vOverviewMatrix)/2
for i in range(vOverviewMatrix.shape[0]):
    for j in range(vOverviewMatrix.shape[1]):
        vColor = "white" if vOverviewMatrix[i, j] < vColorThreshold else "black"
        pyplot.gca().text(j, i, f"{vOverviewMatrix[i, j]:.3f}", ha="center", va="center", color=vColor)
pyplot.show()

print(f"Prediction success: {numpy.mean(vSuccess)}")

print (f"Average training time: {vTrainTime}s")
print (f"Average prediction time: {vPredictTime}s")

#print (f"Predicted value in training data: {vDecodedPredictions[0] in vYTrain}") #DEBUG