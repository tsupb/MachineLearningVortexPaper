import os
from PIL import Image
import numpy
import time
from matplotlib import pyplot
import scipy.ndimage as ndi
import statistics

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from CattoPolFFT import CattoPolFFT

from pickle import dump
from popandpermut import popandpermut
from savePredictions import savePredictions

from saveParaAverage import saveParaAverage

#CHANGE:

#MAIN PARAMETERS
vTestName = "SAMPLE NAMES" #Sample directory name
vLoadLimiter = 1000 #If number of loaded samples per class should be limited (to test influence of sample size) (-1 for unlimited)
vUseImagePreprocessor = True #If the ImagePreprocessor should be used (normalises and centers density)
vAdvancedPreprocessor = True #Transforms into polar coordinates, fourier transforms, cuts complex phase windows
vComposition = "process-III-phase" #The preprocess (step) used for this model, choose "process-I" (normal real space states), "process-II" (state in polar coordinates), "process-III-phase" (phase information after FFT), "process-III-amplitude" (amplitude information after FFT)

vClassifierName = "AdaBoost" #Choose classifier, options are ["AdaBoost", "Nearest Neighbor", "LinearSVC", "Decision Tree", "Random Forest"]
#MAIN PARAMETERS

vPreprocessorTest = False #Plots the output of a specific preprocessor step
vTestType = "1" #Class to test preprocessor on
vTestID = 1 #ID of sample to test preprocessor on

vOneforOneTest = False #Test case for using all but one sample to train the model and using the last sample for test purposes, used to search for outliers, results are saved in a spreadsheet

#Parameters for the view window in k-Space
pKxMin = 3
vKxRange = 10
vKyRange = 15

vDiscreteness = 31 #Discreteness samples (i.e. color depth)

vSaveModel = True #If the trained model should be saved for future tests

vParaValues = range(50, 50 + 1, 1)#When using "AdaBoost", this parameter can be used to test the influence of AdaBosst estimators, results will be saved in a spreadsheet

vTestSize = 0.2 #Which part of the samples should be used for testing ((1-vTestSize) is used for training)
vRepetitions = 200 #How many times the test/training split is performed and tested, improves result stability

#DO NOT CHANGE:
vSaveName = f"{vTestName}-{vClassifierName}-{vComposition}-KV_{pKxMin}_{vKxRange}_{vKyRange}-DIS_{vDiscreteness}-{vLoadLimiter}"

if vPreprocessorTest:
    vLoadLimiter = vTestID + 1
    vUseImagePreprocessor = True

vTestDirectory = "./samples/" + vTestName
vDirectories = os.walk(vTestDirectory)

vParaTestName = f"Ada Itt {vTestName}-{vRepetitions}-{vComposition}"

vSuccesses = []

vTrainTime = 0
vPredictTime = 0

vTypeLoads = []
vPredictionsResults = []

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

    #Coordinate Transform + FFT (Process-II + Process-III)
    if vAdvancedPreprocessor:
        vImage = CattoPolFFT(vImage, pKxMin, vKxRange, vKyRange, vComposition, vPreprocessorTest)
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

    vLoads = vLoadLimiter if (0 <= vLoadLimiter < len(vTypeFileNames)) else len(vTypeFileNames)
    vTypeLoads.append({"name" : vType, "loads" : vLoads})

    for vID, vFileName in enumerate(vTypeFileNames):
        vImage = Image.open(vTypeDirectory + "/" + vFileName)
        if vUseImagePreprocessor:
            vImage = ImagePreprocessor(vImage)
            print(f"\r    Processing: {vID / len(vTypeFileNames) * 100:.1f}%", end="")
        else :
            vImage = numpy.asarray(vImage)

        if vPreprocessorTest:
            vTypeSamples[vType].append(vImage)
        else:
            vTypeSamples[vType].append(vImage.reshape(-1))

    if vUseImagePreprocessor:
        print(f"\r    Processing: {1 * 100:.1f}%")

print(f"With Image Preprocessor: {vUseImagePreprocessor}")

vTypeData = []
vImageData = []

for vType in vDirectoryNames:
    vTypeData.extend(numpy.full(len(vTypeSamples[vType]), vType))
    vImageData.extend(vTypeSamples[vType])

if vOneforOneTest:
    vRepetitions = 0
    for vType in vTypeLoads:
        vRepetitions = vRepetitions + vType["loads"]

    vTestSize = 1/vRepetitions

vTupleReps = 1
vTupleResults = []

vParaChange = []
vParaSuccess = []
vParaSTDdev = []
vParaMinSuc = []
vParaMaxSuc = []
vParaTTime = []
vParaPTime = []

if vPreprocessorTest:
        pyplot.imshow(vTypeSamples[vTestType][vTestID], cmap='viridis')
        pyplot.colorbar()
        pyplot.show()
else:

    for paraRep in vParaValues:
        vSuccesses = []

        vAdaMaxItters = paraRep

        vTrainTime = 0
        vPredictTime = 0

        vClassifiers = {
            "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42), random_state=42, n_estimators=vAdaMaxItters, learning_rate=1.0),
            "Nearest Neighbor": KNeighborsClassifier(3, algorithm="auto"),
            "LinearSVC": SVC(kernel="linear", C=0.025, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)
        }
        # print(f"Training size: {round((1-vTestSize) * len(vTypeData))}")

        oClassifier = vClassifiers[vClassifierName]

        vPreprocessor = preprocessing.LabelEncoder()

        vOverviewAverage = numpy.zeros([len(vDirectoryNames), len(vDirectoryNames)])

        for rep in range(vRepetitions):
            #split training and test data

            vXTrain = vXTest = vYTrain = vYTest = numpy.asarray([])

            if vTestSize == 0.0:
                vXTrain = vImageData
                vYTrain = vTypeData
            else :
                if vOneforOneTest:
                    vXTrain, vXTest, vYTrain, vYTest = popandpermut(vImageData, vTypeData, rep)
                else:
                    vXTrain, vXTest, vYTrain, vYTest = train_test_split(
                        vImageData, vTypeData, test_size=vTestSize, shuffle=True
                    )

            #preprocessing (classifiers require numerical IDs instead of string names)
            vYTrainEncoded = vPreprocessor.fit_transform(vYTrain)

            #Train
            vTrainStart = time.time()
            oClassifier.fit(vXTrain, vYTrainEncoded)
            vTrainEnd = time.time()

            vTrainTime = vTrainTime + (vTrainEnd-vTrainStart)

            if vTestSize > 0.0:
                #Test
                vPredictStart = time.time()
                vPredictions = oClassifier.predict(vXTest)
                vPredictEnd = time.time()

                vPredictTime = vPredictTime + (vPredictEnd - vPredictStart)

                #reapply preprocessor
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

                vSuccess = numpy.trace(vOverviewMatrix) / numpy.sum(vOverviewMatrix)

                if vOneforOneTest:
                    vPredictionsResults.append(vSuccess == 1.0) #vSuccess can only be 1.0 or 0.0 here as only one sample is tested

                vSuccesses.append(vSuccess)

                print(f"{rep+1}/{vRepetitions}: {vOverview} ({vSuccess:.3f})")

                vOverviewAverage = vOverviewAverage + vOverviewMatrix/len(vDecodedPredictions)

        vOverviewAverage = vOverviewAverage / vRepetitions

        if vOneforOneTest:
            savePredictions(vTestName, vPredictionsResults, vTypeLoads)

        if vSaveModel:
            vFile = open(f"modelSaves/{vSaveName}.pkl", "wb")
            dump(oClassifier, vFile, protocol=5)
            print(f"Saved as modelSaves/{vSaveName}.pkl")
            vFile.close()

        #Plot
        pyplot.matshow(vOverviewAverage)
        pyplot.xlabel("predicted type")
        pyplot.ylabel("real type")
        pyplot.xticks(ticks = range(len(vDirectoryNames)), labels = vDirectoryNames)
        pyplot.yticks(ticks = range(len(vDirectoryNames)), labels = vDirectoryNames)
        pyplot.gca().invert_yaxis()
        pyplot.gca().xaxis.set_ticks_position('bottom')

        vColorThreshold = numpy.max(vOverviewAverage)/2
        for i in range(vOverviewAverage.shape[0]):
            for j in range(vOverviewAverage.shape[1]):
                vColor = "white" if vOverviewAverage[i, j] < vColorThreshold else "black"
                pyplot.gca().text(j, i, f"{vOverviewAverage[i, j]:.3f}", ha="center", va="center", color=vColor)

        print(f"Average prediction success: {numpy.mean(vSuccesses)}")
        if vRepetitions > 1:
            print(f"Lowest prediction success: {numpy.min(vSuccesses)}")
            print(f"Highest prediction success: {numpy.max(vSuccesses)}")
            print(f"Standard deviation prediction success: {statistics.pstdev(vSuccesses)}")

        print(f"Training size: {round((1-vTestSize) * len(vTypeData))}")
        print(f"Test size: {round(vTestSize * len(vTypeData))}")

        print (f"Average training time: {vTrainTime/vRepetitions}s")
        print (f"Average prediction time: {vPredictTime/vRepetitions}s")

        if len(vParaValues) > 1:
            vParaChange.append(paraRep)
            vParaSuccess.append(numpy.mean(vSuccesses))
            vParaSTDdev.append(statistics.pstdev(vSuccesses))
            vParaMinSuc.append(numpy.min(vSuccesses))
            vParaMaxSuc.append(numpy.max(vSuccesses))
            vParaTTime.append(vTrainTime/vRepetitions)
            vParaPTime.append(vPredictTime/vRepetitions)

    if len(vParaValues) > 1:
        saveParaAverage(vParaTestName, vParaChange, vParaSuccess, vParaSTDdev, vParaMinSuc, vParaMaxSuc, vParaTTime, vParaPTime)

    pyplot.show()