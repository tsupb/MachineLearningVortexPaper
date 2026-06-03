import numpy as np

def popandpermut(pxData, pyData, pPopID):
    if len(pxData) != len(pyData):
        raise Exception("xData and yData need to be the same length")

    if pPopID >= len(pxData):
        raise Exception("The PopID is too large for the data")

    vxData = pxData.copy()
    vyData = pyData.copy()
    vxTest = [vxData.pop(pPopID)]
    vyTest = [vyData.pop(pPopID)]

    vPermutation = np.random.permutation(len(vxData))

    vxTrain = [vxData[i] for i in vPermutation]
    vyTrain = [vyData[i] for i in vPermutation]

    return vxTrain, vxTest, vyTrain, vyTest