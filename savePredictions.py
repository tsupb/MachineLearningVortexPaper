from pandas import DataFrame
import numpy as np

def savePredictions(pFileName, pPredictions, pTypeLoads):
    vTypes = []
    vIDs = []
    vResults = []

    vTotalLength = 0

    for vType in pTypeLoads:
        vTypes = vTypes + ([vType["name"]] * vType["loads"])
        vIDs = vIDs + list(range(1, vType["loads"] + 1))

        vTotalLength = vTotalLength + vType["loads"]

    vResults = pPredictions

    if (len(vResults) != vTotalLength):
        raise Exception("The result length does not match the load length")

    df = DataFrame({'Type': vTypes, 'ID': vIDs, 'Result': vResults})
    df.to_excel(pFileName + '.xlsx', index=False)