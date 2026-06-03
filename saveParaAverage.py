from pandas import DataFrame
import numpy as np
def saveParaAverage(pFileName, pParaChange, pParaSuccess, pParaSTDdev, pParaMinSuc, pParaMaxSuc, pParaTTime, pParaPTime):
    df = DataFrame({'Para Value': pParaChange, 'Success': pParaSuccess, 'STD dev': pParaSTDdev, 'Min Suc': pParaMinSuc, 'Max Suc': pParaMaxSuc, 'T Time':pParaTTime, 'P Time': pParaPTime})
    df.to_excel(pFileName + '.xlsx', index=False)