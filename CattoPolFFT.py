import numpy as np
import time
import scipy

from sympy import false

def CattoPolFFT(pState, pKxMin, pKxRange, pKyRange, pOutPutType, pTime = false, pTestThreshold = 0.001, pSplitDirection = "x"):
    if pTime: vTStart = time.time()

    vRhoMax = np.min(pState.shape)/2

    vxmid = (pState.shape[0] + 1) / 2
    vymid = (pState.shape[1] + 1) / 2

    vxymid = [vxmid, vymid]

    vRhoSize = vRhoMax.astype(int)
    vPhiSize = pState.shape[1]
    if pTime: print(f"Prep {time.time() - vTStart}s")

    #Use linear interpolation to turn results in cartesian coordinates into polar coordinates
    if pTime: vTStart = time.time()
    vrhoPoints = np.linspace(0, vRhoMax, vRhoSize, endpoint=False)
    vphiPoints = np.linspace(0, 2 * np.pi, vPhiSize, endpoint=False)

    vRho, vPhi = np.meshgrid(vrhoPoints, vphiPoints, indexing="ij")

    vx = vxymid[0] + vRho * np.cos(vPhi)
    vy = vxymid[1] + vRho * np.sin(vPhi)

    vCoords = np.array([vx.reshape(-1), vy.reshape(-1)])
    vPolState = scipy.ndimage.map_coordinates(pState, vCoords, order=1).reshape([vRhoSize, vPhiSize])
    if pTime: print(f"Pol {time.time() - vTStart}s")

    #FFT polar results and shift them for easier processing
    if pTime: vTStart = time.time()
    vPolFFT = np.fft.fftshift(np.fft.fft2(vPolState))
    if pTime: print(f"FFT {time.time() - vTStart}s")

    #Due to point symmetry half the data is irrelevant
    vYSize = vRhoSize

    if pTime: vTStart = time.time()
    match pSplitDirection:
        case "x":
            vPolFFTL = vPolFFT[:, 0:(np.ceil((vPolFFT.shape[1]) / 2)).astype(int)]
            vPolFFTL = np.flip(np.flip(vPolFFTL, 0), 1)
            vYSize = vPolFFT.shape[0]
        case "y":
            vPolFFTL = vPolFFT[0:(np.ceil((vPolFFT.shape[0]) / 2)).astype(int), :]
            vPolFFTL = np.flip(np.flip(vPolFFTL, 0), 1)
            vPolFFTL = np.transpose(vPolFFTL)
            vYSize = vPolFFT.shape[1]

    if (pKyRange >= 0):
        vxmin = ((np.ceil(vYSize) / 2) - pKyRange - 1).astype(int)
        vxmax = ((np.ceil(vYSize) / 2) + pKyRange).astype(int)
    else:
        vxmin = 0
        vxmax = vPolFFTL.shape[0]-1

    if (pKxMin >= 0):
        vKyMin = pKxMin
    else:
        vKyMin = 0

    if (pKxRange >= 0):
        vKxRange = pKxRange
    else:
        vKxRange = vPolFFTL.shape[1]-1

    #Limit to chosen window
    vPolFFTL = vPolFFTL[vxmin:vxmax, vKyMin:pKxRange] #ky = 0,1,2,3 only contains irrelevant values

    if pTime: print(f"Cut {time.time() - vTStart}s")

    #Normalise results and stitch density/angle if necessary
    match pOutPutType:
        case "all":
            vAngle = (np.angle(vPolFFTL)/np.pi)/2+0.5

            #vAngleZero = vAngle[np.floor(vAngle.shape[0] / 2).astype(int) , 0] - 0.5
            #vAngle = (vAngle - vAngleZero)%1.0

            return np.concatenate((np.abs(vPolFFTL) / np.max(np.abs(vPolFFTL)), vAngle), 1)
            #return np.concatenate((np.abs(vPolFFTL) / np.max(np.abs(vPolFFTL)), (np.angle(vPolFFTL) / np.pi) / 2 + 0.5),1)
        case "abs" | "process-III-amplitude":
            return np.abs(vPolFFTL) / np.max(np.abs(vPolFFTL))
        case "angle" | "phase" | "process-III" | "process-III-phase":
            vAngle = (np.angle(vPolFFTL) / np.pi) / 2 + 0.5

            # vAngleZero = vAngle[np.floor(vAngle.shape[0] / 2).astype(int) , 0] - 0.5
            # vAngle = (vAngle - vAngleZero)

            return vAngle
        case "test":
            return np.abs(vPolFFTL) > pTestThreshold * np.max(np.abs(vPolFFTL))
        case "real":
            return np.real(vPolFFTL)
        case "imag":
            return np.imag(vPolFFTL)
        case "complex":
            vComp = np.concatenate(((np.real(vPolFFTL)), np.imag(vPolFFTL)), 1)
            return vComp / np.max(vComp)
        case "state" | "process-I":
            return pState
        case "pol" | "process-II":
            return vPolState
        case "FFT":
            return np.abs(vPolFFT)