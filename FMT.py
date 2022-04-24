from urllib import response
from matplotlib import scale
import numpy as np
import cv2
import matplotlib.pyplot as plt

from parseData import RANGE_RESOLUTION_M, convertCartesianImageToPolar, convertPolarImageToCartesian, getCartImageFromImgPaths, getPolarImageFromImgPaths, getPolarImageFromImgPaths, getRadarImgPaths, convertPolarImgToLogPolar
from utils import normalize_angles


def getRotationUsingFMT(srcPolarImg: np.ndarray,
                        targetPolarImg: np.ndarray,
                        downsampleFactor: int = 10) -> float:
    '''
    @brief Get rotation using the Fourier-Mellin Transform
    @note We attempt to downsample in the range direction. 
          Since we are already in the polar domain, we just need to convert to a logpolar image
    apply phase correlation to get the rotation (which is a "\Delta Y" translation)

    @param[in] srcPolarImg Source image in polar (not log-polar) form
    @param[in] targetPolarImg Target image in polar (not log-polar) form
    @param[in] How much to further downsample in 

    @return angleRad Angle in radians, where `R(angleRad) @ src = target`
    @return scaling Scaling factor
    '''
    assert srcPolarImg.shape == targetPolarImg.shape, "Images need to have the same shape!"

    H, W = srcPolarImg.shape

    resizeSize = (int(W // downsampleFactor), H)
    srcPolarImgDownsampled = cv2.resize(srcPolarImg, resizeSize)
    targetPolarImgDownsampled = cv2.resize(targetPolarImg, resizeSize)

    # Convert to log polar form
    # TODO: Check if this is a problem
    srcLogPolar = convertPolarImgToLogPolar(srcPolarImgDownsampled)
    targetLogPolar = convertPolarImgToLogPolar(targetPolarImgDownsampled)

    hanningSize = (srcLogPolar.shape[1], srcLogPolar.shape[0])
    hanningWindow = cv2.createHanningWindow(hanningSize, cv2.CV_32F)
    deltas, response = cv2.phaseCorrelate(srcLogPolar, targetLogPolar, hanningWindow)

    # Angle
    scale, angle = deltas
    print(deltas, response)

    angle = -(float(angle) * 2 * np.pi) / srcLogPolar.shape[0]
    angle = normalize_angles(angle)

    # TODO: Unsure where the log_base is
    log_base = np.e
    scale = log_base ** scale
    # scale = np.exp(scale)

    return angle, scale, response


def rotateImg(image, angle_degrees):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_degrees, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


def plotCartPolar(prevImgPolar, currImgPolar, prevImgCart, currImgCart):
    ROWS = 2
    COLS = 2
    i = 0

    i += 1
    plt.subplot(ROWS, COLS, i)
    if prevImgPolar is not None:
        plt.imshow(prevImgPolar)
        plt.title("Prev Image Polar")

    i += 1
    plt.subplot(ROWS, COLS, i)

    if currImgPolar is not None:
        plt.imshow(currImgPolar)
        plt.title("Curr Image Polar")

    i += 1
    plt.subplot(ROWS, COLS, i)
    if prevImgCart is not None:
        plt.imshow(prevImgCart)
        plt.title("Prev Image Cartesian")

    i += 1
    plt.subplot(ROWS, COLS, i)
    if currImgCart is not None:
        plt.imshow(currImgCart)
        plt.title("Curr Image Cartesian")


if __name__ == "__main__":
    import os
    import sys

    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    startSeqInd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    endSeqInd = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    # Get initial Polar image
    imgPathArr = getRadarImgPaths(dataPath, timestampPath)
    sequenceSize = len(imgPathArr)

    if endSeqInd < 0:
        endSeqInd = sequenceSize - 1

    '''
    # Perfect Image test
    prevPolarImg = getPolarImageFromImgPaths(imgPathArr, startSeqInd)
    prevCartImg = convertPolarImageToCartesian(prevPolarImg, downsampleFactor=1)

    # TODO: manually rotate then get rotation
    
    for deg in np.arange(-30, 30, 0.2):
        rotCartImg = rotateImg(prevCartImg, deg)
        # rotCartImg = getCartImageFromImgPaths(imgPathArr, startSeqInd + 5)
        rotPolarImg = convertCartesianImageToPolar(rotCartImg,
                                                shapeHW=prevPolarImg.shape)

        plotCartPolar(prevCartImg, rotCartImg, prevPolarImg, rotPolarImg)
        # plt.show()

        rot, scale, response = getRotationUsingFMT(prevPolarImg, rotPolarImg)

        print(f"Pred: {np.rad2deg(rot):.2f} deg | Actual: {deg} deg")
        print(f"Scale Factor: {scale:.2f}")
    exit()
    # '''

    prevImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd)
    prevImgCart = convertPolarImageToCartesian(prevImgPolar, downsampleFactor=1)

    currImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd + 5)
    # currImgCart = convertPolarImageToCartesian(currImgPolar,
    #                                            downsampleFactor=1)

    rot, scale, response = getRotationUsingFMT(prevImgPolar, currImgPolar)
    print(f"Pred: {np.rad2deg(rot):.2f} [deg] {rot:.2f} [radians]")

    exit()

    prevImgCart = getCartImageFromImgPaths(imgPathArr, startSeqInd)
    prevImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd)

    for seqInd in range(startSeqInd + 1, endSeqInd + 1, 5):
        # Obtain image
        currImgCart = getCartImageFromImgPaths(imgPathArr, seqInd)
        currImgPolar = getPolarImageFromImgPaths(imgPathArr, seqInd)

        rot, scale, response = getRotationUsingFMT(prevImgPolar, currImgPolar)

        # print(f"===========Seq {seqInd}=========")
        print(f"Pred: {np.rad2deg(rot):.2f} [deg] {rot:.2f} [radians]")
        print(f"Scale Factor: {scale:.2f}, Response {response:.2f}")

        plotCartPolar(prevImgCart, currImgCart, prevImgPolar, currImgPolar)

        prevImgPolar = currImgPolar.copy()
        prevImgCart = currImgCart.copy()