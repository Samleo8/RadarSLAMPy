from urllib import response
from matplotlib import scale
import numpy as np
import cv2
import matplotlib.pyplot as plt

from parseData import RANGE_RESOLUTION_M, convertCartesianImageToPolar, convertPolarImageToCartesian, getCartImageFromImgPaths, getPolarImageFromImgPaths, getPolarImageFromImgPaths, getRadarImgPaths, convertPolarImgToLogPolar


def getRotationUsingFMT(srcPolarImg: np.ndarray,
                        targetPolarImg: np.ndarray,
                        downsampleFactor: int = 2) -> float:
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
    srcLogPolar = convertPolarImgToLogPolar(srcPolarImgDownsampled)
    targetLogPolar = convertPolarImgToLogPolar(targetPolarImgDownsampled)

    deltas, response = cv2.phaseCorrelate(srcLogPolar, targetLogPolar)

    # Angle
    scale, angle = deltas

    angle = -(float(angle) * 2 * np.pi) / srcLogPolar.shape[0]
    scale = np.exp(scale)

    return angle, scale, response


def rotateImg(image, angle_degrees):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_degrees, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


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
    prevCartImg = getCartImageFromImgPaths(imgPathArr, startSeqInd)

    # TODO: manually rotate then get rotation
    deg = 90
    rotCartImg = rotateImg(prevCartImg, deg)
    rotPolarImg = convertCartesianImageToPolar(rotCartImg, shapeHW=prevPolarImg.shape)
    # prevPolarImg = convertCartesianImageToPolar(prevCartImg, size=sz)

    plt.subplot(2, 2, 1)
    plt.imshow(prevPolarImg)
    plt.subplot(2, 2, 2)
    plt.imshow(rotPolarImg)

    plt.subplot(2, 2, 3)
    plt.imshow(prevCartImg)
    plt.subplot(2, 2, 4)
    plt.imshow(rotCartImg)
    plt.show()

    rot, scale, response = getRotationUsingFMT(prevPolarImg, rotPolarImg)

    print(f"Pred: {np.rad2deg(rot):.2f} deg | Actual: {deg} deg")
    print(f"Scale Factor: {scale:.2f}")
    '''

    prevImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd)

    for seqInd in range(startSeqInd + 1, endSeqInd + 1):
        # Obtain image
        currImgCart = getCartImageFromImgPaths(imgPathArr, seqInd)

        currImgPolar = getPolarImageFromImgPaths(imgPathArr, seqInd)
        rot, scale, response = getRotationUsingFMT(prevImgPolar, currImgPolar)

        print(f"===========Seq {seqInd}=========")
        print(f"Pred: {np.rad2deg(rot):.2f} [deg] {rot:.2f} [radians]")
        print(f"Scale Factor: {scale:.2f}, Response {response:.2f}")

        prevImgPolar = currImgPolar

        plt.imshow(currImgCart)
        plt.pause(0.01)