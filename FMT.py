from urllib import response
from matplotlib import scale
import numpy as np
import cv2

from parseData import RANGE_RESOLUTION_M, getPolarImageFromImgPaths, getPolarImageFromImgPaths, getRadarImgPaths, convertPolarImgToLogPolar

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

    @return angleRad Angle in radians 
    @return scaling Scaling factor
    '''
    assert srcPolarImg.shape == targetPolarImg.shape, "Images need to have the same shape!"

    H, W = srcPolarImg.shape

    resizeSize = (H, int(W // downsampleFactor))
    srcPolarImgDownsampled = cv2.resize(srcPolarImg, resizeSize)
    targetPolarImgDownsampled = cv2.resize(targetPolarImg, resizeSize)

    # Convert to log polar form
    srcLogPolar = convertPolarImgToLogPolar(srcPolarImgDownsampled)
    targetLogPolar = convertPolarImgToLogPolar(targetPolarImgDownsampled)

    print(srcPolarImg.shape)
    print(srcLogPolar.shape)

    deltas, response = cv2.phaseCorrelate(srcLogPolar, targetLogPolar)

    # Angle
    angle, scale = deltas

    angle = -(float(angle) * np.pi) / srcLogPolar.shape[1]
    scale = np.exp(scale)

    return angle, scale


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

    prevImg = getPolarImageFromImgPaths(imgPathArr, startSeqInd)

    if endSeqInd < 0:
        endSeqInd = sequenceSize - 1

    # TODO: manually rotate then get rotation
    deg = 10
    rotImg = rotateImg(prevImg, deg)
    rot, scale = getRotationUsingFMT(prevImg, rotImg)

    print(f"Pred: {np.rad2deg(rot)} deg | Actual: {deg} deg")

    exit()

    for seqInd in range(startSeqInd + 1, endSeqInd + 1):
        # Obtain image
        currImg = getPolarImageFromImgPaths(imgPathArr, seqInd)
        rot = getRotationUsingFMT(prevImg, currImg)

        print(seqInd, rot)

        prevImg = currImg
