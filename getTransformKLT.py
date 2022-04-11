import numpy as np
import cv2
import os, sys
from getFeatures import getBlobsFromCart

import matplotlib.pyplot as plt

from parseData import getCartImageFromImgPaths, getRadarImgPaths

def getTransformKLT(srcImg: np.ndarray, targetImg: np.ndarray,
                    blobIndicesSrc: np.ndarray,
                    blobIndicesTarget: np.ndarray) -> np.ndarray:
    featurePtSrc = np.ascontiguousarray(np.fliplr(blobIndicesSrc[:, :-1]))
    featurePtTarget = np.ascontiguousarray(np.fliplr(blobIndicesTarget[:, :-1]))

    # Temporary display
    plt.subplot(1, 2, 1)
    plt.imshow(srcImg)
    plt.scatter(featurePtSrc[:, 0],
                featurePtSrc[:, 1],
                marker='.',
                color='red')
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(targetImg)
    plt.scatter(featurePtTarget[:, 0], featurePtTarget[:, 1],
                marker='.',
                color='red')
    plt.axis("off")
    plt.show()

    # TODO: THIS!!
    nextPtsGenerated, correspondenceStatus, inverseConfidence = \
        cv2.calcOpticalFlowPyrLK(srcImg, targetImg, featurePtSrc, featurePtTarget)

    plt.scatter(nextPtsGenerated[:, 0],
                nextPtsGenerated[:, 1],
                marker='+',
                color='blue')

    plt.tight_layout()
    plt.show()

    return None


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    # Incremental streaming
    imgPathArr = getRadarImgPaths(dataPath, timestampPath)
    nImgs = len(imgPathArr)

    for imgNo in range(nImgs):
        imgCart = getCartImageFromImgPaths(imgPathArr, imgNo)

        # TODO: What are the values for num, min and max sigma
        blobIndices = getBlobsFromCart(imgCart,
                                       min_sigma=0.01,
                                       max_sigma=10,
                                       num_sigma=3,
                                       threshold=.0005,
                                       method="doh")

        if imgNo:
            getTransformKLT(prevImg, imgCart, prevBlobIndices, blobIndices)

        prevImg = np.copy(imgCart)
        prevBlobIndices = np.copy(blobIndices)

    cv2.destroyAllWindows()