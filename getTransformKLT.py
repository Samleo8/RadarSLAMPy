import numpy as np
import cv2
import os, sys
from getFeatures import getBlobsFromCart

import matplotlib.pyplot as plt

from parseData import getCartImageFromImgPaths, getRadarImgPaths


def visualize_transform(prevImg: np.ndarray,
                        currImg: np.ndarray,
                        prevFeatureInd: np.ndarray,
                        newFeatureInd: np.ndarray,
                        alpha: float = 1,
                        extraLabel: str = "",
                        show: bool = False) -> None:
    '''
    @brief Visualize transform of good and bad points in 2 images
    '''
    # Visualize
    # Display prev img with old features
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(prevImg)
    plt.scatter(prevFeatureInd[:, 1],
                prevFeatureInd[:, 0],
                marker='.',
                color='red')
    plt.title("Old Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    '''

    # Display current image with new features
    if currImg is not None:
        plt.imshow(currImg)

    if newFeatureInd is not None:
        plt.scatter(newFeatureInd[:, 1],
                    newFeatureInd[:, 0],
                    marker='+',
                    color='blue',
                    alpha=alpha,
                    label=f'Tracked Features{extraLabel}')

    # TODO: Remove, show feature points of old images
    if prevFeatureInd is not None:
        plt.scatter(prevFeatureInd[:, 1],
                    prevFeatureInd[:, 0],
                    marker='.',
                    color='yellow',
                    alpha=alpha,
                    label=f'Image 0 Features{extraLabel}')

    plt.legend()
    plt.axis("off")
    plt.title("New Image")

    if show:
        plt.show()


# https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/video/optical_flow/optical_flow.py
LK_PARAMS = dict(
    maxLevel=2,  # level of pyramid search
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03
              )  # termination criteria
)


def getTrackedPointsKLT(srcImg: np.ndarray, targetImg: np.ndarray,
                        blobIndicesSrc: np.ndarray) -> np.ndarray:
    # NOTE: conversion to float32 type necessary
    featurePtSrc = np.ascontiguousarray(blobIndicesSrc[:, :2]).astype(np.float32)

    # TODO: Change window size based on average of blob sizes perhaps?
    winSize = (15, 15)  # window size around features

    # TODO: re-generate new features if below certain threshold
    nFeatures = featurePtSrc.shape[0]

    # Perform KLT to get corresponding points
    # Stupid conversions to appropriate types
    srcImgInt = (srcImg * 255).astype(np.uint8)
    targetImgInt = (targetImg * 255).astype(np.uint8)

    nextPtsGenerated, correspondenceStatus, inverseConfidence = \
        cv2.calcOpticalFlowPyrLK(srcImgInt, targetImgInt, featurePtSrc, None, winSize=winSize, **LK_PARAMS)

    # Select good points (and also bad points, for visualization)
    goodCorrespondence = (correspondenceStatus == 1).flatten()
    nGoodFeatures = np.count_nonzero(goodCorrespondence)
    print(f"Num good features: {nGoodFeatures} of {nFeatures} ({(nGoodFeatures / nFeatures) * 100:.2f}%)")

    badCorrespondence = ~goodCorrespondence
    if nextPtsGenerated is not None:
        good_new = nextPtsGenerated[goodCorrespondence,:]
        good_old = nextPtsGenerated[goodCorrespondence,:]

        bad_new = nextPtsGenerated[badCorrespondence,:]
        bad_old = nextPtsGenerated[badCorrespondence,:]
    else:
        print("Completely bad features!")

    plt.clf()
    visualize_transform(srcImg, targetImg, good_old, good_new)

    if nGoodFeatures != nFeatures:
        visualize_transform(None, None, bad_old, bad_new, alpha=0.4, extraLabel=" (Bad Correspondences)")
        
    plt.show()

    return good_new


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    # Incremental streaming
    imgPathArr = getRadarImgPaths(dataPath, timestampPath)
    nImgs = len(imgPathArr)

    # TODO: What are the values for num, min and max
    # Get initial features
    prevImg = getCartImageFromImgPaths(imgPathArr, 0)
    blobs = getBlobsFromCart(prevImg,
                                   min_sigma=0.01,
                                   max_sigma=10,
                                   num_sigma=3,
                                   threshold=.0005, # lower threshold for more features
                                   method="doh")
    blobIndices = blobs[:, :2]

    for imgNo in range(1, nImgs):
        imgCart = getCartImageFromImgPaths(imgPathArr, imgNo)

        blobIndices = getTrackedPointsKLT(prevImg, imgCart, blobIndices)
        prevImg = np.copy(imgCart)

    cv2.destroyAllWindows()