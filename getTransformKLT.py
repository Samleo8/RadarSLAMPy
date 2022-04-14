import numpy as np
import cv2
import os, sys
from getFeatures import getFeatures

import matplotlib.pyplot as plt

from parseData import getCartImageFromImgPaths, getRadarImgPaths


def visualize_transform(prevImg: np.ndarray,
                        currImg: np.ndarray,
                        prevFeatureCoord: np.ndarray,
                        newFeatureCoord: np.ndarray,
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

    if newFeatureCoord is not None:
        plt.scatter(newFeatureCoord[:, 0],
                    newFeatureCoord[:, 1],
                    marker='+',
                    color='blue',
                    alpha=alpha,
                    label=f'Tracked Features{extraLabel}')

    # TODO: Remove, show feature points of old images
    if prevFeatureCoord is not None:
        plt.scatter(prevFeatureCoord[:, 0],
                    prevFeatureCoord[:, 1],
                    marker='.',
                    color='yellow',
                    alpha=alpha,
                    label=f'Image 0 Features{extraLabel}')

    plt.legend()
    plt.axis("off")
    plt.title("New Image")

    plt.tight_layout()

    if show:
        plt.show()


# https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/video/optical_flow/optical_flow.py
LK_PARAMS = dict(
    maxLevel=3,  # level of pyramid search
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03
              )  # termination criteria
)

# Thresholds for feature loss
PERCENT_FEATURE_LOSS_THRESHOLD = 0.75
N_FEATURES_BEFORE_RETRACK = -1 # TODO: Make it dynamic (find the overall loss)

# Thresholds for errors
ERR_THRESHOLD = 8

def calculateFeatureLossThreshold(nInitialFeatures):
    return PERCENT_FEATURE_LOSS_THRESHOLD * nInitialFeatures

def getTrackedPointsKLT(
    srcImg: np.ndarray, targetImg: np.ndarray, blobCoordSrc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    @brief Get tracked points using the OpenCV KLT algorithm given the
           src and target img, and points from the src img to track
    @param[in] srcIimg Source image
    @param[in] targetImg Target image
    @param[in] blobIndicesSrc Indices source features (K x 2) (potentially (K x 3)) @note [x, y] format

    @return good_new, good_old, bad_new, bad_old
    '''
    # NOTE: conversion to float32 type necessary
    featurePtSrc = np.ascontiguousarray(blobCoordSrc[:, :2]).astype(np.float32)

    # TODO: Change window size based on average of blob sizes perhaps?
    winSize = (15, 15)  # window size around features

    # Re-generate new features if below certain threshold
    nFeatures = featurePtSrc.shape[0]

    global N_FEATURES_BEFORE_RETRACK
    if nFeatures < N_FEATURES_BEFORE_RETRACK:
        newFeatureCoord, newFeatureRadii = getFeatures(srcImg)
        # print(newFeatureCoord.shape)
        print("Added", newFeatureCoord.shape[0], "new features!")

        # NOTE: Also remove duplicate features, will sort the array
        featurePtSrc = np.unique(np.vstack((featurePtSrc, newFeatureCoord)), axis=0)
        featurePtSrc = np.ascontiguousarray(featurePtSrc).astype(np.float32)

        # Recalculate threshold for feature retracking
        nFeatures = featurePtSrc.shape[0]
        N_FEATURES_BEFORE_RETRACK = calculateFeatureLossThreshold(nFeatures)

    # Perform KLT to get corresponding points
    # Stupid conversions to appropriate types
    srcImgInt = (srcImg * 255).astype(np.uint8)
    targetImgInt = (targetImg * 255).astype(np.uint8)

    nextPtsGenerated, correspondenceStatus, inverseConfidence = \
        cv2.calcOpticalFlowPyrLK(srcImgInt, targetImgInt, featurePtSrc, None, winSize=winSize, **LK_PARAMS)

    # TODO: How to use inverseConfidence?
    correspondenceStatus &= (inverseConfidence < ERR_THRESHOLD)

    # Select good points (and also bad points, for visualization)
    goodCorrespondence = (correspondenceStatus == 1).flatten()
    badCorrespondence = ~goodCorrespondence
    if nextPtsGenerated is not None:
        good_new = nextPtsGenerated[goodCorrespondence, :]
        good_old = nextPtsGenerated[goodCorrespondence, :]

        bad_new = nextPtsGenerated[badCorrespondence, :]
        bad_old = nextPtsGenerated[badCorrespondence, :]
    else:
        print("[ERROR] Completely bad features!")
        # TODO: Maybe re-run with new features?

    return good_new, good_old, bad_new, bad_old


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    # Incremental streaming
    imgPathArr = getRadarImgPaths(dataPath, timestampPath)
    nImgs = len(imgPathArr)

    # Save path
    toSavePath = os.path.join(".", "img", "track_klt_dynamic_thresholding", datasetName)
    os.makedirs(toSavePath, exist_ok=True)

    # Get initial features
    startImgInd = 0
    prevImg = getCartImageFromImgPaths(imgPathArr, startImgInd)
    blobCoord, blobRadii = getFeatures(prevImg)

    N_FEATURES_BEFORE_RETRACK = calculateFeatureLossThreshold(blobCoord.shape[0])
    print("Inital Features: ", N_FEATURES_BEFORE_RETRACK)

    for imgNo in range(startImgInd + 1, nImgs):
        try:
            currImg = getCartImageFromImgPaths(imgPathArr, imgNo)

            good_new, good_old, bad_new, bad_old = \
                getTrackedPointsKLT(prevImg, currImg, blobCoord)

            nGoodFeatures = good_new.shape[0]
            nBadFeatures = bad_new.shape[0]
            nFeatures = nGoodFeatures + nBadFeatures

            print(
                f"{imgNo} | Num good features: {nGoodFeatures} of {nFeatures} ({(nGoodFeatures / nFeatures) * 100:.2f}%)"
            )

            # Visualizations
            plt.clf()
            visualize_transform(prevImg, currImg, good_old, good_new)

            if nBadFeatures > 0:
                visualize_transform(None,
                                    None,
                                    bad_old,
                                    bad_new,
                                    alpha=0.4,
                                    extraLabel=" (Bad Correspondences)")

            toSaveImgPath = os.path.join(toSavePath, f"{imgNo:04d}.jpg")
            plt.savefig(toSaveImgPath)

            plt.suptitle(f"Tracking on Image {imgNo:04d}")
            # plt.pause(0.01) # animation

            # Setup for next iteration
            blobCoord = good_new.copy()
            prevImg = np.copy(currImg)
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()

    # Generate mp4 and save that
    print("Generating mp4 with script (requires bash and FFMPEG command)...")
    try:
        os.system(f"./img/mp4-from-folder.sh {toSavePath}")
        print(f"mp4 added to {toSavePath} folder!")
    except:
        print(
            "Failed to generate mp4 with script. Likely failed system requirements."
        )
