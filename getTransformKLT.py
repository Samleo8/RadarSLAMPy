import shutil
import numpy as np
import cv2
import os, sys
from getFeatures import getFeatures, appendNewFeatures

import matplotlib.pyplot as plt

from parseData import getCartImageFromImgPaths, getRadarImgPaths
from utils import tic, toc


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
                    color='red',
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
    # level of pyramid search
    maxLevel=3,
    # termination criteria
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Thresholds for errors
ERR_THRESHOLD = 10  # TODO: Figure out what this is: somewhat arbitrary for now?


def calculateTransform(
        srcCoords: np.ndarray,
        targetCoords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Calculate transform given 2 point correspondences
    @see getCorrespondences.py
    '''
<<<<<<< HEAD
    assert len(srcCoords) == len(targetCoords)
    R = np.zeros((2, 2))
=======
    # TODO: THIS @KEVIN
    A = np.zeros((2, 2))
>>>>>>> fa184e63a4febabc3455aa247954b8e21960dae6
    h = np.zeros((2, 1))

    N = len(srcCoords)
    
    A = np.empty((N * 2, 3))
    b = np.empty((N * 2, 1))
    for i in range(N):
        src = srcCoords[i]
        target = targetCoords[i]

        A[2 * i : 2 * i + 1, :] = np.array([[-src[1], 1, 0],
                                            [src[0],  0, 1]])
        b[2 * i : 2 * i + 1, 0] = np.array([src[0] - target[0],
                                            -src[1] - target[1]])
    # Negate b because we want to go from Ax + b to min|| Ax - b ||
    x = np.linalg.inv(A.T @ A) @ A.T @ -b

    R = np.array([[1, -x[0]],
                  [x[0], 1]])
    h = x[1:]
    
    return R, h


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
        featurePtSrc, N_FEATURES_BEFORE_RETRACK = \
            appendNewFeatures(srcImg, featurePtSrc)

    # Perform KLT to get corresponding points
    # Stupid conversions to appropriate types
    srcImgInt = (srcImg * 255).astype(np.uint8)
    targetImgInt = (targetImg * 255).astype(np.uint8)

    nextPtsGenerated, correspondenceStatus, inverseConfidence = \
        cv2.calcOpticalFlowPyrLK(srcImgInt, targetImgInt, featurePtSrc, None, winSize=winSize, **LK_PARAMS)

    # TODO: How to use inverseConfidence?

    if nextPtsGenerated is not None:
        correspondenceStatus &= (inverseConfidence < ERR_THRESHOLD)

        # Select good points (and also bad points, for visualization)
        goodCorrespondence = (correspondenceStatus == 1).flatten()
        badCorrespondence = ~goodCorrespondence

        # Prune according to good and bad
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
    startImgInd = imgNo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    REMOVE_OLD_RESULTS = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

    assert (imgNo >= 0)

    # Data and timestamp paths
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    # Incremental streaming
    imgPathArr = getRadarImgPaths(dataPath, timestampPath)
    nImgs = len(imgPathArr)

    # Save path
    imgSavePath = os.path.join(".", "img", "track_klt_thresholding",
                               datasetName)

    saveFeaturePath = os.path.join(
        imgSavePath.strip(os.path.sep) + f"_{imgNo}.npz")
    os.makedirs(imgSavePath, exist_ok=True)

    # Get initial features
    prevImg = getCartImageFromImgPaths(imgPathArr, imgNo)

    if os.path.exists(saveFeaturePath):
        with np.load(saveFeaturePath) as data:
            blobCoord = data["blobCoord"]
            # blobRadii = data["blobRadii"]
            # N_FEATURES_BEFORE_RETRACK = data["N_FEATURES_BEFORE_RETRACK"]
            N_FEATURES_BEFORE_RETRACK = 80
    else:
        blobCoord = np.empty((0, 2))
        # blobRadii = np.empty((0, 1))

        blobCoord, N_FEATURES_BEFORE_RETRACK = appendNewFeatures(
            prevImg, blobCoord)

    print("Inital Features: ", blobCoord.shape[0])

    for imgNo in range(startImgInd + 1, nImgs):
        try:
            start = tic()
            currImg = getCartImageFromImgPaths(imgPathArr, imgNo)

            good_new, good_old, bad_new, bad_old = \
                getTrackedPointsKLT(prevImg, currImg, blobCoord)

            nGoodFeatures = good_new.shape[0]
            nBadFeatures = bad_new.shape[0]
            nFeatures = nGoodFeatures + nBadFeatures

            print(
                f"{imgNo} | Num good features: {nGoodFeatures} of {nFeatures} ({(nGoodFeatures / nFeatures) * 100:.2f}%) | Time: {toc(start):.2f}s"
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

            toSaveImgPath = os.path.join(imgSavePath, f"{imgNo:04d}.jpg")
            plt.savefig(toSaveImgPath)

            plt.suptitle(f"Tracking on Image {imgNo:04d}")
            plt.pause(0.01) # animation

            # Setup for next iteration
            blobCoord = good_new.copy()
            prevImg = np.copy(currImg)
        except KeyboardInterrupt:
            break

    # Destroy windows/clear
    cv2.destroyAllWindows()

    # Save feature npz for continuation
    saveFeaturePath = os.path.join(
        imgSavePath.strip(os.path.sep) + f"_{imgNo}.npz")
    np.savez(saveFeaturePath, blobCoord=blobCoord)  # , blobRadii=blobRadii)

    # Generate mp4 and save that
    # Also remove folder of images to save space
    print("Generating mp4 with script (requires bash and FFMPEG command)...")
    try:
        os.system(f"./img/mp4-from-folder.sh {imgSavePath}")
        print(f"mp4 saved to {imgSavePath.strip(os.path.sep)}.mp4")

        if REMOVE_OLD_RESULTS:
            shutil.rmtree(imgSavePath)
            print("Old results folder removed.")
    except:
        print(
            "Failed to generate mp4 with script. Likely failed system requirements."
        )
