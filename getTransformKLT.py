import shutil
import numpy as np
import cv2
import os, sys
from getFeatures import appendNewFeatures
from genFakeData import *

import matplotlib.pyplot as plt
from outlierRejection import rejectOutliers

from parseData import getCartImageFromImgPaths, getRadarImgPaths, RANGE_RESOLUTION_CART_M
from utils import tic, toc

from trajectoryPlotting import Trajectory, getGroundTruthTrajectory, plotGtAndEstTrajectory
from utils import *

PLOT_BAD_FEATURES = False
N_FEATURES_BEFORE_RETRACK = 80

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

    if newFeatureCoord is not None or alpha == 0:
        plt.scatter(newFeatureCoord[:, 0],
                    newFeatureCoord[:, 1],
                    marker='+',
                    color='red',
                    alpha=alpha,
                    label=f'Tracked Features{extraLabel}')

    # TODO: Remove, show feature points of old images
    if prevFeatureCoord is not None or alpha == 0:
        plt.scatter(prevFeatureCoord[:, 0],
                    prevFeatureCoord[:, 1],
                    marker='.',
                    color='yellow',
                    alpha=alpha,
                    label=f'Image 0 Features{extraLabel}')

    plt.legend()
    plt.axis("off")
    # plt.title("New Image")

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


def estimateTransformUsingDelats(srcCoords: np.ndarray,
                                 targetCoords: np.ndarray):
    '''
    @brief Estimate KLT [x, y] frame translation by taking average of deltaX and deltaYs from source
    '''
    deltas = (srcCoords - targetCoords)
    deltaAvg = np.mean(deltas, axis=0)
    deltaStdDev = np.std(deltas, axis=0)

    print("Estimated global frame x, y translation")
    print("\t[px]:", deltaAvg)
    print("\t[m]:", deltaAvg * RANGE_RESOLUTION_CART_M)

    print("Deltas StdDev:")
    print("\t[px]:", deltaStdDev)
    print("\t[m]:", deltaStdDev * RANGE_RESOLUTION_CART_M)

    deltaX, deltaY = deltaAvg

    theta = np.arctan2(deltaY, deltaX)
    dist = np.sqrt(deltaX**2 + deltaY**2)

    cth = np.cos(theta)
    sth = np.sin(theta)
    R = np.array([[cth, -sth], [sth, cth]])
    t = np.array((dist, 0))[:, np.newaxis]

    # Scale resolution
    t *= RANGE_RESOLUTION_CART_M

    # TODO: Invert transform
    # R = R.T
    # t = -R @ t

    print(
        f"Est distance: \n\t{dist:.2f} [px]\n\t{dist * RANGE_RESOLUTION_CART_M:.2f} [m]"
    )

    print(f"Est theta: \n\t{theta:.2f} [rad]\n\t{np.rad2deg(theta):.2f} [deg]")

    return R, t

def calculateTransformSVD(
        srcCoords: np.ndarray,
        targetCoords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Calculate transform given 2 point correspondences using SVD.
    Conventions:
    Rx1 + h = x0

    Reference: https://www.sciencedirect.com/science/article/pii/002192909400116L
               http://nghiaho.com/?page_id=671
    @see getCorrespondences.py
    Inputs:
    srcCoords       - (N, 2) array of source points, x0
    targetCoords    - (N, 2) array of target points, x1
    Outputs:
    (R, h)          - (2 x 2), (2 x 1) arrays: rotation and translation. Apply
                      to old points srcCoords to get new points targetCoords, i.e.
                      R * srcCoords + h = targetCoords
    '''
    x1_mean = np.mean(targetCoords, axis = 0, keepdims = True)
    norm_x1 = targetCoords - x1_mean
    x0_mean = np.mean(srcCoords, axis = 0, keepdims = True)
    norm_x0 = srcCoords - x0_mean
    C = norm_x0.T @ norm_x1 # 2 x 2

    U, _ , V_T = np.linalg.svd(C) 
    det = np.linalg.det(U @ V_T)
    remove_reflection = np.eye(C.shape[0])
    remove_reflection[-1, -1] = det
    R = U @ remove_reflection @ V_T

    h = x0_mean - (R @ x1_mean.T).T

    return R, h.T

def calculateTransformDth(
        srcCoords: np.ndarray,
        targetCoords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(srcCoords) == len(targetCoords)
    R = np.zeros((2, 2))
    h = np.zeros((2, 1))

    N = len(srcCoords)

    # Form A and b
    A = np.empty((N * 2, 1))
    b = np.empty((N * 2, 1))

    # TODO: Please make this numpy vectorized
    for i in range(N):
        src = srcCoords[i]
        target = targetCoords[i]
        # Convention: x = [lambda, hx]
        A[2 * i:2 * i + 2, :] = np.array([[-src[1]], [src[0]]])
        b[2 * i:2 * i + 2, 0] = np.array([src[0] - target[0], src[1] - target[1]])

    # Negate b because we want to go from Ax + b to min|| Ax - b ||
    x = np.linalg.inv(A.T @ A) @ A.T @ b

    # Approximate least squares solution
    theta = float(x[0])
    cth = np.cos(theta)
    sth = np.sin(theta)
    R = np.array([[cth, -sth], [sth, cth]])
    h = np.array([[0], [0]])

    return R, h

def calculateTransformDxDth(
        srcCoords: np.ndarray,
        targetCoords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(srcCoords) == len(targetCoords)
    R = np.zeros((2, 2))
    h = np.zeros((2, 1))

    N = len(srcCoords)

    # Form A and b
    A = np.empty((N * 2, 2))
    b = np.empty((N * 2, 1))

    # TODO: Please make this numpy vectorized
    for i in range(N):
        src = srcCoords[i]
        target = targetCoords[i]
        # Convention: x = [lambda, hx]
        A[2 * i:2 * i + 2, :] = np.array([[-src[1], 1], [src[0], 0]])
        b[2 * i:2 * i + 2, 0] = np.array([src[0] - target[0], src[1] - target[1]])

    # Negate b because we want to go from Ax + b to min|| Ax - b ||
    x = np.linalg.inv(A.T @ A) @ A.T @ b

    # Approximate least squares solution
    theta = float(x[0])
    cth = np.cos(theta)
    sth = np.sin(theta)
    R = np.array([[cth, -sth], [sth, cth]])
    h = np.array([[*x[1]], [0]])
    print(f"Pixel displacement: {flatten(x)}")

    return R, h

def calculateTransform(
        srcCoords: np.ndarray,
        targetCoords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    @brief Calculate transform given 2 point correspondences.

    TODO: Make this work with SVD 
    @see getCorrespondences.py
    Inputs:
    srcCoords       - (N, 2) array of source points
    targetCoords    - (N, 2) array of target points
    Outputs:
    (R, h)          - (2 x 2), (2 x 1) arrays: rotation and translation. Apply
                      to old points srcCoords to get new points targetCoords, i.e.
                      R * srcCoords + h = targetCoords
    '''
    assert len(srcCoords) == len(targetCoords)
    R = np.zeros((2, 2))
    h = np.zeros((2, 1))

    N = len(srcCoords)

    # Form A and b
    A = np.empty((N * 2, 3))
    b = np.empty((N * 2, 1))

    # TODO: Please make this numpy vectorized
    for i in range(N):
        src = srcCoords[i]
        target = targetCoords[i]
        # Convention: x = [lambda, hx, hy]
        A[2 * i:2 * i + 2, :] = np.array([[-src[1], 1, 0], [src[0], 0, 1]])
        b[2 * i:2 * i + 2, 0] = np.array([src[0] - target[0], src[1] - target[1]])

    # Negate b because we want to go from Ax + b to min|| Ax - b ||
    x = np.linalg.inv(A.T @ A) @ A.T @ b

    # Approximate least squares solution
    theta = float(x[0])
    cth = np.cos(theta)
    sth = np.sin(theta)
    R = np.array([[cth, -sth], [sth, cth]])
    print(f"Pixel displacement: {flatten(x)}")

    h = x[1:]
    '''

    # Iterative version: for precise R estimate
    num_iters = 0
    max_iters = 10
    converged = False
    R = eye(2)
    h = np.zeros((2,1))
    while num_iters < max_iters and not converged:
        A = np.empty((N * 2, 3))
        b = np.empty((N * 2, 1))
        
        src1 = (R @ srcCoords.T).T + h
        target1 = (R @ targetCoords.T).T + h

        for i in range(N):
            src = src1[i]
            target = target1[i]

            A[2 * i : 2 * i + 1, :] = np.array([[-src[1], 1, 0],
                                                [src[0],  0, 1]])
            b[2 * i : 2 * i + 1, 0] = np.array([src[0] - target[0],
                                                -src[1] - target[1]])

        # Negate b because we want to go from Ax + b to min|| Ax - b ||
        x = np.linalg.inv(A.T @ A) @ A.T @ -b

        R_adjust = np.array([[1, -x[0]],
                             [x[0], 1]])
        delta_h = x[1:]

        # Can define convergence with respect to R_adjust and delta_h here:
        # convergence = ...

        R = R_adjust @ R
        h[:, 0] += delta_h
    '''

    return R, h


def getTrackedPointsKLT(
    srcImg: np.ndarray, targetImg: np.ndarray, blobCoordSrc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    @brief Get tracked points using the OpenCV KLT algorithm given the
           src and target img, and points from the src img to track

    @param[in] srcIimg      (M x N) Source image
    @param[in] targetImg    (M x N) Target image
    @param[in] blobIndicesSrc Indices source features (K x 2) (potentially (K x 3)) @note [x, y] format

    @note  Will append k more features if it finds that there are not enough features to track
    @note  Will also prune away features. Hence might have K' points instead

    @return good_new    (K'  x 2) New points considered as good correspondences
    @return good_old    (K'  x 2) Old points considered as good correspondences

    @return bad_new     (K'' x 2) New points considered as bad correspondences 
    @return bad_old     (K'' x 2) Old points considered as bad correspondences
    
    @return correspondenceStatus    ((K + k) x 2) Status of correspondences (1 for valid, 0 for invalid/error) 
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
        good_old = featurePtSrc[goodCorrespondence, :]

        bad_new = nextPtsGenerated[badCorrespondence, :]
        bad_old = featurePtSrc[badCorrespondence, :]
    else:
        print("[ERROR] Completely bad features!")
        # TODO: Maybe re-run with new features?

    return good_new, good_old, bad_new, bad_old, correspondenceStatus


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
    trajSavePath = os.path.join(".", "img", "track_klt_thresholding",
                                datasetName + '_traj')

    saveFeaturePath = os.path.join(
        imgSavePath.strip(os.path.sep) + f"_{imgNo}.npz")
    os.makedirs(imgSavePath, exist_ok=True)
    os.makedirs(trajSavePath, exist_ok=True)

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

    # setup trajectory plotter
    gtTrajPath = os.path.join("data", datasetName, "gt", "radar_odometry.csv")
    gtTraj = getGroundTruthTrajectory(gtTrajPath)
    initTimestamp = radarImgPathToTimestamp(imgPathArr[startImgInd])
    
    initPose = gtTraj.getPoseAtTimes(initTimestamp)
    estTraj = Trajectory([initTimestamp], [initPose])

    good_old = None
    for imgNo in range(startImgInd + 1, nImgs):
        try:
            start = tic()
            currImg = getCartImageFromImgPaths(imgPathArr, imgNo)

            # Need previous good and old correspondences to perform outlier rejection
            prev_good_old = good_old

            # Obtain Point Correspondences
            good_new, good_old, bad_new, bad_old, corrStatus = \
                getTrackedPointsKLT(prevImg, currImg, blobCoord)

            nGoodFeatures = good_new.shape[0]
            nBadFeatures = bad_new.shape[0]
            nFeatures = nGoodFeatures + nBadFeatures

            print(
                f"{imgNo} | Num good features: {nGoodFeatures} of {nFeatures} ({(nGoodFeatures / nFeatures) * 100:.2f}%) | Time: {toc(start):.2f}s"
            )

            # Outlier rejection
            # if prev_good_old is not None:
            #     # Check if appended new features
            #     prev_old_size = prev_good_old.shape[0]
            #     if nFeatures > prev_old_size:
            #         corrStatus = corrStatus[:prev_old_size, :]
            #         print(corrStatus.shape, prev_good_old.shape)

            #     # Appended features should be handled here
            #     prev_good_old = prev_good_old[(corrStatus == 1).flatten(), :]
            #     print(prev_good_old.shape)

            good_old, good_new = rejectOutliers(good_old, good_new)

            # Obtain transforms
            #R, h = calculateTransformDxDth(good_old, good_new)
            R, h = calculateTransformSVD(good_old, good_new)
            # print(h)
            # h[0] += 0
            # for i in range(good_old.shape[0]):
            #     plotFakeFeatures(good_old[i:i+1,:], (R @ good_new[i:i+1,:].T + h).T, show= True)
            transformed_pts = (R @ good_new.T + h).T
            # print(f"RMSE = {np.sum(np.square(good_old - transformed_pts))}")
            #plotFakeFeatures(good_old, good_new, show = True)
            plotFakeFeatures(good_old, transformed_pts, good_new, show = True)
            h *= RANGE_RESOLUTION_CART_M

            #R, h = estimateTransformUsingDelats(good_old, good_new)

            currTimestamp = radarImgPathToTimestamp(imgPathArr[imgNo])
            gt_deltas = gtTraj.getGroundTruthDeltasAtTime(currTimestamp)
            gt_deltas[2] = np.rad2deg(gt_deltas[2])
            est_deltas = convertRandHtoDeltas(R, h)
            est_deltas[2] = np.rad2deg(est_deltas[2])
            print(f"GT Deltas: {f_arr(gt_deltas)}")
            print(f"Est Deltas: {f_arr(est_deltas)} (*dth in degrees)")

            # Visualizations
            plt.clf()
            visualize_transform(prevImg, currImg, good_old, good_new)

            if PLOT_BAD_FEATURES and nBadFeatures > 0:
                visualize_transform(None,
                                    None,
                                    bad_old,
                                    bad_new,
                                    alpha=0.4,
                                    extraLabel=" (Bad Correspondences)")

            toSaveImgPath = os.path.join(imgSavePath, f"{imgNo:04d}.jpg")
            plt.savefig(toSaveImgPath)

            plt.suptitle(f"Tracking on Image {imgNo:04d}")
            plt.pause(0.01)  # animation

            # Plot Trajectories
            timestamp = radarImgPathToTimestamp(imgPathArr[imgNo])
            est_deltas = convertRandHtoDeltas(R, h)
            dx = est_deltas[0]
            dth = est_deltas[2]
            estTraj.appendRelativeDxDth(timestamp, dx, dth)
            # estTraj.appendRelativeTransform(timestamp, R, h)
            toSaveTrajPath = os.path.join(trajSavePath, f"{imgNo:04d}.jpg")
            plotGtAndEstTrajectory(gtTraj,
                                   estTraj,
                                   f'[{imgNo}]\n'
                                   f'Est Pose: {f_arr(estTraj.poses[-1])}\n'
                                   f'GT Deltas: {f_arr(gt_deltas)}\n'
                                   f'Est Deltas: {f_arr(est_deltas)}\n',
                                   savePath=toSaveTrajPath)
            # plt.pause(0.01)

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
        # Save video sequence
        os.system(f"./img/mp4-from-folder.sh {imgSavePath}")
        print(f"mp4 saved to {imgSavePath.strip(os.path.sep)}.mp4")

        if REMOVE_OLD_RESULTS:
            shutil.rmtree(imgSavePath)
            print("Old results folder removed.")

        # Save traj sequence
        os.system(f"./img/mp4-from-folder.sh {trajSavePath}")
        print(f"mp4 saved to {trajSavePath.strip(os.path.sep)}.mp4")

        if REMOVE_OLD_RESULTS:
            shutil.rmtree(trajSavePath)
            print("Old trajectory results folder removed.")
    except:
        print(
            "Failed to generate mp4 with script. Likely failed system requirements."
        )
