# from RawROAMSystem import RawROAMSystem
import os
from typing import Tuple
from matplotlib import pyplot as plt

import numpy as np
from FMT import getRotationUsingFMT, getTranslationUsingPhaseCorrelation, rotateImg
from getTransformKLT import calculateTransformDxDth, calculateTransformSVD, getTrackedPointsKLT, visualize_transform
from outlierRejection import rejectOutliers
from parseData import RANGE_RESOLUTION_CART_M
from trajectoryPlotting import Trajectory
from utils import tic, toc


class Tracker():

    def __init__(self, sequenceName: str, imgPathArr: list[str],
                 filePaths: dict[str], paramFlags: dict[bool]) -> None:
        self.sequenceName = sequenceName

        self.imgPathArr = imgPathArr
        self.sequenceSize = len(self.imgPathArr)

        self.filePaths = filePaths
        self.paramFlags = paramFlags

        self.estTraj = None
        self.gtTraj = None

    def initTraj(self, estTraj: Trajectory, gtTraj: Trajectory = None):
        self.estTraj = estTraj
        self.gtTraj = gtTraj

    def track(
            self,
            prevImgCart: np.ndarray,
            currImgCart: np.ndarray,
            prevImgPolar: np.ndarray,
            currImgPolar: np.ndarray,
            featureCoord: np.ndarray,
            seqInd: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        @brief Track based on previous and current image

        @param[in] prevImg Previous Cartesian radar image (N x N)
        @param[in] prevImg Current Cartesian radar image (N x N)
        @param[in] prevImg Previous polar radar image (? x ?)
        @param[in] prevImg Current polar radar image (? x ?)

        @param[in] blobCoord Coordinates of feature points (K x 2) in [x, y] format

        @return good_old Coordinates of old good feature points (K' x 2) in [x, y] format
        @return good_new Coordinates of new good feature points (K' x 2) in [x, y] format
        @return angleRotRad Angle used to rotate image
        '''
        # Timing
        start = tic()

        # Using FMT, obtain the rotation estimate
        angleRotRad, scale, response = getRotationUsingFMT(
            prevImgPolar, currImgPolar)

        # Correct for rotation using rotational estimate
        useFMT = self.paramFlags.get("useFMT", False)
        if useFMT:
            # TODO: Do something about combining the rotation transforms
            # prevImgCartRot = rotateImg(prevImgCart, angleRotRad)
            prevImgCartRot = prevImgCart
        else:
            prevImgCartRot = prevImgCart

        # Obtain Point Correspondences
        good_new, good_old, bad_new, bad_old, corrStatus = \
            getTrackedPointsKLT(prevImgCartRot, currImgCart, featureCoord)

        nGoodFeatures = good_new.shape[0]
        nBadFeatures = bad_new.shape[0]
        nFeatures = nGoodFeatures + nBadFeatures

        print(
            f"{seqInd} | Num good features: {nGoodFeatures} of {nFeatures} ({(nGoodFeatures / nFeatures) * 100:.2f}%) | Time: {toc(start):.2f}s"
        )

        # Outlier rejection
        doOutlierRejection = self.paramFlags.get("rejectOutliers", True)
        if doOutlierRejection:
            good_old, good_new = rejectOutliers(good_old, good_new)

        return good_old, good_new, angleRotRad

    def getTransform(self, srcCoord: np.ndarray,
                     targetCoord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        @brief Obtain transform from coordinate correspondnces
        
        @param[in] srcCoord Coordinates of src feature points (K' x 2) in [x, y] format
        @param[in] targetCoord Coordinates of target feature points (K' x 2) in [x, y] format

        @note target = R @ src + h
        @return R rotation matrix (2 x 2)
        @return h translation matrix (2 x 1), units in meters [m]
        '''
        # Obtain transforms
        # R, h = calculateTransformDxDth(srcCoord, targetCoord)
        R, h = calculateTransformSVD(srcCoord, targetCoord)
        h *= RANGE_RESOLUTION_CART_M

        return R, h

    def plot(self,
             prevImg,
             currImg,
             good_old,
             good_new,
             seqInd,
             save=True,
             show=False):
        imgSavePath = self.filePaths["imgSave"]

        # Visualizations
        visualize_transform(prevImg, currImg, good_old, good_new, show=False)
        plt.title(f"Tracking on Image {seqInd:04d}")

        if save:
            toSaveImgPath = os.path.join(imgSavePath, f"{seqInd:04d}.jpg")
            plt.savefig(toSaveImgPath)

        if show:
            plt.pause(0.01)  # animation