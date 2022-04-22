# from RawROAMSystem import RawROAMSystem
from typing import DefaultDict, Tuple

import numpy as np
from getTransformKLT import calculateTransformDxDth, calculateTransformSVD, getTrackedPointsKLT
from outlierRejection import rejectOutliers
from parseData import RANGE_RESOLUTION_CART_M
from trajectoryPlotting import Trajectory
from utils import tic, toc


class Tracker():

    def __init__(self,
                 sequenceName: str,
                 imgPathArr: list[str],
                 filePaths: DefaultDict[str],
                 hasGroundTruth: bool = True) -> None:
        self.sequenceName = sequenceName

        self.imgPathArr = imgPathArr
        self.sequenceSize = len(self.imgPathArr)

        self.filePaths = filePaths
        self.hasGroundTruth = hasGroundTruth

        self.estTraj = None
        self.gtTraj = None

    def initTraj(self, estTraj: Trajectory, gtTraj: Trajectory = None):
        self.estTraj = estTraj
        self.gtTraj = gtTraj

    def track(self, prevImg: np.ndarray, currImg: np.ndarray,
              blobCoord: np.ndarray,
              seqInd: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        '''
        # Timing
        start = tic()

        # Obtain Point Correspondences
        good_new, good_old, bad_new, bad_old, corrStatus = \
            getTrackedPointsKLT(prevImg, currImg, blobCoord)

        nGoodFeatures = good_new.shape[0]
        nBadFeatures = bad_new.shape[0]
        nFeatures = nGoodFeatures + nBadFeatures

        print(
            f"{seqInd} | Num good features: {nGoodFeatures} of {nFeatures} ({(nGoodFeatures / nFeatures) * 100:.2f}%) | Time: {toc(start):.2f}s"
        )

        # Outlier rejection
        good_old, good_new = rejectOutliers(good_old, good_new)

        return good_old, good_new

    def getTransform(self, good_old, good_new):
        # Obtain transforms
        R, h = calculateTransformDxDth(good_old, good_new)
        # R, h = calculateTransformSVD(good_old, good_new)
        h *= RANGE_RESOLUTION_CART_M

        return R, h