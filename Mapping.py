import numpy as np
import scipy as sp

from trajectoryPlotting import Trajectory
import m2dp
from getPointCloud import getPointCloudPolarInd

# Thresholds
ROT_THRESHOLD = 0.2 # radians
TRANS_THRESHOLD = 2.0 # meters
TRANS_THRESHOLD_SQ = TRANS_THRESHOLD * TRANS_THRESHOLD # meters^2

# Keyframe class
class Keyframe():

    def __init__(self, pose: np.ndarray, featurePoints: np.ndarray,
                 radarPolarImg: np.ndarray) -> None:
        self.pose = pose
        self.featurePoints = featurePoints  # set of (tracked) feature points
        self.radarPolarImg = radarPolarImg  # radar polar image

        # TODO: Not sure if needed/useful
        self.pointCloud = getPointCloudPolarInd(radarPolarImg)


# Map class
class Map():

    def __init__(self, sequenceName: str, estTraj: Trajectory,
                 imgPathArr: list[str], filePaths: dict[str]) -> None:
        self.sequenceName = sequenceName

        self.imgPathArr = imgPathArr
        self.sequenceSize = len(self.imgPathArr)

        self.filePaths = filePaths

        self.estTraj = estTraj
        self.keyframes = []

    # TODO: might not want to make keyframe before adding it
    def isGoodKeyframe(self, keyframe: Keyframe):
        # Get information of prev KF's pose
        prevKF = self.keyframes[-1]
        srcPose = prevKF.pose

        # Get the information of possible KF's pose
        targetPose = keyframe.pose

        # Check rotation condition relative to last keyframe
        # TODO: Check if radians or degrees?
        deltaTh = np.abs(srcPose[2] - targetPose[2])

        if (deltaTh >= ROT_THRESHOLD):
            return True

        # Check translation condition
        deltaTrans = (srcPose[0:2] - targetPose[0:2]) ** 2
        deltaTrans = deltaTrans.sum()

        if (deltaTrans >= TRANS_THRESHOLD_SQ):
            return True

        return False

    def addKeyframe(self, keyframe: Keyframe):
        self.keyframes.append(keyframe)
        pass