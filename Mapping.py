import numpy as np
import scipy as sp

from parseData import MAX_RANGE_CLIP_DEFAULT
from trajectoryPlotting import Trajectory
import m2dp
from getPointCloud import getPointCloudPolarInd

# Thresholds
ROT_THRESHOLD = 0.2  # radians
TRANS_THRESHOLD = 2.0  # meters
TRANS_THRESHOLD_SQ = TRANS_THRESHOLD * TRANS_THRESHOLD  # meters^2


# Keyframe class
class Keyframe():

    def __init__(self, pose: np.ndarray, featurePoints: np.ndarray,
                 radarPolarImg: np.ndarray) -> None:
        '''
        @brief Keyframe class. Contains pose, feature points and point cloud information
        @param[in] pose (3 x 1) Pose information [x, y, th] in (m, m, rad) # TODO: Confirm these units
        @param[in] featurePoints (K x 2) Tracked feature points from previous keyframe
        @param[in] radarPolarImg (M x N) Radar polar (range-azimuth) image 
        '''
        self.pose = pose
        self.featurePoints = featurePoints  # set of (tracked) feature points
        self.radarPolarImg = radarPolarImg  # radar polar image

        # TODO: Not sure if needed/useful
        self.pointCloud = getPointCloudPolarInd(radarPolarImg)

    # def isVisible(self, keyframe):
    #     '''
    #     @brief Return points that are visible from keyframe
    #     @deprecated
    #     '''
    #     MAX_RANGE_CLIP_DEFAULT


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
    def isGoodKeyframe(self, keyframe: Keyframe) -> bool:
        '''
        @brief Check if a keyframe is good for adding using information about relative rotation and translation
        @return If keyframe passes checks
        '''
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
        deltaTrans = (srcPose[0:2] - targetPose[0:2])**2
        deltaTrans = deltaTrans.sum()

        if (deltaTrans >= TRANS_THRESHOLD_SQ):
            return True

        return False

    def addKeyframe(self, keyframe: Keyframe) -> None:
        '''
        @brief Add a keyframe to the running pose graph
        @param[in] keyframe Keyframe to add
        '''
        self.keyframes.append(keyframe)

    def bundleAdjustment(self) -> None:
        '''
        @brief Perform bundle adjustment on the last 2 keyframes
        @return None
        '''
        pass