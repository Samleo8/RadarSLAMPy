from matplotlib import pyplot as plt
import numpy as np
import scipy as sp

from parseData import MAX_RANGE_CLIP_DEFAULT, RANGE_RESOLUTION_CART_M, convertPolarImageToCartesian
from trajectoryPlotting import Trajectory
# import m2dp
from getPointCloud import getPointCloudPolarInd
from utils import getRotationMatrix

# Thresholds
ROT_THRESHOLD = 0.2  # radians
TRANS_THRESHOLD = 2.0  # meters
TRANS_THRESHOLD_SQ = TRANS_THRESHOLD * TRANS_THRESHOLD  # meters^2

RADAR_CART_CENTER = None


# Keyframe class
class Keyframe():

    def __init__(self, globalPose: np.ndarray, featurePointsLocal: np.ndarray,
                 radarPolarImg: np.ndarray) -> None:
        '''
        @brief Keyframe class. Contains pose, feature points and point cloud information
        @param[in] globalPose           (3 x 1) Pose information [x, y, th] in global coordinates, 
                                                        units of [m, m, rad] # TODO: Confirm these units
        @param[in] featurePointsLocal   (K x 2) Tracked feature points from previous keyframe,
                                                in local coordinates (pixels)
        @param[in] radarPolarImg        (M x N) Radar polar (range-azimuth) image

        @see updateInfo() 
        '''
        self.updateInfo(globalPose, featurePointsLocal, radarPolarImg)

    def updateInfo(self, globalPose: np.ndarray,
                   featurePointsLocal: np.ndarray,
                   radarPolarImg: np.ndarray) -> None:
        '''
        @brief Update internal information: pose, feature points and point cloud information
        @param[in] globalPose           (3 x 1) Pose information [x, y, th] in global coordinates, 
                                                        units of [m, m, rad] # TODO: Confirm these units
        @param[in] featurePointsLocal   (K x 2) Tracked feature points from previous keyframe,
                                                in local coordinates (pixels)
        @param[in] radarPolarImg        (M x N) Radar polar (range-azimuth) image 
        '''
        self.pose = globalPose
        self.radarPolarImg = radarPolarImg  # radar polar image

        # Figure out and cache the center of the Cartesian image, needed for converting coordinates
        global RADAR_CART_CENTER
        if RADAR_CART_CENTER is None:
            radarCartImg = convertPolarImageToCartesian(radarPolarImg)
            RADAR_CART_CENTER = np.array(radarCartImg.shape) / 2

        self.featurePointsLocal = featurePointsLocal  # set of original feature points, in local (px coordinates)
        self.prunedFeaturePoints = self.featurePointsLocal  # set of pruned feature points (tracked until the next keyframe)

        # TODO: Not sure if needed/useful
        self.pointCloud = getPointCloudPolarInd(radarPolarImg)

    def copyFromOtherKeyframe(self, keyframe) -> None:
        self.updateInfo(keyframe.globalPose, keyframe.featurePointsLocal,
                        keyframe.radarPolarImg)

    def convertFeaturesLocalToGlobal(
            self, featurePointsLocal: np.ndarray) -> np.ndarray:
        '''
        @brief Local feature points to convert into global coordinates, given internal pose
        @param[in] featurePointsLocal   (K x 2) Tracked feature points from previous keyframe,
                                        in local coordinates (pixels)
        @return featurePointsGlobal     (K x 2) Feature points in global coordinates, meters
        '''
        x, y, th = self.pose

        # First translate local points to origin at center
        featurePointsGlobal = featurePointsLocal - RADAR_CART_CENTER

        # Then we need to convert to meters
        featurePointsGlobal *= RANGE_RESOLUTION_CART_M  # px * (m/px) = m

        # Rotate and translate to make into global coordinate system
        R = getRotationMatrix(th)
        t = np.array([x, y]).reshape(2, 1)
        featurePointsGlobal = (R @ featurePointsGlobal.T + t).T

        return featurePointsGlobal

    def pruneFeaturePoints(self, corrStatus: np.ndarray) -> None:
        '''
        @brief Prune feature points based on correspondence status
        @param[in] corrStatus 
        @note In place changing of `self.prunedFeaturePoints` function, which aims to track and prune away the feature points that are part of this keyframe
        '''
        self.prunedFeaturePoints = self.prunedFeaturePoints[corrStatus]

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

        # TODO: Instead of storing set of all keyframes, only store the last keyframe, and then a big array of map points in global coordinates
        self.mapPoints = [
        ]  # should be a large np.array of global feature points
        self.keyframes = []

    def updateInternalTraj(self, traj: Trajectory):
        self.estTraj = traj

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
        # TODO: Should actually bundle adjust on all keyframes within range
        @return None
        '''
        # Cannot do BA on only 1 KF
        if len(self.keyframes) <= 1:
            return

        # TODO: Perform bundle adjustment on last 2 keyframes
        old_kf = self.keyframes[-2]
        new_kf = self.keyframes[-1]

        # Obtain relevant information
        pose_old = old_kf.pose
        pose_new = new_kf.pose
        points_old_local = old_kf.prunedFeaturePoints
        points_new_local = new_kf.featurePointsLocal

        # NOTE: remember ot convert to global coordinates before doing anything with the keyframes
        points_old = old_kf.convertFeaturesLocalToGlobal(points_old_local)
        points_new = new_kf.convertFeaturesLocalToGlobal(points_new_local)

        # TODO: iterative optimisation and related solvers here
        # 2d bundle adjustment

        pass

    def plot(self, fig: plt.figure, show: bool = False):
        '''
        @brief Plot map points on plt figure
        @param[in] fig plt figure to plot on @todo do this
        '''
        # TODO: For now, plot all the keyframe feature points
        for kf in self.keyframes:
            points_local = kf.featurePointsLocal
            points_global = kf.convertFeaturesLocalToGlobal(points_local)
            plt.plot(points_global)

        if (show):
            plt.show()

        return