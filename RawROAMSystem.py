import os
import shutil
from matplotlib import pyplot as plt
import numpy as np

from Mapping import Keyframe, Map
from getFeatures import N_FEATURES_BEFORE_RETRACK, appendNewFeatures
from parseData import convertPolarImageToCartesian, getCartImageFromImgPaths, getPolarImageFromImgPaths, getRadarImgPaths, RANGE_RESOLUTION_CART_M
from trajectoryPlotting import Trajectory, getGroundTruthTrajectory, plotGtAndEstTrajectory
from utils import convertRandHtoDeltas, f_arr, getRotationMatrix, plt_savefig_by_axis, radarImgPathToTimestamp
from Tracker import Tracker
from motionDistortion import MotionDistortionSolver
from utils import *

# Bad solution. better solution is to save in config between mapping and this file
RADAR_CART_CENTER = np.array([1012, 1012])
wantToPlot = -1
class RawROAMSystem():

    def __init__(self,
                 sequenceName: str,
                 paramFlags: bool = dict(),
                 hasGroundTruth: bool = True) -> None:
        '''
        @brief Initializer for ROAM system
        @param[in] sequenceName Name of sequence. Should be in ./data folder
        @param[in] pararmFlags Set of flags indicating turning on and off of certain algorithm features
                                - rejectOutliers: Whether to use graph-based outlier rejection
                                - useANMS: Whether to use ANMS
                                - useFMT: Whether to use FMT to correct things
                                - correctMotionDistortion: Whether to correct for motion distortion
        @param[in] hasGroundTruth Whether sequence has ground truth to be used for plotting @deprecated
        '''

        # Data and timestamp paths
        self.sequenceName = sequenceName
        self.paramFlags = paramFlags
        self.hasGroundTruth = hasGroundTruth

        dataPath = os.path.join("data", sequenceName, "radar")
        timestampPath = os.path.join("data", sequenceName, "radar.timestamps")

        assert os.path.exists(dataPath), \
            "Failed to find radar data for sequence " + self.sequenceName
        assert os.path.exists(timestampPath), \
            "Failed to find radar timestamp information for sequence " + self.sequenceName

        # Incremental streaming
        # Obtain image paths and set a starting index
        self.imgPathArr = getRadarImgPaths(dataPath, timestampPath)
        self.sequenceSize = len(self.imgPathArr)

        # Create Save paths for imaging
        imgSavePath = os.path.join(".", "img", "roam_mapping",
                                   sequenceName).strip(os.path.sep)
        trajSavePath = imgSavePath + '_traj'

        os.makedirs(imgSavePath, exist_ok=True)
        os.makedirs(trajSavePath, exist_ok=True)

        # Initialize paths as dictionary
        self.filePaths = {
            "data": dataPath,
            "timestamp": timestampPath,
            "trajSave": trajSavePath,
            "imgSave": imgSavePath
        }

        # Initialize visualization
        self.fig = plt.figure(figsize=(11, 5))

        # Initialize Trajectories
        self.gtTraj = None  # set in run() function
        self.estTraj = None  # set in run() function

        # Initialize Tracker
        self.tracker = Tracker(self.sequenceName, self.imgPathArr,
                               self.filePaths, self.paramFlags)

        # TODO: Initialize mapping
        self.map = Map(self.sequenceName, self.estTraj, self.imgPathArr,
                       self.filePaths)

        pass

    def updateTrajFromTracker(self):
        tracker = self.tracker

        self.estTraj = tracker.estTraj
        self.gtTraj = tracker.gtTraj

    def run(self, startSeqInd: int = 0, endSeqInd: int = -1) -> None:
        '''
        @brief Do a full run the ROAMing algorithm on sequence,
               starting from and ending at specified indices, 
               incrementally calling the @see Tracker.track() function 

        @param[in] startImgInd Starting index of sequence. Default 0. 
        @param[in] startImgInd Ending index of sequence. Default -1. 
                               Negative numbers to indicate end.
        '''
        # Initialize locals
        sequenceName = self.sequenceName
        sequenceSize = self.sequenceSize
        imgPathArr = self.imgPathArr

        tracker = self.tracker

        # Assertions
        assert startSeqInd >= 0, "Starting Seq Index must be >= 0"
        assert startSeqInd < sequenceSize, f"Starting Seq Index must be < sequence size ({sequenceSize})"

        if endSeqInd < 0:
            endSeqInd = sequenceSize - 1

        assert endSeqInd < sequenceSize, f"Ending Seq Index must be < sequence size ({sequenceSize})"
        assert startSeqInd <= endSeqInd, f"Should have startSeqInd <= endSeqInd"

        # Initialize Trajectories
        gtTrajPath = os.path.join("data", sequenceName, "gt",
                                  "radar_odometry.csv")
        gtTraj = getGroundTruthTrajectory(gtTrajPath)
        initTimestamp = radarImgPathToTimestamp(imgPathArr[startSeqInd])

        initPose = gtTraj.getPoseAtTimes(initTimestamp)
        estTraj = Trajectory([initTimestamp], [initPose])

        self.gtTraj = gtTraj
        self.estTraj = estTraj

        # Initialialize Motion Distortion Solver
        # Covariance matrix, point errors
        cov_p = np.diag([4, 4]) # sigma = 2 pixels
        # Covariance matrix, velocity errors
        cov_v = np.diag([1, 1, (5 * np.pi / 180) ** 2]) # 1 pixel/s, 1 pixel/s, 5 degrees/s
        MDS = MotionDistortionSolver(cov_p, cov_v)
        # Prior frame's pose
        prev_pose = convertPoseToTransform(initPose)

        # Actually run the algorithm
        # Get initial polar and Cartesian image
        prevImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd)
        prevImgCart = convertPolarImageToCartesian(prevImgPolar)

        # Get initial features from Cartesian image
        blobCoord = np.empty((0, 2))
        blobCoord, _ = appendNewFeatures(prevImgCart, blobCoord)

        # Initialize first keyframe
        metricCoord = (blobCoord - RADAR_CART_CENTER) * RANGE_RESOLUTION_CART_M
        zero_velocity = np.zeros((3,))
        old_kf = Keyframe(initPose, metricCoord, prevImgPolar, zero_velocity) # pointer to previous kf
        self.map.addKeyframe(old_kf)

        possible_kf = Keyframe(initPose, metricCoord, prevImgPolar, zero_velocity)

        for seqInd in range(startSeqInd + 1, endSeqInd + 1):
            # Obtain polar and Cart image
            currImgPolar = getPolarImageFromImgPaths(imgPathArr, seqInd)
            currImgCart = convertPolarImageToCartesian(currImgPolar)

            # Perform tracking
            # TODO: Figure out how to integrate the keyframe addition when creation of new features
            good_old, good_new, rotAngleRad, corrStatus = tracker.track(
                prevImgCart, currImgCart, prevImgPolar, currImgPolar,
                blobCoord, seqInd)
            '''
            if seqInd == wantToPlot:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.scatter(good_old[:,0], good_old[:,1])
                plt.subplot(1, 2, 2)
                plt.title("Good old")
                #applied = homogenize(centered_new) @ new_transform.T
                plt.scatter(good_new[:,0], good_new[:,1])
                plt.title(f"Good new")
                plt.show()
            '''
            # Keyframe updating
            old_kf.pruneFeaturePoints(corrStatus)
            
            print("Detected", np.rad2deg(rotAngleRad), "[deg] rotation")
            estR = getRotationMatrix(-rotAngleRad)
            
            R, h = tracker.getTransform(good_old, good_new, pixel = False)
            # R = estR
            
            # Solve for Motion Compensated Transform
            p_w = old_kf.getPrunedFeaturesGlobalPosition() # centered

            #TODO: Scatter p_w, then try the transform on the centered new points
            # Scatter that on the same plot
            centered_new = (good_new - RADAR_CART_CENTER) * RANGE_RESOLUTION_CART_M
            # Initial Transform guess
            T_wj = prev_pose @ np.block([[R,                h],
                                         [np.zeros((2,)),   1]])

            # Give Motion Distort info on two new frames
            debug = False
            if seqInd == wantToPlot:
                debug = False
            # Centered_new is in meters, p_w is in meters, T_wj is in meters, prev_pose is meters
            MDS.update_problem(prev_pose, p_w, centered_new, T_wj, debug)
            undistort_solution = MDS.optimize_library()

            # Extract new info
            pose_vector = undistort_solution[3:]
            new_transform = convertPoseToTransform(pose_vector)
            relative_transform = MDS.T_wj0_inv @ new_transform
            '''
            if seqInd == wantToPlot:
                plt.figure()
                plt.subplot(1, 3, 1)
                plt.scatter(p_w[:,0], p_w[:,1])
                plt.subplot(1, 3, 2)
                plt.title("World coordinates")
                applied = homogenize(centered_new) @ new_transform.T
                plt.scatter(centered_new[:,0], centered_new[:,1])
                plt.title(f"Post-Transform: {(np.max(centered_new[:,0]) - np.min(centered_new[:,0]))/(np.max(p_w[:,0]) - np.min(p_w[:,0]))}")
                plt.subplot(1, 3, 3)
                diff = p_w - applied[:, :2]
                plt.scatter(diff[:,0], diff[:,1])
                plt.show()
            '''
            R = relative_transform[:2, :2]
            h = relative_transform[:2, 2:]
            velocity = undistort_solution[:3]
            #velocity[:2] /= RANGE_RESOLUTION_CART_M

            # Update trajectory
            #self.updateTrajectory(R, h, seqInd)
            self.updateTrajectoryAbsolute(pose_vector, seqInd)

            latestPose = pose_vector #self.estTraj.poses[-1]
            # Good new is given in pixels, given velocity in meters, uh oh, pose in meters
            possible_kf.updateInfo(latestPose, centered_new, currImgPolar, velocity)

            # Add a keyframe if it fulfills criteria
            # 1) large enough translation from previous keyframe
            # 2) large enough rotation from previous KF
            # TODO: Not sure if this criteria is actually correct, perhaps we should be adding the previous keyframe instead
            # 3) not enough features in current keyframe (ie about to have new features coming up)
            # NOTE: Feature check and appending will only be checked in the next iteration,
            #       so we can prematuraly do it here and add a keyframe first
            nFeatures = good_new.shape[0]
            retrack = (nFeatures <= N_FEATURES_BEFORE_RETRACK)
            if retrack or \
                self.map.isGoodKeyframe(possible_kf):

                print("\nAdding keyframe...\n")

                #old_kf.copyFromOtherKeyframe(possible_kf)
                #self.map.addKeyframe(possible_kf)

                # TODO: Aliasing above? old_kf is never assigned the object possible_kf,
                # map ends up with a list of N pointers to the same keyframe
                # Proposed fix: old_kf = possible_kf # switch ptr to new object
                # Initialize new poss_kf for new ptr
                old_kf = possible_kf
                # TODO: Never replenished blobCoord. Proposed fix: Done here.
                if retrack:
                    good_new, _ = appendNewFeatures(currImgCart, good_new)
                    centered_new = (good_new - RADAR_CART_CENTER) * RANGE_RESOLUTION_CART_M
                    old_kf.updateInfo(latestPose, centered_new, currImgPolar, velocity)
                possible_kf = Keyframe(latestPose, centered_new, currImgPolar, velocity)
                # TODO: do bundle adjustment here
                #self.map.bundleAdjustment()

            # Plotting and prints and stuff
            if seqInd == endSeqInd:
                self.plot(prevImgCart, currImgCart, good_old, good_new, R, h,
                        seqInd, save = True, show = False)
            else:
                self.plot(prevImgCart, currImgCart, good_old, good_new, R, h,
                        seqInd, save = False, show = False)
            # Update incremental variables
            blobCoord = good_new.copy()
            prevImgCart = currImgCart
            prev_pose = convertPoseToTransform(latestPose)

    # TODO: Move into trajectory class?
    def updateTrajectory(self, R, h, seqInd):
        imgPathArr = self.imgPathArr

        timestamp = radarImgPathToTimestamp(imgPathArr[seqInd])
        est_deltas = convertRandHtoDeltas(R, h)
        self.estTraj.appendRelativeDeltas(timestamp, est_deltas)
        # self.estTraj.appendRelativeTransform(timestamp, R, h)

    def updateTrajectoryAbsolute(self, pose_vector, seqInd):
        imgPathArr = self.imgPathArr

        timestamp = radarImgPathToTimestamp(imgPathArr[seqInd])
        self.estTraj.appendAbsoluteTransform(timestamp, pose_vector)

    def plot(self,
             prevImg,
             currImg,
             good_old,
             good_new,
             R,
             h,
             seqInd,
             save=True,
             show=False):

        # Draw as subplots
        plt.clf()
        self.fig.clear()

        ax1 = self.fig.add_subplot(1, 2, 1)
        self.tracker.plot(prevImg,
                          currImg,
                          good_old,
                          good_new,
                          seqInd,
                          save=False,
                          show=False)

        ax2 = self.fig.add_subplot(1, 2, 2)
        # TODO: Plotting for map points
        #self.map.plot(self.fig, show=False)

        self.plotTraj(seqInd, R, h, save=False, show=False)

        trajSavePath = self.filePaths["trajSave"]
        trajSavePathInd = os.path.join(trajSavePath, f"{seqInd:04d}.jpg")
        # plt_savefig_by_axis(trajSavePathInd, self.fig, ax2)

        plt.tight_layout()
        self.fig.savefig(trajSavePathInd)

        # # Save by subplot
        if save:
            imgSavePath = self.filePaths["imgSave"]
            imgSavePathInd = os.path.join(imgSavePath, f"{seqInd:04d}.jpg")
            plt_savefig_by_axis(imgSavePathInd, self.fig, ax1)

            trajSavePath = self.filePaths["trajSave"]
            trajSavePathInd = os.path.join(trajSavePath, f"{seqInd:04d}.jpg")
            plt_savefig_by_axis(trajSavePathInd, self.fig, ax2)

        if show:
            plt.pause(0.01)

    def plotTraj(self, seqInd, R, h, save=False, show=False):
        # Init locals
        gtTraj = self.gtTraj
        estTraj = self.estTraj
        imgPathArr = self.imgPathArr
        trajSavePath = self.filePaths["trajSave"]

        # Get timestamps for plotting etc
        currTimestamp = radarImgPathToTimestamp(imgPathArr[seqInd])

        # Debugging information with current poses and deltas
        # Thetas are in DEGREES for readibiliy
        gt_deltas = gtTraj.getGroundTruthDeltasAtTime(currTimestamp)
        gt_deltas[2] = np.rad2deg(gt_deltas[2])
        est_deltas = convertRandHtoDeltas(R, h)
        est_deltas[2] = np.rad2deg(est_deltas[2])
        est_pose = estTraj.poses[-1].copy()
        est_pose[2] = np.rad2deg(est_pose[2])

        info = f'Timestamp: {currTimestamp}\n'
        info += f'EST Pose: {f_arr(est_pose, th_deg=True)}\n'
        info += f'GT Deltas: {f_arr(gt_deltas, th_deg=True)}\n'
        info += f'EST Deltas: {f_arr(est_deltas, th_deg=True)}'
        print(info)

        # Plot Trajectories
        toSaveTrajPath = os.path.join(trajSavePath, f"{seqInd:04d}.jpg") \
             if save else None

        plotGtAndEstTrajectory(gtTraj,
                               estTraj,
                               title=f'[{seqInd}]',
                               info=info,
                               savePath=toSaveTrajPath)

        if show:
            plt.pause(0.01)


if __name__ == "__main__":
    import sys

    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    startSeqInd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    endSeqInd = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    REMOVE_OLD_RESULTS = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False

    # Initialize system with parameter flags
    paramFlags = {
        "rejectOutliers": True,
        "useFMT": False,
        # Below all currently unused actually
        "useANMS": False,
        "correctMotionDistortion": False
    }

    system = RawROAMSystem(datasetName,
                           paramFlags=paramFlags,
                           hasGroundTruth=True)

    try:
        system.run(startSeqInd, endSeqInd)
    except KeyboardInterrupt:
        pass

    imgSavePath = system.filePaths["imgSave"]
    trajSavePath = system.filePaths["trajSave"]

    # Generate mp4 and save that
    # Also remove folder of images to save space
    print("Generating mp4 with script (requires bash and FFMPEG command)...")
    try:
        # Save video sequence
        os.system(f"./img/mp4-from-folder.sh {imgSavePath} {startSeqInd + 1} 20")
        print(f"mp4 saved to {imgSavePath.strip(os.path.sep)}.mp4")

        if REMOVE_OLD_RESULTS:
            shutil.rmtree(imgSavePath)
            print("Old results folder removed.")

        # Save traj sequence
        os.system(f"./img/mp4-from-folder.sh {trajSavePath} {startSeqInd + 1}")
        print(f"mp4 saved to {trajSavePath.strip(os.path.sep)}.mp4")

        if REMOVE_OLD_RESULTS:
            shutil.rmtree(trajSavePath)
            print("Old trajectory results folder removed.")
    except:
        print(
            "Failed to generate mp4 with script. Likely failed system requirements."
        )
