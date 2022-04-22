import os
import shutil
from matplotlib import pyplot as plt

import numpy as np
from getFeatures import appendNewFeatures
from parseData import getCartImageFromImgPaths, getRadarImgPaths
from trajectoryPlotting import Trajectory, getGroundTruthTrajectory, plotGtAndEstTrajectory
from utils import convertRandHtoDeltas, f_arr, radarImgPathToTimestamp
from Tracker import Tracker


class RawROAMSystem():

    def __init__(self, sequenceName: str, hasGroundTruth: bool = True) -> None:
        '''
        @brief Initializer for ROAM system
        @param[in] sequenceName Name of sequence. Should be in ./data folder
        @param[in] hasGroundTruth Whether sequence has ground truth to be used for plotting 
        '''

        # Data and timestamp paths
        self.sequenceName = sequenceName
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
        imgSavePath = os.path.join(".", "img", "roam",
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

        # Initialize Trajectories
        self.gtTraj = None  # set in run() function
        self.estTraj = None  # set in run() function

        # Initialize Tracker
        self.tracker = Tracker(self.sequenceName, self.imgPathArr,
                               self.filePaths)

        # TODO: Initialize mapping

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

        # Actually run the algorithm
        # Get initial Cartesian image
        prevImg = getCartImageFromImgPaths(imgPathArr, startSeqInd)

        # Get initial features from Cartesian image
        blobCoord = np.empty((0, 2))
        blobCoord, _ = appendNewFeatures(prevImg, blobCoord)

        for seqInd in range(startSeqInd + 1, sequenceSize):
            # Obtain image
            currImg = getCartImageFromImgPaths(imgPathArr, seqInd)

            # Perform tracking
            good_old, good_new = tracker.track(prevImg, currImg, blobCoord,
                                               seqInd)
            R, h = tracker.getTransform(good_old, good_new)

            # Update trajectory
            self.updateTrajectory(R, h, seqInd)

            # Plotting and prints and stuff
            self.plot(prevImg, currImg, good_old, good_new, R, h, seqInd)

            # Update incremental variables
            blobCoord = good_new.copy()
            prevImg = currImg

    # TODO: Move into trajectory class?
    def updateTrajectory(self, R, h, seqInd):
        imgPathArr = self.imgPathArr

        timestamp = radarImgPathToTimestamp(imgPathArr[seqInd])
        est_deltas = convertRandHtoDeltas(R, h)
        dx = est_deltas[0]
        dy = est_deltas[1]
        dth = est_deltas[2]
        self.estTraj.appendRelativeDeltas(timestamp, [dx, dy, dth])
        # self.estTraj.appendRelativeTransform(timestamp, R, h)

    def plot(self, prevImg, currImg, good_old, good_new, R, h, seqInd):
        plt.clf()
        
        self.tracker.plot(prevImg, currImg, good_old, good_new, seqInd)
        plt.show(block=False)

        self.plotTraj(seqInd, R, h)
        
        plt.pause(0.01)


    def plotTraj(self, seqInd, R, h):
        # Init locals
        gtTraj = self.gtTraj
        estTraj = self.estTraj
        imgPathArr = self.imgPathArr
        trajSavePath = self.filePaths["trajSave"]

        # Get timestamps for plotting etc
        currTimestamp = radarImgPathToTimestamp(imgPathArr[seqInd])
        gt_deltas_deg = gtTraj.getGroundTruthDeltasAtTime(currTimestamp)
        gt_deltas_deg[2] = np.rad2deg(gt_deltas_deg[2])

        est_deltas_deg = convertRandHtoDeltas(R, h)
        est_deltas_deg[2] = np.rad2deg(est_deltas_deg[2])

        print(f"GT Deltas: {f_arr(gt_deltas_deg)} (*dth in degrees)")
        print(f"Est Deltas: {f_arr(est_deltas_deg)} (*dth in degrees)")

        # Plot Trajectories
        toSaveTrajPath = os.path.join(trajSavePath, f"{seqInd:04d}.jpg")
        plotGtAndEstTrajectory(gtTraj,
                               estTraj, f'[{seqInd}]\n'
                               f'Est Pose: {f_arr(estTraj.poses[-1])}\n'
                               f'GT Deltas: {f_arr(gt_deltas_deg)}\n'
                               f'Est Deltas: {f_arr(est_deltas_deg)}\n',
                               savePath=toSaveTrajPath)


if __name__ == "__main__":
    import sys

    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    startImgInd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    endImgInd = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    REMOVE_OLD_RESULTS = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False

    # Initialize system
    system = RawROAMSystem(datasetName, hasGroundTruth=True)

    try:
        system.run()
    except KeyboardInterrupt:
        pass

    imgSavePath = system.filePaths["imgSave"]
    trajSavePath = system.filePaths["trajSave"]

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
