import os

import numpy as np
from getFeatures import appendNewFeatures
from parseData import getCartImageFromImgPaths, getRadarImgPaths
from trajectoryPlotting import Trajectory, getGroundTruthTrajectory
from utils import radarImgPathToTimestamp
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

        dataPath = os.path.join("data", sequenceName, "radar")
        timestampPath = os.path.join("data", sequenceName,
                                          "radar.timestamps")

        assert os.path.exists(dataPath), \
            "Failed to find radar data for sequence " + self.sequenceName
        assert os.path.exists(timestampPath), \
            "Failed to find radar timestamp information for sequence " + self.sequenceName

        # Incremental streaming
        # Obtain image paths and set a starting index
        self.imgPathArr = getRadarImgPaths(dataPath, timestampPath)
        self.sequenceSize = len(self.imgPathArr)

        # Create Save paths for imaging
        imgSavePath = os.path.join(".", "img", "roam", sequenceName).strip(os.path.sep)
        trajectorySavePath = imgSavePath + '_traj'

        featureSavePath = os.path.join(imgSavePath + f"_{self.imgInd}.npz")
        os.makedirs(imgSavePath, exist_ok=True)
        os.makedirs(trajectorySavePath, exist_ok=True)

        # Initialize paths as dictionary
        self.filePaths = {
            "data": dataPath,
            "timestamp": timestampPath,
            "traj": trajectorySavePath,
            "trajectory": trajectorySavePath,
            "featureSave": featureSavePath,
            "imgSave": imgSavePath
        }

        # Initialize Trajectories
        self.gtTraj = None # set in run() function
        self.estTraj = None # set in run() function

        # Initialize Tracker
        self.tracker = Tracker(self.sequenceName, self.imgPathArr, self.filePaths, self.hasGroundTruth)

        self.hasGroundTruth = hasGroundTruth

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

        # tracker.initTraj(estTraj, gtTraj)

        # Actually run the algorithm
        # Get initial features
        blobCoord = np.empty((0, 2))
        blobCoord, _ = appendNewFeatures(prevImg, blobCoord)

        # Get initial Cartesian image
        prevImg = getCartImageFromImgPaths(imgPathArr, startSeqInd)

        for seqInd in range(startSeqInd + 1, sequenceSize):
            currImg = getCartImageFromImgPaths(imgPathArr, seqInd)

            good_old, good_new = tracker.track(prevImg, currImg, blobCoord, seqInd)
            R, h = tracker.getTransform(good_old, good_new)



            pass

        # self.updateTrajFromTracker()
