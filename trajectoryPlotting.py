import os
import sys
import csv
import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate
from utils import *
from parseData import *

class Trajectory():
    def __init__(self, timestamps, poses):
        '''
        @param[in] timestamps np.ndarray of timestamps (N)
        @param[in] poses np.ndarray of poses (N x 2)
        '''
        self.timestamps = timestamps
        self.poses = poses
        self.nFrames = timestamps.shape[0]
        self.interpX = scipy.interpolate.interp1d(self.timestamps, self.poses[:,0], kind='cubic', bounds_error=False)
        self.interpY = scipy.interpolate.interp1d(self.timestamps, self.poses[:,1], kind='cubic', bounds_error=False)
    
    def getPoseAtTime(self, t):
        '''
        @brief Given timestamps, return the pose at that time using cubic interpolation
        @param[in] t float or np.ndarray of timestamps
        '''
        return np.vstack((self.interpX(t), self.interpY(t))).T

    def plotTrajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(self.poses[:,0], self.poses[:,1], 'r-', label='Trajectory')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(True)
        ax.legend()
        # ax.set_aspect('equal', adjustable='box')
        fig.canvas.draw()
        plt.show(block=True)

    def computeRMSE(self, estTraj):
        '''
        @brief Compute the Root Mean Square Error between the prediction and the actual trajectory
        '''
        estimatedPoses = estTraj.getPoseAtTime(self.timestamps)
        rmse = np.sqrt(np.mean((np.linalg.norm(self.poses - estimatedPoses, axis=-1))**2))
        return rmse

def getGroundTruthTrajectory(gtPath : str):
    '''
    @brief Returns ground truth trajectory given radar_odometry.csv
    @param[in] gtPath Path to ground truth file
    @return Trajectory object
    '''
    with open(gtPath) as gt_file:
        gt_reader = csv.reader(gt_file)
        _ = next(gt_file) # headers
        gt_timestamps = []
        gt_poses = []

        x, y, th = 0, 0, 0
        for row in gt_reader:
            timestamp = int(row[0]) # source_timestamp
            gt_timestamps.append(timestamp)
            dx = float(row[2]) # x
            dy = float(row[3]) # y
            dth = float(row[7]) # yaw
            x += dx * np.cos(th) + dy * -np.sin(th)
            y += dx * np.sin(th) + dy * np.cos(th)
            th += dth
            gt_poses.append([x,y])
    gt_timestamps = np.array(gt_timestamps)
    gt_poses = np.array(gt_poses)
    return Trajectory(gt_timestamps, gt_poses)

def getGroundTruthTrajectoryGPS(gtPath : str):
    '''
    @brief Returns ground truth trajectory given gps.csv
    @param[in] gtPath Path to ground truth file
    @return Trajectory object
    '''
    with open(gtPath) as gt_file:
        gt_reader = csv.reader(gt_file)
        _ = next(gt_file) # headers
        gt_timestamps = []
        gt_poses = []
        for row in gt_reader:
            timestamp = int(row[0]) # source_timestamp
            gt_timestamps.append(timestamp)
            x = float(row[2]) # x
            y = float(row[3]) # y
            gt_poses.append([x,y])
    gt_timestamps = np.array(gt_timestamps)
    gt_poses = np.array(gt_poses)
    return Trajectory(gt_timestamps, gt_poses)

def viewResults(gtTraj, estTraj, title):
    '''
    @brief Plot ground truth trajectory and estimated trajectory
    @param[in] gtTrajectory Ground truth trajectory
    @param[in] estTrajectory Estimated trajectory
    @param[in] title Title of the plot
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(gtTraj.poses[:,0], gtTraj.poses[:,1], 'b-', label='Ground Truth')
    ax.plot(estTraj.poses[:,0], estTraj.poses[:,1], 'r-', label='Estimated')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True)
    ax.legend()
    ax.set_title(f'{title}: RMSE={estTraj.computeRMSE(gtTraj):.2f}')
    # ax.set_aspect('equal', adjustable='box')
    fig.canvas.draw()
    savePath = os.path.join(os.getcwd(), 'results', title)
    plt.savefig(savePath)
    plt.show(block=True)
    
if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    plt.ion()

    # gps ground truth
    if datasetName == "tiny":
        gtPath = os.path.join("data", datasetName, "gps", "gps.csv")
        gtTraj = getGroundTruthTrajectoryGPS(gtPath)
        gtTraj.plotTrajectory()

    # radar odometry ground truth
    gtPath = os.path.join("data", datasetName, "gt", "radar_odometry.csv")
    gtTraj = getGroundTruthTrajectory(gtPath)
    gtTraj.plotTrajectory()
    keyframe_timestamps = np.arange(gtTraj.timestamps[0], gtTraj.timestamps[-1], (gtTraj.timestamps[-1]-gtTraj.timestamps[0]) / 1000)
    estArr = gtTraj.getPoseAtTime(keyframe_timestamps)
    noise = np.random.multivariate_normal(mean=(.01,.05),cov=np.array([[.8,.2],[.2,.8]])*1e-2,size=(keyframe_timestamps.shape[0]))
    noise = np.cumsum(noise,axis=0) # integration
    estArr = estArr + noise
    estTraj = Trajectory(keyframe_timestamps, estArr)
    viewResults(gtTraj, estTraj, datasetName)