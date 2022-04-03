import numpy as np
import cv2
import os, sys
import csv
from parseData import *
from matplotlib import pyplot as plt
import scipy.interpolate

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
        fig.show()

    def computeRMSE(self, estTraj):
        '''
        @brief Compute the Root Mean Square Error between the prediction and the actual trajectory
        '''
        estimatedPoses = estTraj.getPoseAtTime(self.timestamps)
        rmse = np.sqrt(np.mean((np.linalg.norm(self.poses - estimatedPoses))**2))
        return rmse
    
def getGroundTruthTrajectory(gtPath : str, integrate=False):
    '''
    @brief Returns ground truth trajectory
    @param[in] gtPath Path to ground truth file
    @return Trajectory object
    '''
    with open(gtPath) as gt_file:
        gt_reader = csv.reader(gt_file)
        _ = next(gt_file) # headers
        gt_timestamps = []
        gt_poses = []
        # import pdb; pdb.set_trace()
        for row in gt_reader:
            timestamp = int(row[0]) # source_timestamp
            gt_timestamps.append(timestamp)
            x = float(row[2]) # x
            y = float(row[3]) # y
            gt_poses.append([x,y])
    gt_timestamps = np.array(gt_timestamps)
    gt_poses = np.array(gt_poses)
    if integrate:
        gt_poses = np.cumsum(gt_poses,axis=0) # integration
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
    fig.show()
    savePath = os.path.join(os.getcwd(), 'results', title)
    plt.savefig(savePath)
    
if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    # gps ground truth
    gtPath = os.path.join("data", datasetName, "gps", "gps.csv")
    gtTraj = getGroundTruthTrajectory(gtPath, integrate=False)
    gtTraj.plotTrajectory()
    estArr = gtTraj.getPoseAtTime(gtTraj.timestamps)
    noise = np.random.multivariate_normal(mean=(1e-11,3e-11),cov=np.array([[1,.5],[.2,1]]) * 1e-9,size=(gtTraj.nFrames))
    noise = np.cumsum(noise,axis=0) # integration
    estArr = estArr + noise
    estTraj = Trajectory(gtTraj.timestamps, estArr)
    viewResults(gtTraj, estTraj, datasetName)

    # radar odometry ground truth seems fishy
    gtPath = os.path.join("data", datasetName, "gt", "radar_odometry.csv")
    gtTraj = getGroundTruthTrajectory(gtPath, integrate=True)
    gtTraj.plotTrajectory()
    plt.waitforbuttonpress()