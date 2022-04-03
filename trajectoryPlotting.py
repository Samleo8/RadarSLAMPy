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
        plt.axis('square')
        fig, ax = plt.subplots()
        ax.plot(self.poses[:,0], self.poses[:,1], 'b-', label='Trajectory')
        ax.set_ylabel('y [m]')
        ax.set_xlabel('x [m]')
        ax.legend()
        fig.show()
        plt.waitforbuttonpress()
        plt.close(fig)

    def computeRMSE(self, estTraj):
        '''
        @brief Compute the Root Mean Square Error between the prediction and the actual trajectory
        '''
        estimatedPoses = estTraj.getPoseAtTime(self.timestamps)
        rmse = np.sqrt(np.mean((self.poses - estimatedPoses)**2))
        return rmse
    
def getGroundTruthTrajectory(gtPath : str):
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
        for row in gt_reader:
            timestamp = int(row[0]) # source_timestamp
            gt_timestamps.append(timestamp)
            x = float(row[2]) # x
            y = float(row[3]) # y
            gt_poses.append([x,y])
    gt_timestamps = np.array(gt_timestamps)
    gt_poses = np.array(gt_poses)
    # import pdb; pdb.set_trace()
    gt_poses = np.cumsum(gt_poses,axis=0) # integration
    return Trajectory(gt_timestamps, gt_poses)

def viewResults(gtTraj, estTraj, title):
    '''
    @brief Plot ground truth trajectory and estimated trajectory
    @param[in] gtTrajectory Ground truth trajectory
    @param[in] estTrajectory Estimated trajectory
    @param[in] title Title of the plot
    '''
    plt.axis('square')
    fig, ax = plt.subplots()
    ax.plot(gtTraj.poses[:,0], gtTraj.poses[:,1], 'b-', label='Ground Truth')
    ax.plot(estTraj.poses[:,0], estTraj.poses[:,1], 'r-', label='Estimated')
    ax.set_title(f'{title}: RMSE={estTraj.computeRMSE(gtTraj):.2f}')
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    ax.legend()
    savePath = os.path.join(os.getcwd(), 'results', title)
    fig.savefig(savePath)
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)
    
if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")
    gtPath = os.path.join("data", datasetName, "gt", "radar_odometry.csv")
    
    gtTraj = getGroundTruthTrajectory(gtPath)
    gtTraj.plotTrajectory()
    estArr = gtTraj.getPoseAtTime(gtTraj.timestamps)
    noise = np.random.multivariate_normal(mean=(0,0),cov=[[.0001,.0001],[.0001,.0001]],size=(gtTraj.nFrames))
    estArr = estArr + noise

    estTraj = Trajectory(gtTraj.timestamps, estArr)
    viewResults(gtTraj, estTraj, datasetName)