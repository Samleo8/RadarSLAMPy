import os
import sys
import csv
import numpy as np
import scipy
from matplotlib import pyplot as plt
from utils import *
from parseData import *

class Trajectory():
    def __init__(self, timestamps, poses):
        '''
        @param[in] timestamps np.ndarray of sorted timestamps (N)
        @param[in] pose_matrices np.ndarray of poses (N x 3)
        '''
        self.timestamps = np.array(timestamps)
        self.poses = np.array(poses)
        self.pose_transform = convertPoseToTransform(self.poses[-1])

    def getGroundTruthDeltasAtTime(self, time):
        '''
        @brief Given a timestamp, return the ground truth deltas at that time in (dx, dy, dth) list for debugging
        '''
        return self.gt_deltas[time]

    def appendRelativeDxDth(self, time, dx, dth):
        self.timestamps = np.append(self.timestamps, time)
        x, y, th = self.poses[-1]
        x += dx * np.cos(th)
        y += dx * np.sin(th)
        th += dth
        self.poses = np.vstack((self.poses, [x, y, th]))
    
    def appendRelativeTransform(self, time, R, h):
        '''
        @brief Append a relative transform to the trajectory
               h should already be scaled by radar resolution
        @param[in] time timestamp of the transform
        @param[in] R rotation matrix (2 x 2)
        @param[in] h translation vector (2 x 1)
        '''
        # Add to timestamps
        self.timestamps = np.append(self.timestamps, time)
        
        # Convert R, h to transformation matrix
        A = np.block([
            [R              , h             ],
            [np.zeros((1,2)), 1             ]
        ])

        # Update pose_transforms and poses
        self.pose_transform = A @ self.pose_transform
        new_pose = convertTransformToPose(self.pose_transform)

        # T = convertPoseToTransform(self.poses[-1])
        # xy = T @ [*h, 1]
        # dth = np.arctan2(T[1,0], T[0,0])
        # new_pose = [*xy[0], *xy[1], self.poses[-1,2] + dth]
        self.poses = np.vstack((self.poses, new_pose))

    def getPoseAtTimes(self, times):
        '''
        @brief Given timestamps, return the pose at that time using cubic interpolation
        @param[in] times np.ndarray of sorted timestamps
        '''
        try:
            # attempt cubic interpolation, will fail if insufficient points
            interpX = scipy.interpolate.interp1d(self.timestamps, self.poses[:,0], kind='cubic', bounds_error=False)
            interpY = scipy.interpolate.interp1d(self.timestamps, self.poses[:,1], kind='cubic', bounds_error=False)
            interpTH = scipy.interpolate.interp1d(self.timestamps, self.poses[:,2], kind='cubic', bounds_error=False)
            poses = np.vstack((interpX(times), interpY(times), interpTH(times))).T
        except:
            # if cubic interpolation fails, return recorded pose at nearest timestamp
            poses = np.zeros((len(times), 3))
            for i,t in enumerate(times):
                poses[i,:] = self.poses[np.argmin(np.abs(self.timestamps - t))]
        if poses.shape[0] == 1 and type(times) == int:
            poses = poses[0,:]
        return poses

    def plotTrajectory(self, title='My Trajectory', savePath=False):
        plt.clf()
        plt.plot(self.poses[:,0], self.poses[:,1], 'b-')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid(True)
        plt.axis('square')
        plt.title(title)
        if savePath:
            plt.tight_layout()
            plt.savefig(savePath)

def computePosesRMSE(gtPoses, estPoses):
    '''
    @brief Compute the Root Mean Square Error between the prediction and the actual poses
    '''
    euclidean_err = np.linalg.norm(gtPoses[:,:-1] - estPoses[:,:-1], axis=-1)
    rmse = np.sqrt(np.mean(euclidean_err**2))
    return rmse

def plotGtAndEstTrajectory(gtTraj, estTraj, title='GT and EST Trajectories', savePath=None, arrow=True):
    '''
    @brief Plot ground truth trajectory and estimated trajectory
    @param[in] gtTrajectory Ground truth trajectory
    @param[in] estTrajectory Estimated trajectory
    @param[in] title Title of the plot
    '''
    plt.clf()
    earliestTimestamp = estTraj.timestamps[0]
    latestTimestamp = estTraj.timestamps[-1]
    timestamps = [t for t in gtTraj.timestamps if earliestTimestamp <= t <= latestTimestamp]
    gtPoses = gtTraj.getPoseAtTimes(timestamps)
    estPoses = estTraj.getPoseAtTimes(timestamps)
    if arrow:
        quiver(gtPoses, c='b')
        quiver(estPoses, c='r')
    else:
        plt.plot(gtPoses[:,0], gtPoses[:,1], 'b-', label='Ground Truth')
        plt.plot(estPoses[:,0], estPoses[:,1], 'r-', label='Estimated')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.legend()
    plt.axis('square')
    plt.title(f'{title}: RMSE={computePosesRMSE(gtPoses, estPoses):.2f}')
    if savePath:
        plt.tight_layout()
        plt.savefig(savePath)

def getGroundTruthTrajectory(gtPath):
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
        d_xyths = {}
        x, y, th = 0, 0, 0
        for row in gt_reader:
            timestamp = int(row[9]) # destination_radar_timestamp
            gt_timestamps.append(timestamp)
            dx = float(row[2]) # x
            dy = float(row[3]) # y
            dth = float(row[7]) # yaw
            x += dx * np.cos(th) + dy * -np.sin(th)
            y += dx * np.sin(th) + dy * np.cos(th)
            th += dth
            th = normalize_angles(th)
            gt_poses.append([x,y,th])
            d_xyths[timestamp] = [dx,dy,dth]
    gt_timestamps = np.array(gt_timestamps)
    gt_poses = np.array(gt_poses)
    gt_traj = Trajectory(gt_timestamps, gt_poses)
    gt_traj.gt_deltas = d_xyths
    return gt_traj

def getGroundTruthTrajectoryGPS(gtPath):
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
            gt_poses.append([x,y,0])
    gt_timestamps = np.array(gt_timestamps)
    gt_poses = np.array(gt_poses)
    return Trajectory(gt_timestamps, gt_poses)
    
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
    estPoses = gtTraj.getPoseAtTimes(keyframe_timestamps)
    noise = np.random.multivariate_normal(mean=(.01,.05),cov=np.array([[.8,.2],[.2,.8]])*1e-2,size=(keyframe_timestamps.shape[0]))
    noise = np.cumsum(noise,axis=0) # integration
    estPoses[:,:2] += noise
    estTraj = Trajectory(keyframe_timestamps, estPoses)
    plotGtAndEstTrajectory(gtTraj, estTraj, datasetName)
    plt.show(block=True)