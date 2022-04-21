import os
import sys
import csv
import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate
from utils import *
from parseData import *

class Trajectory():
    def __init__(self, timestamps, pose_transforms):
        '''
        @param[in] timestamps np.ndarray of sorted timestamps (N)
        @param[in] pose_matrices np.ndarray of transformation matrices (N x 3 x 3)
        '''
        assert timestamps.shape[0] == pose_transforms.shape[0], "first axis of timestamps and pose_transforms must be same length"

        self.timestamps = np.array(timestamps)
        self.pose_transforms = np.array(pose_transforms)
    
    '''
    Ax_1 + b = x_0
    => x_1 = A^-1(x_0 - b)
    A_prime  [A h]
             [0 1]
    A_prime^-1 @ prev_pose = new_pose

    A_prime^-1 : transformation from the old robot frame to the new robot frame
    initial: global frame == robot_frame_0
    pose_i = A_primei^-1 @ A_prime{i-1}^-1 @ A_prime{i-2}^-1 ... @ I
    pose: 
    [R t]
    [0 1]
    initial:
    [100]
    [010]
    [001]
    '''
    def appendRelativePoseTransform(self, t, R, t):
        # Add to timestamps
        self.timestamps = np.append(self.timestamps, t)


        
        # not sure im computing p_{t+1} correctly given A, h
        # Get all the relevant transform variables
        transf_th = float(np.arctan2(A[0,0], A[1,0]))
        transf_x = float(h[0])
        transf_y = float(h[1])
        print(f"theta: {transf_th:.04f}, dx: {transf_x:.04f}, dy: {transf_y:.04f}")

        # Previous pose x y theta
        x, y, th = self.poses[-1,:]
        last_pose_matrix = self.pose_matrix[-1, :, :]
        new_transform = np.block([[A,               h],
                                    [np.zeros((1,2)), np.ones((1,1))]])
        new_pose_matrix = np.linalg.inv(new_transform) @ last_pose_matrix
        self.pose_matrix = np.concatenate((self.pose_matrix,
                                        np.expand_dims(new_pose_matrix, axis=0)),
                                            axis = 0)
        
        
        # Convert from relative pose wrt to current robot position to absolute pose
        #x_p, y_p, th_p = rel_to_abs_pose([x,y,th], [transf_x, transf_y, transf_th])
        
        x_prime = x - transf_x
        y_prime = y - transf_y
        state = np.array([x_prime, y_prime])
        new_state = np.linalg.inv(A) @ state
        x_p = new_state[0]
        y_p = new_state[1]
        th_p = normalize_angles(th - transf_th)

        self.poses = np.vstack((self.poses, np.array([x_p, y_p, th_p])))

        print(f"Time {t}: [{x_p:.2f},{y_p:.2f},{th_p:.2f}]")

    def getPoseTransformsAtTimes(self, times):
        '''
        @brief Given timestamps, return the recorded pose transforms at the closest timestamps
        @param[in] times np.ndarray of sorted timestamps
        '''
        # TODO: Vectorize
        pose_tranforms = np.zeros((len(times), 3, 3))
        for i,t in enumerate(times):
            pose_tranforms[i,:,:] = self.pose_transforms[np.argmin(np.abs(self.timestamps - t))]
        return pose_tranforms

    def plotTrajectory(self, title='My Trajectory', savePath=False):
        plt.clf()
        poses = convertTransformToPose(self.pose_transforms)
        plt.plot(poses[:,0], poses[:,1], 'b-')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.grid(True)
        plt.axis('square')
        plt.title(title)
        if savePath:
            plt.savefig(savePath)
        plt.show(block=True)

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
            gt_poses.append([x,y,th])
    gt_timestamps = np.array(gt_timestamps)
    gt_poses = np.array(gt_poses)
    gt_pose_transforms = convertPoseToTransform(gt_poses)
    return Trajectory(gt_timestamps, gt_pose_transforms)

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
    gt_pose_transforms = convertPoseToTransform(gt_poses)
    return Trajectory(gt_timestamps, gt_pose_transforms)

def computePosesRMSE(gtPoses, estPoses):
    '''
    @brief Compute the Root Mean Square Error between the prediction and the actual poses
    '''
    euclidean_err = np.linalg.norm(gtPoses[:,:-1] - estPoses[:,:-1], axis=-1)
    rmse = np.sqrt(np.mean(euclidean_err**2))
    return rmse

def plotGtAndEstTrajectory(gtTraj, estTraj, title='GT and EST Trajectories', savePath=None):
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
    gtPoseTransforms = gtTraj.getPoseTransformsAtTimes(timestamps)
    gtPoses = convertTransformToPose(gtPoseTransforms)
    estPoseTransforms = estTraj.getPoseTransformsAtTimes(timestamps)
    estPoses = convertTransformToPose(estPoseTransforms)
    plt.plot(gtPoses[:,0], gtPoses[:,1], 'b-', label='Ground Truth')
    plt.plot(estPoses[:,0], estPoses[:,1], 'r-', label='Estimated')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.legend()
    plt.axis('square')
    plt.title(f'{title}: RMSE={computePosesRMSE(gtPoses, estPoses):.2f}')
    if savePath:
        plt.savefig(savePath)
    
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
    estPoseTransforms = gtTraj.getPoseTransformsAtTimes(keyframe_timestamps)
    estPoses = convertTransformToPose(estPoseTransforms)
    noise = np.random.multivariate_normal(mean=(.01,.05),cov=np.array([[.8,.2],[.2,.8]])*1e-2,size=(keyframe_timestamps.shape[0]))
    noise = np.cumsum(noise,axis=0) # integration
    estPoses[:,:2] += noise
    estPoseTransforms = convertPoseToTransform(estPoses)
    estTraj = Trajectory(keyframe_timestamps, estPoseTransforms)
    plotGtAndEstTrajectory(gtTraj, estTraj, datasetName)
    plt.show(block=True)