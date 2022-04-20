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
        @param[in] poses np.ndarray of x,y,theta poses (N x 3)
        '''
        self.timestamps = np.array(timestamps)
        self.poses = np.array(poses)
        x = self.poses[-1, 0]
        y = self.poses[-1, 1]
        theta = self.poses[-1, 2]
        first_pose_matrix_gt = np.array([[np.cos(theta), -np.sin(theta), x],
                                        [np.sin(theta), np.cos(theta), y],
                                        [0, 0, 1]])
        self.pose_matrix = np.array([first_pose_matrix_gt]) # N x 3 x 3
    
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
    def appendRelativePose(self, t, dist, theta):
        # Add to timestamps
        self.timestamps = np.append(self.timestamps, t)
        
        # not sure im computing p_{t+1} correctly given A, h
        # Get all the relevant transform variables
        # print(f"theta: {transf_th:.04f}, dx: {transf_x:.04f}, dy: {transf_y:.04f}")

        # Previous pose x y theta
        x, y, th = self.poses[-1,:]
        last_pose_matrix = self.pose_matrix[-1, :, :]

        new_transform = np.block([[A,               h],
                                  [np.zeros((1,2)), np.ones((1,1))]])
        new_pose_matrix = new_transform @ last_pose_matrix
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
    
    def getPoseAtTime(self, times):
        '''
        @brief Given timestamps, return the pose at that time using cubic interpolation
        @param[in] t float or np.ndarray of timestamps
        '''
        try:
            self.interpX = scipy.interpolate.interp1d(self.timestamps, self.poses[:,0], kind='cubic', bounds_error=False)
            self.interpY = scipy.interpolate.interp1d(self.timestamps, self.poses[:,1], kind='cubic', bounds_error=False)
            self.interpTH = scipy.interpolate.interp1d(self.timestamps, self.poses[:,1], kind='cubic', bounds_error=False)
            return np.vstack((self.interpX(times), self.interpY(times), self.interpTH(times))).T
        except:
            print("Warning: Could not interpolate trajectory")
            poses = np.zeros((len(times), 3))
            for i,t in enumerate(times):
                poses[i,:] = self.poses[np.argmin(np.abs(self.timestamps - t))]
            return poses

    def plotTrajectory(self, block=False):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(True)
        # ax.set_aspect('equal', adjustable='box')
        fig.canvas.draw()
        if not block:
            plt.show(block=False)
        ax.plot(self.poses[:,0], self.poses[:,1], 'r-', label='Trajectory')
        ax.legend()
        plt.pause(0.1)
        if block:
            plt.show(block=True)

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
            gt_poses.append([x,y,0])
    gt_timestamps = np.array(gt_timestamps)
    gt_poses = np.array(gt_poses)
    return Trajectory(gt_timestamps, gt_poses)

def computeRMSE(gtPoses, estPoses):
    '''
    @brief Compute the Root Mean Square Error between the prediction and the actual poses
    '''
    rmse = np.sqrt(np.mean((np.linalg.norm(gtPoses[:,:-1] - estPoses[:,:-1], axis=-1))**2))
    return rmse

def plotGtAndEstTrajectory(gtTraj, estTraj, title, savePath=None):
    '''
    @brief Plot ground truth trajectory and estimated trajectory
    @param[in] gtTrajectory Ground truth trajectory
    @param[in] estTrajectory Estimated trajectory
    @param[in] title Title of the plot
    '''
    plt.clf()
    earliestTimestamp = estTraj.timestamps[0]
    latestTimestamp = estTraj.timestamps[-1]
    timestamps = [x for x in gtTraj.timestamps if earliestTimestamp <= x <= latestTimestamp]
    gtPoses = gtTraj.getPoseAtTime(timestamps)
    #estPoses = estTraj.getPoseAtTime(timestamps)
    estPosesX = estTraj.pose_matrix[:, 0, 2] # slices all X positions, matrix[0,2]
    estPosesY = estTraj.pose_matrix[:, 1, 2]
    plt.plot(gtPoses[:,0], gtPoses[:,1], 'b-', label='Ground Truth')
    #plt.plot(estPoses[:,0], estPoses[:,1], 'r-', label='Estimated')
    plt.plot(estPosesX, estPosesY, 'r-', label='Estimated')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.legend()
    plt.axis('square')
    #plt.title(f'{title}: RMSE={computeRMSE(gtPoses, estPoses):.2f}')
    #plt.title(f'{title}: RMSE={computeRMSE(gtPoses, np.hstack((estPosesX, estPosesY))):.2f}')
    if savePath:
        plt.savefig(savePath)
    # plt.show(block=False)
    
if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    plt.ion()

    # gps ground truth
    if datasetName == "tiny":
        gtPath = os.path.join("data", datasetName, "gps", "gps.csv")
        gtTraj = getGroundTruthTrajectoryGPS(gtPath)
        gtTraj.plotTrajectory(block=True)

    # radar odometry ground truth
    gtPath = os.path.join("data", datasetName, "gt", "radar_odometry.csv")
    gtTraj = getGroundTruthTrajectory(gtPath)
    gtTraj.plotTrajectory(block=True)
    keyframe_timestamps = np.arange(gtTraj.timestamps[0], gtTraj.timestamps[-1], (gtTraj.timestamps[-1]-gtTraj.timestamps[0]) / 1000)
    estArr = gtTraj.getPoseAtTime(keyframe_timestamps)
    noise = np.random.multivariate_normal(mean=(.01,.05),cov=np.array([[.8,.2],[.2,.8]])*1e-2,size=(keyframe_timestamps.shape[0]))
    noise = np.cumsum(noise,axis=0) # integration
    estArr[:,:2] += noise
    estTraj = Trajectory(keyframe_timestamps, estArr)
    plotGtAndEstTrajectory(gtTraj, estTraj, datasetName)
    plt.show(block=True)