import numpy as np
import math
import time
import os
import numpy as np
from matplotlib import pyplot as plt

def tic():
    return time.time()

def toc(tic):
    return time.time() - tic

def f_arr(xs):
    return '[' + ','.join([f'{x:03f}' for x in xs]) + ']'

def radarImgPathToTimestamp(radarImgPath):
    """
    eg: radarImgPathToTimestamp('data\\tiny\\radar\\1547131046353776.png') -> 1547131046353776
    """
    return int(os.path.basename(radarImgPath)[:-4])

def normalize_angles(th):
    """
    Normalize an angle to be between -pi and pi
    """
    return (th + np.pi) % (2 * np.pi) - np.pi

def getRotationMatrix(th, degrees=False):
    if degrees:
        th = np.deg2rad(th)
    cth = np.cos(th)
    sth = np.sin(th)
    R = np.array([[cth, -sth], [sth, cth]])

    return R

def convertPoseToTransform(poses):
    '''
    @param[in] poses np.ndarray of (3,) or (N x 3)
    @return pose_transforms np.ndarray of (3 x 3) or (N x 3 x 3)
    '''
    if type(poses) == list:
        poses = np.array(poses)
    single = False
    if len(poses.shape) == 1:
        single = True
        poses = np.expand_dims(poses, axis=0)
    xs = poses[:,0]
    ys = poses[:,1]
    ths = poses[:,2]
    cths = np.cos(ths)
    sths = np.sin(ths)
    pose_transform = np.zeros((len(ths),3,3))
    pose_transform[:,0,0] = cths
    pose_transform[:,0,1] = -sths
    pose_transform[:,1,0] = sths
    pose_transform[:,1,1] = cths
    pose_transform[:,0,2] = xs
    pose_transform[:,1,2] = ys
    pose_transform[:,2,2] = 1
    if single:
        pose_transform = pose_transform[0,:,:]
    return pose_transform

def convertTransformToPose(pose_transforms):
    '''
    @param[in] pose_transforms np.ndarray of (3 x 3) or (N x 3 x 3)
    @return poses np.ndarray of (3,) or (N x 3)
    '''
    if type(pose_transforms) == list:
        pose_transforms = np.array(pose_transforms)
    single = False
    if len(pose_transforms.shape) == 2:
        single = True
        pose_transforms = np.expand_dims(pose_transforms, axis=0)
    ths = np.arctan2(pose_transforms[:,1,0], pose_transforms[:,0,0])
    xs = pose_transforms[:,0,2]
    ys = pose_transforms[:,1,2]
    pose_transforms = np.stack([xs, ys, ths], axis=1)
    if single:
        pose_transforms = pose_transforms[0, :]
    return pose_transforms

def flatten(x):
    return x.reshape(x.shape[0])

def convertRandHtoDeltas(R, h):
    dx = float(*h[0])
    dy = float(*h[1])
    dth = np.arctan2(R[1,0], R[0,0])
    return np.array([dx, dy, dth])

def quiver(poses, c='r'):
    poses = np.array(poses)
    plt.quiver(
        poses[:, 0],
        poses[:, 1],
        np.cos(poses[:, 2]),
        np.sin(poses[:, 2]),
        color=c,
        width=0.02,
        scale=10,
        alpha=.5,
    )