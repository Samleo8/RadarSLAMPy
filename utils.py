import numpy as np

import time
import os
import numpy as np
from matplotlib import pyplot as plt

def tic():
    return time.time()

def toc(tic):
    return time.time() - tic

def normalize_angles(th):
    """
    Normalize an angle to be between -pi and pi
    """
    return (th + np.pi) % (2 * np.pi) - np.pi

def abs_to_rel_pose(rob_pose, rel_pose):
    """
    rob_pose wrt to world
    rel_pose wrt to world
    return rel_pose wrt to rob
    """
    T = np.array(
        [
            [np.cos(rob_pose[2]), -np.sin(rob_pose[2]), rob_pose[0]],
            [np.sin(rob_pose[2]), np.cos(rob_pose[2]), rob_pose[1]],
            [0, 0, 1],
        ]
    )
    xy = np.linalg.inv(T) @ np.array([rel_pose[0], rel_pose[1], 1])
    th = normalize_angles(rel_pose[2] - rob_pose[2])
    rel_pose = np.array([float(xy[0]), float(xy[1]), th])
    return rel_pose


def rel_to_abs_pose(rob_pose, rel_pose):
    """
    rob_pose wrt to world
    rel_pose wrt to robot
    return rel_pose wrt to world
    """
    T = np.array(
        [
            [np.cos(rob_pose[2]), -np.sin(rob_pose[2]), rob_pose[0]],
            [np.sin(rob_pose[2]), np.cos(rob_pose[2]), rob_pose[1]],
            [0, 0, 1],
        ]
    )
    xy = T @ np.array([rel_pose[0], rel_pose[1], 1])
    th = normalize_angles(rel_pose[2] + rob_pose[2])
    abs_pose = np.array([float(xy[0]), float(xy[1]), th])
    return abs_pose
