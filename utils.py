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
    rob_x, rob_y, rob_th = rob_pose[0], rob_pose[1], rob_pose[2]
    cth = np.cos(rob_th)
    sth = np.sin(rob_th)

    T = np.array([
        [cth, -sth, rob_x],
        [sth, cth, rob_y],
        [0, 0, 1],
    ])

    rel_x, rel_y, rel_th = rel_pose[0], rel_pose[1], rel_pose[2]
    xy_homo = T @ np.array([rel_x, rel_y, 1]).astype(float)
    th = normalize_angles(rel_th + rob_th)

    abs_pose = np.array([xy_homo[0], xy_homo[1], th])
    return abs_pose
