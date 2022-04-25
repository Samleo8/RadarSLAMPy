import numpy as np
import time
import os

from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox


def tic():
    return time.time()


def toc(tic):
    return time.time() - tic

def f_arr(xs, th_deg=False):
    xs_str = [f'{x:.3f}' for x in xs]
    if th_deg:
        xs_str[-1] += '°'
    return '[' + ','.join(xs_str) + ']'

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
    xs = poses[:, 0]
    ys = poses[:, 1]
    ths = poses[:, 2]
    cths = np.cos(ths)
    sths = np.sin(ths)
    pose_transform = np.zeros((len(ths), 3, 3))
    pose_transform[:, 0, 0] = cths
    pose_transform[:, 0, 1] = -sths
    pose_transform[:, 1, 0] = sths
    pose_transform[:, 1, 1] = cths
    pose_transform[:, 0, 2] = xs
    pose_transform[:, 1, 2] = ys
    pose_transform[:, 2, 2] = 1
    if single:
        pose_transform = pose_transform[0, :, :]
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
    ths = np.arctan2(pose_transforms[:, 1, 0], pose_transforms[:, 0, 0])
    xs = pose_transforms[:, 0, 2]
    ys = pose_transforms[:, 1, 2]
    pose_transforms = np.stack([xs, ys, ths], axis=1)
    if single:
        pose_transforms = pose_transforms[0, :]
    return pose_transforms


def flatten(x):
    return x.reshape(x.shape[0])


def convertRandHtoDeltas(R, h):
    dx = float(*h[0])
    dy = float(*h[1])
    dth = np.arctan2(R[1, 0], R[0, 0])
    return np.array([dx, dy, dth])


def quiver(poses, c='r', label=None):
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
        label=label
    )


def plt_full_extent(ax, pad=0.0):
    """
    @brief Get the full extent of a plt axes, including axes labels, tick labels, and titles.
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]

    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def plt_savefig_by_axis(filePath, fig, ax, pad=0.0):
    '''
    @brief Save a plt figure by extent of its axis (allows us to save subplots)
    @param[in] filePath Path to save figure to
    @param[in] fig Overall figure containing axis
    @param[in] ax Axis to save
    '''
    extent = plt_full_extent(ax, pad).transformed(fig.dpi_scale_trans.inverted())
    # extent = ax.get_tightbbox(fig.canvas.get_renderer())
    fig.savefig(filePath, bbox_inches=extent)

def invert_transform(T):
    theta = np.arctan2(T[1, 0], T[0, 0])
    x = T[0, 2]
    y = T[1, 2]
    c = np.cos(theta)
    s = np.sin(theta)
    T_inv = np.array([[ c, s, -s * y - c * x],
                      [-s, c, -c * y + s * x],
                      [ 0, 0, 1]])
    return T_inv