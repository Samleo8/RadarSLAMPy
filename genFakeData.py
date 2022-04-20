from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

from parseData import RANGE_RESOLUTION_CART_M


def plotFakeFeatures(srcCoord,
                     targetCoord,
                     targetCoord2=None,
                     clear=False,
                     show=False):
    if clear:
        plt.clear()

    plt.scatter(srcCoord[:, 0],
                srcCoord[:, 1],
                color='blue',
                marker='.',
                label='Features 0')

    plt.scatter(targetCoord[:, 0],
                targetCoord[:, 1],
                color='red',
                marker='+',
                label='Features 1')

    if targetCoord2 is not None:
        plt.scatter(targetCoord2[:, 0],
                    targetCoord2[:, 1],
                    color='green',
                    marker='x',
                    label='Features 2')
    plt.legend()

    plt.tight_layout()

    if show:
        plt.show()


def generateFakeCorrespondences(srcCoord, A, h):
    '''
    @brief Generate fake (but perfect) correspondence points given A, h transformation matrix
    @param[in] srcCoord Source coordinates (K x 2)
    @param[in] A Rotation matrix (2 x 2)
    @param[in] h Translation matrix (2 x 1)

    @return targetCoord = A @ srcCoord + h (K x 2)
    '''
    targetCoord = A @ srcCoord.T + h

    return targetCoord.T


def getRotationMatrix(theta_deg):
    th = np.deg2rad(theta_deg)
    cth = np.cos(th)
    sth = np.sin(th)

    return R


def generateFakeFeatures(n_points=100, max_range_m=10):
    # Generate artificial correspondences in m to pixel
    n_points = 100

    a = np.random.random((n_points, 2))

    a *= max_range_m / RANGE_RESOLUTION_CART_M
    return a