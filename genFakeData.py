from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

from parseData import RANGE_RESOLUTION_CART_M


def transformCoords(srcCoord, A, h):
    '''
    @brief Transform coordinates to get correspondence points given A, h transformation matrix
    @param[in] srcCoord Source coordinates (K x 2)
    @param[in] A Rotation matrix (2 x 2)
    @param[in] h Translation matrix (2 x 1)

    @return targetCoord = A @ srcCoord + h (K x 2)
    '''
    targetCoord = A @ srcCoord.T + h

    return targetCoord.T


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


def generateFakeCorrespondences(srcCoord=None,
                                n_points=100,
                                theta_max_deg=20,
                                max_translation_m=3):
    '''
    @brief Generate fake correspondences with transform, randomly generated from max range and degree
    @param[in] srcCoord Source coordinate to transform from. If none, will randomly generate features
    @param[in] n_points Number of points to generate, only applies if srcCoord = None
    @param[in] theta_max_deg Maximum degree of rotation
    @param[in] max_range_m Maximum range (for translation) in meters

    @return srcCoord Generated or passed in srcCoord
    @return targetCoord Corresponding targetCoord generated using (theta_deg, h)
    @return theta_deg Theta component of transform
    @return h Translation component of transform
    '''

    if srcCoord is None:
        print("Generating fake features..")
        max_range_m = max_translation_m * 3
        srcCoord = generateFakeFeatures(n_points, max_range_m)
    else:
        n_points = srcCoord.shape[0]

    theta_deg = np.random.random() * theta_max_deg
    R = getRotationMatrix(theta_deg)
    h = generateTranslationVector(max_translation_m)

    targetCoord = transformCoords(srcCoord, R, h)

    return srcCoord, targetCoord, theta_deg, h


def getRotationMatrix(theta_deg):
    th = np.deg2rad(theta_deg)

    cth = np.cos(th)
    sth = np.sin(th)
    R = np.array([[cth, -sth], [sth, cth]])

    return R


def generateTranslationVector(max_range_m=10):
    h = np.random.random((2, 1))
    h *= max_range_m / RANGE_RESOLUTION_CART_M
    return h


def generateFakeFeatures(n_points=100, max_range_m=10):
    # Generate artificial correspondences in m to pixel
    a = np.random.random((n_points, 2))

    a *= max_range_m / RANGE_RESOLUTION_CART_M
    return a