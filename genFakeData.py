import numpy as np
import matplotlib.pyplot as plt
from utils import getRotationMatrix

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
                     title_append="",
                     alpha=1,
                     clear=False,
                     show=False):
    if clear:
        plt.clear()

    if len(title_append) > 0:
        title_append = " " + title_append

    if srcCoord is not None:
        plt.scatter(srcCoord[:, 0],
                    srcCoord[:, 1],
                    color='blue',
                    marker='.',
                    alpha=alpha,
                    label=f'Features 0{title_append}')

    if targetCoord is not None:
        plt.scatter(targetCoord[:, 0],
                    targetCoord[:, 1],
                    color='red',
                    marker='+',
                    alpha=alpha,
                    label=f'Features 1{title_append}')

    if targetCoord2 is not None:
        plt.scatter(targetCoord2[:, 0],
                    targetCoord2[:, 1],
                    color='green',
                    marker='x',
                    alpha=alpha,
                    label=f'Features 2{title_append}')
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
    R = getRotationMatrix(theta_deg, degrees=True)
    h = generateTranslationVector(max_translation_m)

    targetCoord = transformCoords(srcCoord, R, h)

    return srcCoord, targetCoord, theta_deg, h


def addNoise(data, variance=2.5):
    '''
    @brief Add 0-mean Gaussian random noise to correspondence data
    @param[in] data Data to add noise to
    @param[in] variance Variance for Gaussian noise
    '''
    noise = np.random.normal(0, variance, size=data.shape)
    noisy_data = data + noise

    return noisy_data


def createOutliers(data, n_outliers, noiseToAdd=10):
    '''
    @brief Create outliers by adding a lot of noise to randomly chosen n_outliers
    @param[in] data Data to create outliers in
    @param[in] n_outliers Number of outliers forced into data
    @param[in] noiseToAdd Amount of guaranteed base noise to add

    @return noisy_data Noisy data with outliers
    @return outlier_ind Indices of outliers
    '''
    n_outliers = int(n_outliers)
    
    K, dim = data.shape
    assert n_outliers < K, "Cannot have more outliers than data"

    outlier_ind = np.random.choice(np.arange(K),
                                   size=n_outliers,
                                   replace=False)

    # Create very noisy data
    # Allow for noise in negative direction too, starting with big outlier movement
    noise = np.random.random((n_outliers, dim))
    noise[noise > 0.5] = +noiseToAdd
    noise[noise < 0.5] = -noiseToAdd

    # Add small noise to the noiseToAdd so as to create randomness
    noisy_data = data.copy()
    noisy_data[outlier_ind, :] += addNoise(noise, 0.5)

    return noisy_data, outlier_ind


def generateTranslationVector(max_range_m=10):
    h = np.random.random((2, 1))
    h *= max_range_m / RANGE_RESOLUTION_CART_M
    # limit y translation
    h[1] /= 300
    return h


def generateFakeFeatures(n_points=100, max_range_m=10):
    # Generate artificial correspondences in m to pixel
    a = np.random.random((n_points, 2))

    a *= max_range_m / RANGE_RESOLUTION_CART_M
    return a