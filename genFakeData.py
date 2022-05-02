import numpy as np
import matplotlib.pyplot as plt
from utils import getRotationMatrix, invert_transform
from parseData import *

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
                     show=False,
                     plotDisplace = False):
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
                    label=f'Instantaneous Radar Scan{title_append}')

    if targetCoord is not None:
        plt.scatter(targetCoord[:, 0],
                    targetCoord[:, 1],
                    color='red',
                    marker='+',
                    alpha=alpha,
                    label=f'Scan with Distortion{title_append}')

    if targetCoord2 is not None:
        plt.scatter(targetCoord2[:, 0],
                    targetCoord2[:, 1],
                    color='green',
                    marker='x',
                    alpha=alpha,
                    label=f'Original Points{title_append}')

    if plotDisplace:
        for i in range(targetCoord.shape[0]):
            src_x = srcCoord[i,0]
            src_y = srcCoord[i,1]
            tar_x = targetCoord[i,0]
            tar_y = targetCoord[i,1]
            dx = tar_x - src_x
            dy = tar_y - src_y
            #plt.arrow(src_x, src_y, dx, dy)
            plt.plot([src_x, tar_x], [src_y, tar_y], color = 'g')

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

def convertPolarPointsToCartesian(points):
    angles = points[:, 0] # - to match data convention: clockwise scan
    ranges = points[:, 1]
    x = np.expand_dims(ranges * np.cos(angles), axis = 1)
    y = np.expand_dims(ranges * np.sin(angles), axis = 1)
    return np.hstack((x, y))

def generateFakeCorrespondencesPolar(currentFrame=None,
                                    n_points=100,
                                    theta_max_deg=20,
                                    max_translation_m=3):
    '''
    @brief Generate fake correspondences with transform, randomly generated from max range and degree
    @param[in] currentFrame Source coordinate to transform from. If none, will randomly generate features
    @param[in] n_points Number of points to generate, only applies if currentFrame = None
    @param[in] theta_max_deg Maximum degree of rotation
    @param[in] max_range_m Maximum range (for translation) in meters

    @return currentFrame Generated or passed in currentFrame
    @return targetCoord Corresponding targetCoord generated using (theta_deg, h)
    @return theta_deg Theta component of transform
    @return h Translation component of transform
    '''

    if currentFrame is None:
        print("Generating fake features..")
        max_range_m = max_translation_m * 3
        currentFrame = generateFakeFeaturesPolar(n_points, max_range_m)
        #print(currentFrame.shape)
        currentFrame = convertPolarPointsToCartesian(currentFrame)
    else:
        n_points = currentFrame.shape[0]

    theta_deg = np.random.random() * theta_max_deg
    R = getRotationMatrix(theta_deg, degrees=True)
    #h = generateTranslationVector(max_translation_m)
    h = np.array([[0], [0]])
    # transform = np.block([[R, h],
    #                       [np.zeros((2,)), 1]])
    # T_inv = invert_transform(transform)
    # R_inv = T_inv[:2, :2]
    # h_inv = T_inv[:2, 2:]
    #print(currentFrame.shape)
    groundTruth = transformCoords(currentFrame, R, h)

    return groundTruth, currentFrame, theta_deg, h

def distort(coords, velocity, frequency, h):
    
    coords_norm = coords - h.flatten() # N x 2
    angles = np.arctan2(coords_norm[:, 1], -coords_norm[:, 0]) # - x to follow clockwise convention
    period = 1 / frequency
    times = angles / (2 * np.pi) * period
    #print(angles) # lesson: need to use arctan2 wisely, it wraps [-pi, pi]
    
    if coords.shape[1] == 2:
        coords = np.hstack((coords, np.ones((coords.shape[0], 1)))) # N x 3

    # Distort
    displacement = np.expand_dims(velocity, axis = 1) * times
    #print(displacement)
    #print(displacement)
    dx = displacement[0, :]
    print(dx)
    dy = displacement[1, :]
    dtheta = displacement[2, :] / 180 * np.pi
    c = np.cos(dtheta)
    s = np.sin(dtheta)
    ones = np.ones(times.shape)
    zeros = np.zeros(times.shape)
    distortion = np.array([[ c, s, -s*dy - c*dx],
                           [-s, c, -c*dy + s*dx],
                           [zeros, zeros, ones]]) # 3 x 3 x N, need to invert?
    distorted = np.transpose(distortion, axes = (2, 0, 1)) @ np.expand_dims(coords, axis = 2) # N x 3 x 1
    distorted = distorted[:, :2, 0]
    return distorted

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

def generateFakeFeaturesPolar(n_points=100, max_range_m=10):
    # Generate artificial correspondences in m to pixel
    data_size = (n_points, 1)
    a_range = np.random.random(data_size)
    thetas = np.arange(400) * 2 * np.pi / 400
    a_angle = np.random.choice(thetas, data_size)
    #print(a_angle)
    a_range *= max_range_m / RANGE_RESOLUTION_CART_M
    return np.hstack((a_angle, a_range))