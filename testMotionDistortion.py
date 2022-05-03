from getTransformKLT import *
from genFakeData import *
from motionDistortion import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N = 100
    outlier_rate = 0.4
    noisy = False
    useOld = False
    frequency = 4
    period = 1 / frequency

    # Generate fake data
    # srcCoord = frame 1
    # currentFrame = frame 2
    # R_theta_deg @ currentFrame + h = srcCoord
    # pose at frame 1 (srcCoord) is I
    groundTruth, currentFrame, theta_deg, h = generateFakeCorrespondencesPolar(n_points=N)
    # This theta_deg reflects the radar's own motion. To distort points, the opposite must be used
    velocity = np.array([h[0, 0], h[1, 0], theta_deg]) / period
    distorted = distort(currentFrame, velocity, frequency, h)
    if noisy:
        currentFrame, outlier_ind = createOutliers(currentFrame, int(N * outlier_rate), 
                                    noiseToAdd=10)

    '''
    Naive Fit: rotation and translation
    '''
    if useOld:
        R_fit, h_fit = calculateTransform(groundTruth, distorted)

        A = np.block([[R_fit, h_fit],
                    [np.zeros((1, 2)), 1]])
        A_inv = np.linalg.inv(A)
        R_fit = A_inv[:2, :2]
        h_fit = A_inv[:2, 2:]
    else:
        R_fit, h_fit = calculateTransformSVD(groundTruth, distorted)
    #R_fit, h_fit = calculateTransformDxDth(srcCoord, currentFrame)

    theta_fit = np.arctan2(R_fit[1, 0], R_fit[0, 0]) * 180 / np.pi
    print(f"Actual Transform:\ntheta:\n{theta_deg}\nh:\n{h}")
    print(f"Fitted Transform:\ntheta:\n{theta_fit}\nh:\n{h_fit}")

    # Visualize
    plt.subplot(1,2,1)
    srcCoord2 = (R_fit @ distorted.T + h_fit).T    
    plotFakeFeatures(groundTruth, srcCoord2, 
                     title_append="", alpha=0.5, clear=False, show=False, 
                     plotDisplace = True)

    '''
    Applying Motion Distortion Solving
    '''
    # Prior Transform
    T_wj0 = np.eye(3)
    # Point world positions
    p_w = groundTruth
    # Observed points at time 1
    p_jt = distorted
    # Initial velocity guess
    v_j0 = np.array([h_fit[0,0], h_fit[1,0], theta_fit * np.pi / 180]) / period
    # Initial Transform guess
    T_wj = np.block([[R_fit,            h_fit],
                     [np.zeros((2,)),    1]])
    # Covariance matrix, point errors
    cov_p = np.diag([4, 4])
    # Covariance matrix, velocity errors
    cov_v = np.diag([1, 1, (5 * np.pi / 180) ** 2]) # 4 ^2 since 4 Hz
    # Information matrix, 
    MDS = MotionDistortionSolver(T_wj0, p_w, p_jt, v_j0, T_wj, cov_p, cov_v)
    MDS.compute_time_deltas(p_jt)
    undistorted = MDS.undistort(v_j0)
    print(undistorted.shape)
    undistorted = undistorted[:, :2, 0]
    R_fit, h_fit = calculateTransformSVD(groundTruth, undistorted)
    srcCoord3 = (R_fit @ undistorted.T + h_fit).T 
    plt.subplot(1,2,2)
    plotFakeFeatures(groundTruth, srcCoord3, 
                     title_append="", alpha=0.5, clear=False, show=True, 
                     plotDisplace = True)


    params = MDS.optimize_library()
    print(f"Parameters:\nvx, vy, dx, dy, dtheta\n{params.flatten()}")

    
    