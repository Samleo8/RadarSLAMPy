from getTransformKLT import *
from genFakeData import *

if __name__ == "__main__":
    N = 100
    outlier_rate = 0.4
    noisy = False
    useOld = False
    '''
    You are personally moving in -theta_deg and -h (inverse)

    the points you see warp theta_deg and h in your coordinate frame

    You learn R, h such that applying them to the points you see gives you their 
    original pre-warp coordinates: this is a -theta_deg and -h transform: aka 
    your movement!
    '''
    # Generate fake data
    # srcCoord = frame 1
    # targetCoord = frame 2
    # R_theta_deg @ srcCoord + h = targetCoord
    # pose at frame 1 (srcCoord) is I
    srcCoord, targetCoord, theta_deg, h = generateFakeCorrespondences(n_points=N)
    #print(f"Original transform angle: {theta_deg}")
    if noisy:
        targetCoord, outlier_ind = createOutliers(targetCoord, int(N * outlier_rate), 
                                    noiseToAdd=10)
    
    # Fit a rotation and translation
    if useOld:
        R_fit, h_fit = calculateTransform(srcCoord, targetCoord)

        A = np.block([[R_fit, h_fit],
                    [np.zeros((1, 2)), 1]])
        A_inv = np.linalg.inv(A)
        R_fit = A_inv[:2, :2]
        h_fit = A_inv[:2, 2:]
    else:
        R_fit, h_fit = calculateTransformSVD(srcCoord, targetCoord)
    #R_fit, h_fit = calculateTransformDxDth(srcCoord, targetCoord)

    theta_fit = np.arctan2(R_fit[1, 0], R_fit[0, 0]) * 180 / np.pi
    print(f"Actual Transform:\ntheta:\n{theta_deg}\nh:\n{h}")
    print(f"Fitted Transform:\ntheta:\n{theta_fit}\nh:\n{h_fit}")

    # Visualize
    srcCoord2 = (R_fit @ targetCoord.T + h_fit).T
    plotFakeFeatures(srcCoord, targetCoord, srcCoord2,
                     title_append="", alpha=0.5, clear=False, show=True)
    