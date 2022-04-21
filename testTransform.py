from getTransformKLT import calculateTransformSVD, calculateTransform
from genFakeData import *

if __name__ == "__main__":
    N = 100
    outlier_rate = 0.4
    noisy = False

    # Generate fake data
    srcCoord, targetCoord, theta_deg, h = generateFakeCorrespondences(n_points=N)
    if noisy:
        targetCoord, outlier_ind = createOutliers(targetCoord, int(N * outlier_rate), 
                                    noiseToAdd=10)
    
    # Fit a rotation and translation
    R_fit, h_fit = calculateTransform(srcCoord, targetCoord)

    A = np.block([[R_fit, h_fit],
                  [np.zeros((1, 2)), 1]])
    A_inv = np.linalg.inv(A)
    R_fit = A_inv[:2, :2]
    h_fit = A_inv[:2, 2:]
    theta_fit = np.arctan2(R_fit[1, 0], R_fit[0, 0]) * 180 / np.pi
    print(f"Actual Transform:\ntheta:\n{theta_deg}\nh:\n{h}")
    print(f"Fitted Transform:\ntheta:\n{theta_fit}\nh:\n{h_fit}")

    # Visualize
    targetCoord2 = (R_fit @ srcCoord.T + h_fit).T
    plotFakeFeatures(srcCoord, targetCoord, targetCoord2,
                     title_append="", alpha=0.5, clear=False, show=True)
    