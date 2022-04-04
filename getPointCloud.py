from turtle import distance
import numpy as np
import cv2
import os, sys

from scipy.signal import find_peaks

from Coord import CartCoord, PolarCoord
from parseData import getRadarStreamPolar, convertPolarImageToCartesian


def getFeaturesPolarInd(polarImage: np.ndarray,
                        peakDistance: float = None,
                        peakProminence: float = None) -> np.ndarray:
    '''
    @brief Given a radar image, generate a list of polar indices 
           based on peak detection with pruning

    @param[in] peakDistance Minimum distance to be counted as a peak
    @param[in] peakProminence Minimum prominence to be counted as a peak

    @return (K x 2) Np array of polar coordinates with each row [thetaInd, rInd] being indices in the polar image
    '''
    M, N = polarImage.shape

    pointCloudPolarIndices = np.empty((0, 2))

    # For each azimuth value
    for azim_ind in range(M):
        # Obtain range readings
        azimuthReading = polarImage[azim_ind, :]

        # Obtain peak information
        peakInd, peakInfoDict = find_peaks(azimuthReading,
                                           distance=peakDistance,
                                           prominence=peakProminence)
        peakHeights = azimuthReading[peakInd]

        # Peak pruning algorithm
        # - first obtain mean and std dev information
        mean = np.mean(peakHeights)
        stddev = np.std(peakHeights)

        # - then obtain the threshold as a part of mean + stddev
        thresh = mean + stddev

        # - get valid peaks that are geq thresh according to algorithm
        validPeakInd = peakInd[peakHeights >= thresh]
        # validPeakHeights = azimuthReading[validPeakInd]
        azimuthIndices = np.full_like(validPeakInd, azim_ind)

        toAppend = np.vstack((azimuthIndices, validPeakInd)).T
        pointCloudPolarIndices = np.vstack((pointCloudPolarIndices, toAppend))

    return pointCloudPolarIndices.astype(int)


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    streamArr = getRadarStreamPolar(dataPath, timestampPath)
    nImgs = streamArr.shape[2]

    for i in range(nImgs):
        imgPolar = streamArr[:, :, i]

        # TODO: What are the values for peak prominence and distance
        featurePolarIndices = getFeaturesPolarInd(imgPolar)

        # TODO: need to convert from polar to Cartesian form?
        # TODO: for now display via weird way
        featurePolarImage = np.zeros_like(imgPolar)
        featureAzim, featureRange = featurePolarIndices[:, 0], featurePolarIndices[:, 1]
        featurePolarImage[featureAzim, featureRange] = 255

        # Display
        imgCart = convertPolarImageToCartesian(imgPolar)
        imgCartRGB = cv2.cvtColor(imgCart, cv2.COLOR_GRAY2BGR)

        featureImgCart = convertPolarImageToCartesian(featurePolarImage)
        imgCartRGB[:,:,2] = np.clip(featureImgCart + imgCartRGB[:,:,2], 0, 255)

        try:
            cv2.imshow("Cartesian Stream with Features", imgCartRGB)
            c = cv2.waitKey(100)
        except KeyboardInterrupt:
            break

        if c == ord('q'):
            break

    # cv2.destroyAllWindows()