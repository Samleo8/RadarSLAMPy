from typing import Tuple
import numpy as np
import cv2
import os, sys
import csv

from Coord import CartCoord

RANGE_RESOLUTION_M = 0.0432  # radar range resolution in m (4.32 cm)


def extractDataFromRadarImage(
    polarImgData: np.ndarray
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    @brief Decode a single Oxford Radar RobotCar Dataset radar example
    @param[in] polarImgData cv image
    @return
        fft_data (np.ndarray): Radar power readings along each azimuth
        range_resolution (float): Range resolution of the polar radar data (metres per pixel)
        azimuth_resolution (float): Azimuth resolution of the polar radar data (radians per pixel)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
    """
    # Hard coded configuration to simplify parsing code
    range_resolution = RANGE_RESOLUTION_M
    encoder_size = 5600

    # Extract actual data and metadata from the image
    timestamps = polarImgData[:, :8].copy().view(np.int64)
    azimuths = (polarImgData[:, 8:10].copy().view(np.uint16) /
                float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = polarImgData[:, 10:11] == 255
    range_azimuth_data = polarImgData[:, 11:].astype(np.float32) / 255.

    azimuth_resolution = azimuths[1] - azimuths[0]

    return range_azimuth_data, azimuths, range_resolution, azimuth_resolution, valid, timestamps


def drawCVPoint(img: np.ndarray,
                point: CartCoord,
                point_color: tuple[int, int, int] = (0, 0, 255)):
    if isinstance(point, CartCoord):
        point = point.asTuple()

    return cv2.circle(img,
                      tuple(point),
                      radius=0,
                      color=point_color,
                      thickness=-1)


def convertPolarImageToCartesian(imgPolar: np.ndarray) -> np.ndarray:
    w, h = imgPolar.shape

    maxRadius = w
    cartSize = (maxRadius * 2, maxRadius * 2)
    center = tuple(np.array(cartSize) / 2)

    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    imgCart = cv2.warpPolar(imgPolar, cartSize, center, maxRadius, flags)

    return imgCart


def getRadarStreamPolar(dataPath: str, timestampPath: str) -> np.ndarray:
    '''
    @brief Returns np array of radar images in POLAR format
    @param[in] dataPath Path to radar image data
    @return radar range-azimuth image (W x H x N)
    '''
    streamArr = None

    timestampPathArr = []
    with open(timestampPath, "r") as f:
        lines = f.readlines()
        for line in lines:
            stamp, valid = line.strip().split(" ")
            if valid:
                stampPath = os.path.join(dataPath, stamp + ".png")
                timestampPathArr.append(stampPath)

    NImgs = len(timestampPathArr)

    for i in range(NImgs):
        imgPath = timestampPathArr[i]
        imgPolarData = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        imgPolar, azimuths, range_resolution, azimuth_resolution, valid, timestamps = \
            extractDataFromRadarImage(imgPolarData)

        # Generate pre-cached np array of streams, to save memory
        if streamArr is None:
            print("Range Resolution:", range_resolution, "[m]")
            print("Azimuth Resolution:", azimuth_resolution, "[rad]",
                  np.rad2deg(azimuth_resolution), "[deg]")

            fullShape = imgPolar.shape + (NImgs, )
            streamArr = np.empty(fullShape, dtype=imgPolar.dtype)

        # Save converted image into stream
        streamArr[:, :, i] = imgPolar

    return streamArr


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    streamArr = getRadarStreamPolar(dataPath, timestampPath)

    nImgs = streamArr.shape[2]

    for i in range(nImgs):
        imgPolar = streamArr[:, :, i]
        imgCart = convertPolarImageToCartesian(imgPolar)

        try:
            cv2.imshow("Cartesian Stream", imgCart)
            c = cv2.waitKey(100)
        except KeyboardInterrupt:
            break

        if c == ord('q'):
            break

    cv2.destroyAllWindows()