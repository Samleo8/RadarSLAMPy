from typing import Tuple
import numpy as np
import cv2
import os, sys

from Coord import CartCoord

RANGE_RESOLUTION_M = 0.0432  # radar range resolution in m (4.32 cm)


def extractDataFromRadarImage(
    polarImgData: np.ndarray
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    @brief Decode a single Oxford Radar RobotCar Dataset radar example
    @param[in] polarImgData cv image
    @return
        range_azimuth_data (np.ndarray): Radar power readings along each azimuth
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
    '''
    @brief Converts polar image to Cartesian formats
    @param[in] imgPolar Polar image to convert
    @return imgCart Converted Cartesian image
    '''
    w, h = imgPolar.shape

    maxRadius = w
    cartSize = (maxRadius * 2, maxRadius * 2)
    center = tuple(np.array(cartSize) / 2)

    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    imgCart = cv2.warpPolar(imgPolar, cartSize, center, maxRadius, flags)

    return imgCart


def getDataFromImgPathsByIndex(
    imgPathArr: list[str], index: int
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
    '''
    @brief Get information from image path array, indexing accordingly
    @param[in] imgPathArr List of image path as strings
    @param[in] index Index to index into

    @return
        imgPolar (np.ndarray): Radar power readings along each azimuth
        azimuth_resolution (float): Azimuth resolution of the polar radar data (radians per pixel)
        range_resolution (float): Range resolution of the polar radar data (metres per pixel)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
    '''
    imgPath = imgPathArr[i]
    imgPolarData = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    imgPolar, azimuths, range_resolution, azimuth_resolution, valid, timestamps = \
        extractDataFromRadarImage(imgPolarData)


def getPolarImageFromImgPaths(imgPathArr: list[str], index: int) -> np.ndarray:
    '''
    @brief Get polar image from image path array, indexing accordingly
    @param[in] imgPathArr List of image path as strings
    @param[in] index Index to index into

    @return imgPolar Polar image
    '''

    imgPolar, _, _, _, _, _ = getDataFromImgPathsByIndex(imgPathArr, index)
    return imgPolar


def getCartImageFromImgPaths(imgPathArr: list[str], index: int) -> np.ndarray:
    '''
    @brief Get polar image from image path array, indexing accordingly
    @param[in] imgPathArr List of image path as strings
    @param[in] index Index to index into

    @return imgCart Cartesian image
    '''

    imgPolar = getPolarImageFromImgPaths(imgPathArr, index)
    return convertPolarImageToCartesian(imgPolar)


def getRadarImgPaths(dataPath: str, timestampPath: str) -> list[str]:
    '''
    @brief Obtain list of radar image paths
    
    @param[in] dataPath Path to radar image data
    @param[in] timestampPath Path to radar timestamp data

    @return list of strings containing paths to radar image
    '''
    imgPathArr = []
    with open(timestampPath, "r") as f:
        lines = f.readlines()
        for line in lines:
            stamp, valid = line.strip().split(" ")
            if valid:
                stampPath = os.path.join(dataPath, stamp + ".png")
                imgPathArr.append(stampPath)

    return imgPathArr


def getRadarStreamPolar(dataPath: str, timestampPath: str) -> np.ndarray:
    '''
    @brief Returns np array of radar images in POLAR format
    @param[in] dataPath Path to radar image data
    @param[in] timestampPath Path to radar timestamp data
    
    @return radar range-azimuth image (W x H x N)
    '''
    streamArr = None

    imgPathArray = getRadarImgPaths(dataPath, timestampPath)

    NImgs = len(imgPathArray)

    for i in range(NImgs):
        imgPolar, azimuths, range_resolution, azimuth_resolution, valid, timestamps = \
            getDataFromImgPathsByIndex(imgPathArray, i)

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