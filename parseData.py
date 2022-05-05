from tkinter.messagebox import NO
from typing import Tuple, List
import numpy as np
import cv2
import os, sys

from Coord import CartCoord

RANGE_RESOLUTION_M = 0.0432  # radar range resolution in m (4.32 cm per pixel)
DOWNSAMPLE_FACTOR = 2

# NOTE: Should NOT * 2 because in the @see convertPolarImageToCartesian function, the Cartesian size is also doubled.
RANGE_RESOLUTION_CART_M = RANGE_RESOLUTION_M * DOWNSAMPLE_FACTOR
MAX_RANGE_CLIP_DEFAULT = 87.5  # according to the paper


def extractDataFromRadarImage(
    polarImgData: np.ndarray,
    maxRangeClipM: float = MAX_RANGE_CLIP_DEFAULT
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    @brief Decode a single Oxford Radar RobotCar Dataset radar example
    @param[in] polarImgData cv image
    @param[in] maxRangeClipM Max range to clip data, in meters. Negative number for no clip
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
    range_resolution = RANGE_RESOLUTION_M  # meters per pixel
    encoder_size = 5600

    # Extract actual data and metadata from the image
    timestamps = polarImgData[:, :8].copy().view(np.int64)
    azimuths = (polarImgData[:, 8:10].copy().view(np.uint16) /
                float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = polarImgData[:, 10:11] == 255
    range_azimuth_data = polarImgData[:, 11:].astype(np.float32) / 255.

    azimuth_resolution = azimuths[1] - azimuths[0]

    # Clip range if specified
    # Figure out range clip in m
    if maxRangeClipM > 0:
        maxRangeClipPx = int(maxRangeClipM / range_resolution)
        range_azimuth_data = range_azimuth_data[:, :maxRangeClipPx]

    return range_azimuth_data, azimuths, range_resolution, azimuth_resolution, valid, timestamps


def drawCVPoint(img: np.ndarray,
                point: CartCoord,
                point_color: Tuple[int, int, int] = (0, 0, 255)):
    if isinstance(point, CartCoord):
        point = point.asTuple()

    return cv2.circle(img,
                      tuple(point),
                      radius=0,
                      color=point_color,
                      thickness=-1)


def convertCartesianImageToPolar(
        imgCart: np.ndarray,
        logPolarMode: bool = False,
        shapeHW: Tuple[int, int] = None) -> np.ndarray:
    '''
    @brief Converts Cartesian image to (potentially log) polar
    @param[in] imgPolar Polar image to convert
    @param[in] logPolarMode Whether to convert in log-polar mode

    @return imgCart Converted Cartesian image
    '''
    h, w = imgCart.shape
    assert w == h, "Should be a square Cartesian image"

    center = (h / 2, w / 2)
    maxRadius = w / 2

    if shapeHW is None:
        size = None
    else: 
        size = (shapeHW[1], shapeHW[0])  # need to invert to make (W, H)

    flags = cv2.WARP_POLAR_LOG if logPolarMode else cv2.WARP_POLAR_LINEAR

    flags += cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS

    imgPolar = cv2.warpPolar(imgCart, size, center, maxRadius, flags)

    return imgPolar


def convertPolarImageToCartesian(
        imgPolar: np.ndarray,
        logPolarMode: bool = False,
        downsampleFactor: int = DOWNSAMPLE_FACTOR,
        changeGlobalRangeResolution: bool = False) -> np.ndarray:
    '''
    @brief Converts polar image to Cartesian formats
    @param[in] imgPolar Polar image to convert
    @param[in] logPolarMode Whether to convert in log-polar mode
    @param[in] downsampleFactor How much to downsample Cartesian image for performance improvements
    @param[in] changeGlobalRangeResolution Whether or not to change the
                                           global range resolution needed
                                           for accurate px to m conversions

    @return imgCart Converted Cartesian image
    '''
    w, h = imgPolar.shape

    if downsampleFactor > 1:
        maxRadius = h // downsampleFactor
    else:
        maxRadius = h

    cartSize = (maxRadius * 2, maxRadius * 2)
    center = tuple(np.array(cartSize) / 2)

    if changeGlobalRangeResolution:
        global RANGE_RESOLUTION_CART_M, RANGE_RESOLUTION_M
        RANGE_RESOLUTION_CART_M = RANGE_RESOLUTION_M * downsampleFactor

    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    if (logPolarMode):
        flags += cv2.WARP_POLAR_LOG

    imgCart = cv2.warpPolar(imgPolar, cartSize, center, maxRadius, flags)
    return imgCart


def convertPolarImgToLogPolar(imgPolar: np.ndarray):
    '''
    @brief Convert an image in polar form into log-polar form
    @note Involves converting from polar to Cartesian to back again
    @see convertPolarImageToCartesian(), convertCartesianImageToPolar()

    '''
    # Involves converting from polar to cartesian to back again
    # TODO: Probably a more efficent way to do this
    # Convert to Cartesian, do no downsample here
    imgCart = convertPolarImageToCartesian(imgPolar,
                                           logPolarMode=False,
                                           downsampleFactor=1,
                                           changeGlobalRangeResolution=False)
    # Convert the Cart image to log-polar
    logPolarImg = convertCartesianImageToPolar(imgCart,
                                               logPolarMode=True,
                                               shapeHW=None)

    return logPolarImg


def getDataFromImgPathsByIndex(
    imgPathArr: List[str], index: int
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
    imgPath = imgPathArr[index]
    imgPolarData = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    return extractDataFromRadarImage(imgPolarData)


def getPolarImageFromImgPaths(imgPathArr: List[str], index: int) -> np.ndarray:
    '''
    @brief Get polar image from image path array, indexing accordingly
    @param[in] imgPathArr List of image path as strings
    @param[in] index Index to index into

    @return imgPolar Polar image
    '''

    imgPolar, _, _, _, _, _ = getDataFromImgPathsByIndex(imgPathArr, index)
    return imgPolar


def getCartImageFromImgPaths(imgPathArr: List[str], index: int) -> np.ndarray:
    '''
    @brief Get polar image from image path array, indexing accordingly
    @param[in] imgPathArr List of image path as strings
    @param[in] index Index to index into

    @return imgCart Cartesian image
    '''

    imgPolar = getPolarImageFromImgPaths(imgPathArr, index)
    return convertPolarImageToCartesian(imgPolar)


def getRadarImgPaths(dataPath: str, timestampPath: str) -> List[str]:
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