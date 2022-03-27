import numpy as np
import cv2
import os, sys


def convertPolarToCartesian(imgPolar):
    w, h = imgPolar.shape

    maxRadius = w
    cartSize = (maxRadius * 2, maxRadius * 2)
    center = np.array(cartSize) / 2

    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    imgCart = cv2.warpPolar(imgPolar, cartSize, center, maxRadius, flags)

    return imgCart


def getRadarStream(dataPath, timestampPath):
    '''
    @brief Returns np array of radar images in Cartesian format
    @param[in] dataPath Path to radar image data
    @return (800 x 800 x N)
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

    for i, imgPath in enumerate(timestampPathArr):
        imgPolar = cv2.imread(imgPath, cv2.COLOR_BGR2GRAY)

        imgCart = convertPolarToCartesian(imgPolar)

        # Generate pre-cached np array of streams, to save memory
        if streamArr is None:
            fullShape = imgCart.shape + (NImgs, )
            streamArr = np.empty(fullShape, dtype=imgCart.dtype)

        # Save converted image into stream
        streamArr[:, :, i] = imgCart

    return streamArr


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("./data", datasetName, "radar")
    timestampPath = os.path.join("./data", datasetName, "radar.timestamps")

    streamArr = getRadarStream(dataPath, timestampPath)
    nImgs = streamArr.shape[2]

    for i in range(nImgs):
        imgCart = streamArr[:, :, i]

        try:
            cv2.imshow("Cart", imgCart)
            c = cv2.waitKey(100)
        except KeyboardInterrupt:
            break

        if c == ord('q'):
            break

    cv2.destroyAllWindows()