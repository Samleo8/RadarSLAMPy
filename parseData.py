# import g2o
import numpy as np
import cv2
import os, sys


def convertPolarToCartesian(imgPolar):
    size = imgPolar.shape
    center = tuple((np.array(imgPolar.shape) / 2))
    print(center)

    maxRadius = imgPolar.shape[0]
    imgCart = cv2.warpPolar(imgPolar, size, center, maxRadius,
                            cv2.WARP_POLAR_LOG + cv2.WARP_INVERSE_MAP)

    return imgCart


def getRadarStream(dataPath):
    for imgName in os.listdir(dataPath):
        imgPath = os.path.join(dataPath, imgName)
        imgPolar = cv2.imread(imgPath, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Polar", imgPolar)
        # cv2.waitKey(0)

        imgCart = convertPolarToCartesian(imgPolar)

        cv2.imshow("Cart", imgCart)
        cv2.waitKey(0)

    return


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("./data", datasetName, "radar")

    getRadarStream(dataPath)