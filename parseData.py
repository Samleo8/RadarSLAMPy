# import g2o
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


def getRadarStream(dataPath):
    for imgName in os.listdir(dataPath):
        imgPath = os.path.join(dataPath, imgName)
        imgPolar = cv2.imread(imgPath, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Polar", imgPolar)
        # cv2.waitKey(0)

        imgCart = convertPolarToCartesian(imgPolar)

        cv2.imshow("Cart", imgCart)
        c = cv2.waitKey(0)

        if c == ord('q'):
            return

    return


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("./data", datasetName, "radar")

    getRadarStream(dataPath)

    cv2.destroyAllWindows()