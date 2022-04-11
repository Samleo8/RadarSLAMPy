import numpy as np
import cv2
import os, sys

from skimage.feature import blob_doh
from Coord import CartCoord, PolarCoord
from parseData import getRadarStreamPolar, convertPolarImageToCartesian


def getBlobsPolarInd(cartImage: np.ndarray,
                     min_sigma: int = 1,
                     max_sigma: int = 30,
                     num_sigma: int = 10,
                     threshold=0.01) -> np.ndarray:
    '''
    @brief Given a radar image, generate a list of blob indices 
           based on Determinant of Hessian
    @note Uses default params from skimage.features function

    @param[in] cartImage Cartesian radar image
    
    @return (K x 3) Np array of blob coordinates with each row [r, c, sigma] being coordinates and sigma of detected blobs
    '''
    M, N = cartImage.shape

    return blob_doh(cartImage,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold)


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    streamArr = getRadarStreamPolar(dataPath, timestampPath)
    nImgs = streamArr.shape[2]

    for i in range(nImgs):
        imgPolar = streamArr[:, :, i]

        # TODO: What are the values for num, min and max sigma
        imgCart = convertPolarImageToCartesian(imgPolar)
        blobIndices = getBlobsPolarInd(imgCart)

        # Display with radii?
        imgCartBGR = cv2.cvtColor(imgCart, cv2.COLOR_GRAY2BGR)
        nIndices = blobIndices.shape[0]
        print("Number of blobs detected", nIndices)
        for i in range(nIndices):
            blobY, blobX, blobSigma = \
                blobIndices[:, 0], blobIndices[:, 1], blobIndices[:, 2]

            coord = (blobX, blobY)
            color = (255, 0, 0)
            imgCartBGR = cv2.circle(imgCartBGR, coord, radius=blobSigma, color=color, thickness=1)

        try:
            cv2.imshow("Cartesian Stream with Blob Features", imgCartBGR)
            c = cv2.waitKey(0)
        except KeyboardInterrupt:
            break

        if c == ord('q'):
            break

    # cv2.destroyAllWindows()