from genericpath import exists
import numpy as np
import cv2
import os, sys

from skimage.feature import blob_doh, blob_dog, blob_log
from Coord import CartCoord, PolarCoord
from parseData import getRadarStreamPolar, convertPolarImageToCartesian


def getBlobsFromCart(cartImage: np.ndarray,
                     min_sigma: int = 1,
                     max_sigma: int = 30,
                     num_sigma: int = 10,
                     threshold=0.01,
                     method="doh") -> np.ndarray:
    '''
    @brief Given a radar image, generate a list of blob indices 
           based on Determinant of Hessian
    @note Uses default params from skimage.features function

    @param[in] cartImage Cartesian radar image
    
    @return (K x 3) Np array of blob coordinates with each row [r, c, sigma] being coordinates and sigma of detected blobs
    '''
    M, N = cartImage.shape

    blob_fns = {"doh": blob_doh, "dog": blob_dog, "log": blob_log}

    blob_fn = blob_fns.get(method, None)

    if blob_fn is None:
        raise NotImplementedError(
            f"{method} not implemented! Use one of {blob_fns.keys()}")

    blobs = blob_fn(cartImage,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold)

    return blobs


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    print("Parsing stream", end="...", flush=True)
    streamArr = getRadarStreamPolar(dataPath, timestampPath)
    nImgs = streamArr.shape[2]
    print("Complete!")

    # Img saving
    toSavePath = os.path.join(".", "img", "blob", datasetName)
    os.makedirs(toSavePath, exist_ok=True)

    for imgNo in range(nImgs):
        imgPolar = streamArr[:, :, imgNo]

        # TODO: What are the values for num, min and max sigma
        imgCart = convertPolarImageToCartesian(imgPolar)
        blobIndices = getBlobsFromCart(imgCart,
                                       min_sigma=0.01,
                                       max_sigma=10,
                                       num_sigma=3,
                                       threshold=.0005,
                                       method="doh")

        # Display with radii?
        imgCartBGR = cv2.cvtColor(imgCart, cv2.COLOR_GRAY2BGR) * 255
        imgCartBGR = imgCartBGR.astype(np.uint8)

        nIndices = blobIndices.shape[0]
        print(imgNo, "| Blobs detected:", nIndices)

        for i in range(nIndices):
            blobY, blobX, blobSigma = \
                int(blobIndices[i, 0]), int(blobIndices[i, 1]), int(blobIndices[i, 2])

            coord = (blobX, blobY)
            color = (0, 0, 255)
            imgCartBGR = cv2.circle(imgCartBGR,
                                    coord,
                                    radius=blobSigma,
                                    color=color,
                                    thickness=1)

        try:
            # Save blob images
            toSaveImgPath = os.path.join(toSavePath, f"{imgNo:04d}.jpg")
            cv2.imwrite(toSaveImgPath, imgCartBGR)

            # cv2.imshow("Cartesian Stream with Blob Features", imgCartBGR)
            # c = cv2.waitKey(10)

            # if c == ord('q'):
            #     break
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()

    # Generate mp4 and save that
    print("Generating mp4 with script (requires bash and FFMPEG command)...")
    try:
        os.system(f"./img/mp4-from-folder.sh {toSavePath}")
        print(f"mp4 added to {toSavePath} folder!")
    except:
        print(
            "Failed to generate mp4 with script. Likely failed system requirements."
        )
