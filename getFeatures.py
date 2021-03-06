from genericpath import exists
from select import select
import numpy as np
import cv2
import os, sys

from skimage.feature import blob_doh, blob_dog, blob_log
from Coord import CartCoord, PolarCoord
from parseData import getRadarStreamPolar, convertPolarImageToCartesian
from ANMS import ssc

# TODO: What are the values for num, min and max
DEFAULT_FEATURE_PARAMS = dict(
    min_sigma=0.01,
    max_sigma=10,
    num_sigma=3,
    threshold=.0005,  # lower threshold for more features
    method="doh")
# Needs more blobs


def getBlobsFromCart(cartImage: np.ndarray,
                     min_sigma: int = 1,
                     max_sigma: int = 30,
                     num_sigma: int = 10,
                     threshold=0.01,
                     method="doh") -> np.ndarray:
    '''
    @brief Given a radar image, generate a list of (K x 3)
           blob indices based on Determinant of Hessian
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

    blobs = blob_fn(cartImage.astype(np.double),
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold)

    return blobs

# Thresholds for feature loss
PERCENT_FEATURE_LOSS_THRESHOLD = 0.75
N_FEATURES_BEFORE_RETRACK = 60  # TODO: Make it dynamic (find the overall loss)


# TODO: Make dynamic?
def calculateFeatureLossThreshold(nInitialFeatures):
    global N_FEATURES_BEFORE_RETRACK
    return 80
    # return PERCENT_FEATURE_LOSS_THRESHOLD * nInitialFeatures

def adaptiveNMS(img, blobs, ret_points = 200, tolerance = 0.1):
    # print(blobs.shape)
    H, W = img.shape
    sort_ind = np.argsort(blobs[:, 2])
    keypoints = blobs[sort_ind, :]
    selected_keypoints = ssc(keypoints, ret_points, tolerance, W, H)
    return selected_keypoints

def getFeatures(img, feature_params: dict = DEFAULT_FEATURE_PARAMS):
    '''
    @brief Get features from image using Hessian blob detector
    @param[in] img Image to detect features from 
    @param[in] feature_params Parameters for feature detection, @see DEFAULT_FEATURE_PARAMS

    @return blobCoord (K x 2) array of [x, y] coordinates of center of blobs on the image
    @return blobRadii (K x 1) array of radius of blobs
    '''

    blobs = getBlobsFromCart(img, **feature_params)

    blobs = adaptiveNMS(img, blobs)

    # split up blobs information
    # only get the [r,c] coordinates thne convert to [x,y] because opencv
    blobCoord = np.fliplr(blobs[:, :2])

    # radii, TODO: possible use for window size?
    blobRadii = blobs[:, 2]

    return blobCoord, blobRadii


def appendNewFeatures(srcImg, oldFeaturesCoord):
    '''
    @brief Append new features obtained from srcImg onto oldFeaturesCoord array
    @see getFeatures()

    @param[in] srcImg Source image to obtain features on
    @param[in] oldFeaturesCoord (K x 2) array of [x, y] coordinate of features
    '''
    newFeatureCoord, newFeatureRadii = getFeatures(srcImg)
    print("Added", newFeatureCoord.shape[0], "new features!")

    featurePtSrc = np.vstack((oldFeaturesCoord, newFeatureCoord))
    # NOTE: Also remove duplicate features, will sort the array
    _, idx = np.unique(featurePtSrc, axis=0, return_index=True)
    featurePtSrc = np.ascontiguousarray(featurePtSrc[np.sort(idx)]).astype(np.float32)

    # TODO: Recalculate threshold for feature retracking?
    nFeatures = featurePtSrc.shape[0]
    N_FEATURES_BEFORE_RETRACK = calculateFeatureLossThreshold(nFeatures)

    return featurePtSrc, N_FEATURES_BEFORE_RETRACK


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

        s_blobIndices = adaptiveNMS(imgCart, blobIndices)

        # Display with radii?
        imgCartBGR = cv2.cvtColor(imgCart, cv2.COLOR_GRAY2BGR) * 255
        imgCartBGR = imgCartBGR.astype(np.uint8)

        nIndices = blobIndices.shape[0]
        nIndicesANMS = s_blobIndices.shape[0]

        print(imgNo, "| Blobs detected:", nIndices)
        print(imgNo, "| ANMS Blobs detected:", nIndicesANMS)

        for i in range(nIndicesANMS):
            blobY, blobX, blobSigma = \
                int(s_blobIndices[i, 0]), int(s_blobIndices[i, 1]), int(s_blobIndices[i, 2])
            coord = (blobX, blobY)
            color = (0, 255, 0)
            imgCartBGR = cv2.circle(imgCartBGR,
                                    coord,
                                    radius=blobSigma,
                                    color=color,
                                    thickness=3)

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
