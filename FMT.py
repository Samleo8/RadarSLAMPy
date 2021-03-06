import shutil
from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt

from parseData import RANGE_RESOLUTION_CART_M, convertCartesianImageToPolar, convertPolarImageToCartesian, getCartImageFromImgPaths, getPolarImageFromImgPaths, getPolarImageFromImgPaths, getRadarImgPaths, convertPolarImgToLogPolar
from utils import normalize_angles

FMT_DOWNSAMPLE_FACTOR = 10  # default downsampling factor for FMT rotation
FMT_RANGE_CLIP_M = 87.5 # range clipping in m

def getTranslationUsingPhaseCorrelation(
        srcImg: np.ndarray,
        targetImg: np.ndarray) -> tuple[tuple[float, float], float]:
    '''
    @brief Using phase correlation, find the translation delta between 2 images
    @param[in] srcImg Source image
    @param[in] targetImg Target image

    @return deltas Delta sub-pixel translation
    @return response Response of phase correlation window (indicating confidence)
    '''

    H, W = srcImg.shape
    hanningSize = (W, H)

    # TODO: Cache the hanning window
    hanningWindow = cv2.createHanningWindow(hanningSize, cv2.CV_32F)
    deltas, response = cv2.phaseCorrelate(srcImg, targetImg,
                                          hanningWindow)

    return deltas, response


def getRotationUsingFMT(srcPolarImg: np.ndarray,
                        targetPolarImg: np.ndarray,
                        downsampleFactor: int = FMT_DOWNSAMPLE_FACTOR,
                        maxRangeClipM=FMT_RANGE_CLIP_M) -> tuple[float, float, float]:
    '''
    @brief Get rotation using the Fourier-Mellin Transform
    @note We attempt to downsample in the range direction. 
          Since we are already in the polar domain, we just need to convert to a logpolar image
    apply phase correlation to get the rotation (which is a "\Delta Y" translation)

    @param[in] srcPolarImg Source image in polar (not log-polar) form
    @param[in] targetPolarImg Target image in polar (not log-polar) form
    @param[in] How much to further downsample in 

    @return angleRad Angle in radians, where `R(angleRad) @ src = target`
    @return scaling Scaling factor
    @return response Response value (indicates confidence)
    '''
    assert srcPolarImg.shape == targetPolarImg.shape, "Images need to have the same shape!"

    # Clip range if needed
    if maxRangeClipM > 0:
        maxRangeClipPx = int(maxRangeClipM / RANGE_RESOLUTION_CART_M)
        srcPolarImg = srcPolarImg[:, :maxRangeClipPx]
        targetPolarImg = targetPolarImg[:, :maxRangeClipPx]

    H, W = srcPolarImg.shape

    resizeSize = (int(W // downsampleFactor), H)
    srcPolarImgDownsampled = cv2.resize(srcPolarImg, resizeSize)
    targetPolarImgDownsampled = cv2.resize(targetPolarImg, resizeSize)

    # Convert to log polar form
    srcLogPolar = convertPolarImgToLogPolar(srcPolarImgDownsampled)
    targetLogPolar = convertPolarImgToLogPolar(targetPolarImgDownsampled)

    deltas, response = getTranslationUsingPhaseCorrelation(
        srcLogPolar, targetLogPolar)

    # Obtain the transforms/deltas
    scale, angle = deltas

    # Formulas for calculation of correct FMT
    # @see https://sthoduka.github.io/imreg_fmt/docs/fourier-mellin-transform/
    H_lp, W_lp = srcLogPolar.shape
    sz = max(H_lp, W_lp)

    angle = -float(angle) * 2 * np.pi / sz
    angle = normalize_angles(angle)

    # Calculate log_base
    log_base = np.exp(np.log(H_lp / 2) / sz)
    scale = log_base**scale

    return angle, scale, response


def rotateImg(image, angle_degrees):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_degrees, 1.0)
    result = cv2.warpAffine(image,
                            rot_mat,
                            image.shape[1::-1],
                            flags=cv2.INTER_LINEAR)
    return result


def plotCartPolar(prevImgPolar, currImgPolar, prevImgCart, currImgCart):
    ROWS = 2
    COLS = 2
    i = 0

    i += 1
    plt.subplot(ROWS, COLS, i)
    if prevImgPolar is not None:
        plt.imshow(prevImgPolar)
        plt.title("Prev Image Polar")

    i += 1
    plt.subplot(ROWS, COLS, i)

    if currImgPolar is not None:
        plt.imshow(currImgPolar)
        plt.title("Curr Image Polar")

    i += 1
    plt.subplot(ROWS, COLS, i)
    if prevImgCart is not None:
        plt.imshow(prevImgCart)
        plt.title("Prev Image Cartesian")

    i += 1
    plt.subplot(ROWS, COLS, i)
    if currImgCart is not None:
        plt.imshow(currImgCart)
        plt.title("Curr Image Cartesian")


def plotCartPolarWithRotation(prevImgCart, currImgCart, rotRad):
    ROWS = 1
    COLS = 4 if rotRad is not None else 2
    i = 0

    i += 1
    plt.subplot(ROWS, COLS, i)
    if prevImgCart is not None:
        plt.imshow(prevImgCart)
        plt.axis("off")
        plt.title("Prev Image")

    i += 1
    plt.subplot(ROWS, COLS, i)
    if currImgCart is not None:
        plt.imshow(currImgCart)
        plt.axis("off")
        plt.title("Curr Image")

    if rotRad is not None:
        i += 1
        plt.subplot(ROWS, COLS, i)
        plt.axis("off")
        rotDeg = np.rad2deg(rotRad)
        rotatedImg = rotateImg(prevImgCart, rotDeg)
        plt.imshow(rotatedImg)
        plt.title(f"Prev, Rotation\nCorrected by {rotDeg:.1f} deg")

        i += 1
        plt.subplot(ROWS, COLS, i)
        plt.axis("off")
        plt.imshow(rotatedImg - currImgCart, cmap='coolwarm')
        plt.title(f"Overlay")

    plt.tight_layout()


if __name__ == "__main__":
    import os
    import sys

    sequenceName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", sequenceName, "radar")
    timestampPath = os.path.join("data", sequenceName, "radar.timestamps")

    startSeqInd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    endSeqInd = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    # Get initial Polar image
    imgPathArr = getRadarImgPaths(dataPath, timestampPath)
    sequenceSize = len(imgPathArr)

    if endSeqInd < 0:
        endSeqInd = sequenceSize - 1
    '''
    # Perfect Image test
    prevPolarImg = getPolarImageFromImgPaths(imgPathArr, startSeqInd)
    prevCartImg = convertPolarImageToCartesian(prevPolarImg, downsampleFactor=20)

    # TODO: manually rotate then get rotation
    
    for deg in np.arange(-30, 30, 0.2):
        rotCartImg = rotateImg(prevCartImg, deg)
        # rotCartImg = getCartImageFromImgPaths(imgPathArr, startSeqInd + 5)
        rotPolarImg = convertCartesianImageToPolar(rotCartImg,
                                                shapeHW=prevPolarImg.shape)

        plotCartPolar(prevCartImg, rotCartImg, prevPolarImg, rotPolarImg)
        # plt.show()

        rot, scale, response = getRotationUsingFMT(prevPolarImg, rotPolarImg)

        print(f"Pred: {np.rad2deg(rot):.2f} deg | Actual: {deg:.2f} deg")
        print(f"Scale Factor: {scale:.2f}")
    exit()
    # '''

    # prevImgCart = getCartImageFromImgPaths(imgPathArr, startSeqInd)
    prevImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd)
    prevImgCart = convertPolarImageToCartesian(prevImgPolar,
                                               downsampleFactor=20)

    imgSavePath = os.path.join(".", "img", "fmt",
                               sequenceName).strip(os.path.sep)
    os.makedirs(imgSavePath, exist_ok=True)

    stepSize = 1
    for seqInd in range(startSeqInd + stepSize, endSeqInd + 1, stepSize):
        try:
            # Obtain image
            # currImgCart = getCartImageFromImgPaths(imgPathArr, seqInd)
            currImgPolar = getPolarImageFromImgPaths(imgPathArr, seqInd)
            currImgCart = convertPolarImageToCartesian(currImgPolar,
                                                       downsampleFactor=20)

            rotRad, scale, response = getRotationUsingFMT(
                prevImgPolar, currImgPolar)

            # print(f"===========Seq {seqInd}=========")
            print(
                f"Pred: {np.rad2deg(rotRad):.2f} [deg] {rotRad:.2f} [radians]")
            print(f"Scale Factor: {scale:.2f}, Response {response:.2f}")

            # Save image
            plt.clf()

            imgSavePathInd = os.path.join(imgSavePath, f"{seqInd:04d}_5.jpg")
            plotCartPolarWithRotation(prevImgCart, currImgCart, rotRad)

            plt.suptitle(f"Sequence {seqInd:04d}")
            plt.tight_layout()

            plt.savefig(imgSavePathInd)
            # plt.pause(0.01)

            prevImgPolar = currImgPolar
            prevImgCart = currImgCart
        except KeyboardInterrupt:
            break

    # exit()

    # Generate mp4 and save that
    REMOVE_OLD_RESULTS = False

    # Also remove folder of images to save space
    print("Generating mp4 with script (requires bash and FFMPEG command)...")
    try:
        # Save video sequence
        os.system(f"./img/mp4-from-folder.sh {imgSavePath} {startSeqInd + 1}")
        print(f"mp4 saved to {imgSavePath.strip(os.path.sep)}.mp4")

        if REMOVE_OLD_RESULTS:
            shutil.rmtree(imgSavePath)
            print("Old results folder removed.")
    except:
        print(
            "Failed to generate mp4 with script. Likely failed system requirements."
        )
