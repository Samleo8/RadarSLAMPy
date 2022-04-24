from genericpath import exists
import shutil
from urllib import response
from matplotlib import scale
import numpy as np
import cv2
import matplotlib.pyplot as plt

from parseData import RANGE_RESOLUTION_M, convertCartesianImageToPolar, convertPolarImageToCartesian, getCartImageFromImgPaths, getPolarImageFromImgPaths, getPolarImageFromImgPaths, getRadarImgPaths, convertPolarImgToLogPolar
from utils import normalize_angles


def getRotationUsingFMT(srcPolarImg: np.ndarray,
                        targetPolarImg: np.ndarray,
                        downsampleFactor: int = 10) -> float:
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
    '''
    assert srcPolarImg.shape == targetPolarImg.shape, "Images need to have the same shape!"

    H, W = srcPolarImg.shape

    resizeSize = (int(W // downsampleFactor), H)
    srcPolarImgDownsampled = cv2.resize(srcPolarImg, resizeSize)
    targetPolarImgDownsampled = cv2.resize(targetPolarImg, resizeSize)

    # Convert to log polar form
    # TODO: Check if this is a problem
    srcLogPolar = convertPolarImgToLogPolar(srcPolarImgDownsampled)
    targetLogPolar = convertPolarImgToLogPolar(targetPolarImgDownsampled)

    hanningSize = (srcLogPolar.shape[1], srcLogPolar.shape[0])
    hanningWindow = cv2.createHanningWindow(hanningSize, cv2.CV_32F)
    deltas, response = cv2.phaseCorrelate(srcLogPolar, targetLogPolar, hanningWindow)

    # Angle
    scale, angle = deltas
    print(deltas, response)

    angle = -(float(angle) * 2 * np.pi) / srcLogPolar.shape[0]
    angle = normalize_angles(angle)

    # TODO: Unsure where the log_base is
    log_base = np.e
    scale = log_base ** scale
    # scale = np.exp(scale)

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
    COLS = 3 if rotRad is not None else 2
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
        plt.imshow(rotateImg(prevImgCart, rotDeg))
        plt.title(f"Prev (Rot by {rotDeg:.1f} deg)")

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
    prevCartImg = convertPolarImageToCartesian(prevPolarImg, downsampleFactor=1)

    # TODO: manually rotate then get rotation
    
    for deg in np.arange(-30, 30, 0.2):
        rotCartImg = rotateImg(prevCartImg, deg)
        # rotCartImg = getCartImageFromImgPaths(imgPathArr, startSeqInd + 5)
        rotPolarImg = convertCartesianImageToPolar(rotCartImg,
                                                shapeHW=prevPolarImg.shape)

        plotCartPolar(prevCartImg, rotCartImg, prevPolarImg, rotPolarImg)
        # plt.show()

        rot, scale, response = getRotationUsingFMT(prevPolarImg, rotPolarImg)

        print(f"Pred: {np.rad2deg(rot):.2f} deg | Actual: {deg} deg")
        print(f"Scale Factor: {scale:.2f}")
    exit()
    # '''

    # prevImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd)
    # prevImgCart = convertPolarImageToCartesian(prevImgPolar, downsampleFactor=4)

    # currImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd + 5)
    # currImgCart = convertPolarImageToCartesian(currImgPolar,
    #                                            downsampleFactor=4)

    # rot, scale, response = getRotationUsingFMT(prevImgPolar, currImgPolar)
    # print(f"Pred: {np.rad2deg(rot):.2f} [deg] {rot:.2f} [radians]")

    # plotCartPolarWithRotation(prevImgCart, currImgCart, rot)
    # plt.show()

    # exit()

    # prevImgCart = getCartImageFromImgPaths(imgPathArr, startSeqInd)
    prevImgPolar = getPolarImageFromImgPaths(imgPathArr, startSeqInd)
    prevImgCart = convertPolarImageToCartesian(prevImgPolar,
                                               downsampleFactor=10)

    imgSavePath = os.path.join(".", "img", "fmt", sequenceName).strip(os.path.sep)
    os.makedirs(imgSavePath, exist_ok=True)

    for seqInd in range(startSeqInd + 1, endSeqInd + 1):
        try:
            # Obtain image
            # currImgCart = getCartImageFromImgPaths(imgPathArr, seqInd)
            currImgPolar = getPolarImageFromImgPaths(imgPathArr, seqInd)
            currImgCart = convertPolarImageToCartesian(currImgPolar, downsampleFactor=10)

            rotRad, scale, response = getRotationUsingFMT(prevImgPolar, currImgPolar)

            # print(f"===========Seq {seqInd}=========")
            print(f"Pred: {np.rad2deg(rotRad):.2f} [deg] {rotRad:.2f} [radians]")
            print(f"Scale Factor: {scale:.2f}, Response {response:.2f}")

            # Save image
            imgSavePathInd = os.path.join(imgSavePath, f"{seqInd:04d}.jpg")
            plotCartPolarWithRotation(prevImgCart, currImgCart, rotRad)
            
            plt.suptitle(f"Sequence {seqInd:04d}")
            plt.tight_layout()
            
            plt.savefig(imgSavePathInd)

            prevImgPolar = currImgPolar
            prevImgCart = currImgCart
        except KeyboardInterrupt:
            break

    # Generate mp4 and save that
    REMOVE_OLD_RESULTS = True
    
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
