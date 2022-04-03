import numpy as np
import cv2
import os, sys
import csv

from Coord import CartCoord


def drawCVPoint(img: np.ndarray, point: CartCoord,
                point_color: tuple[int, int, int] = (0, 0, 255)):
    if isinstance(point, CartCoord):
        point = point.asTuple()

    return cv2.circle(img,
                      tuple(point),
                      radius=0,
                      color=point_color,
                      thickness=-1)

def convertPolarImageToCartesian(imgPolar: np.ndarray) -> np.ndarray:
    w, h = imgPolar.shape

    maxRadius = w
    cartSize = (maxRadius * 2, maxRadius * 2)
    center = tuple(np.array(cartSize) / 2)

    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    imgCart = cv2.warpPolar(imgPolar, cartSize, center, maxRadius, flags)

    return imgCart


def getRadarStreamPolar(dataPath: str, timestampPath: str) -> np.ndarray:
    '''
    @brief Returns np array of radar images in POLAR format
    @param[in] dataPath Path to radar image data
    @return (W x H x N)
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

        # Generate pre-cached np array of streams, to save memory
        if streamArr is None:
            fullShape = imgPolar.shape + (NImgs, )
            streamArr = np.empty(fullShape, dtype=imgPolar.dtype)

        # Save converted image into stream
        streamArr[:, :, i] = imgPolar

    return streamArr


if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")
    gtPath = os.path.join("data", datasetName, "gt", "radar_odometry.csv")

    streamArr = getRadarStreamPolar(dataPath, timestampPath)
    nImgs = streamArr.shape[2]

    global_map = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.imshow("Polar", global_map)
    with open(gtPath) as gt_file:
        gt_reader = csv.reader(gt_file)
        headers = next(gt_file)

        ins_timestamps = [0]
        gt_timestamps = []
        gt_poses = []
        for row in gt_reader:
            timestamp = int(row[0]) # source_timestamp
            ins_timestamps.append(timestamp)
            x = float(row[2])
            y = float(row[3])
            gt_poses.append([x,y]) # x,y
            drawCVPoint(global_map, (int(x*100+100),int(y*100+100)), (0, 255, 0))
            cv2.imshow("Polar", global_map)
            cv2.waitKey(50)

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