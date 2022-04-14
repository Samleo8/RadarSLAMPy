import numpy as np
import os, sys
from utils import tic, toc

from getFeatures import getBlobsFromCart

from parseData import getCartImageFromImgPaths, getRadarImgPaths
from trajectoryPlotting import *
from KLT import *

datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
dataPath = os.path.join("data", datasetName, "radar")
timestampPath = os.path.join("data", datasetName, "radar.timestamps")

# Incremental streaming
imgPathArr = getRadarImgPaths(dataPath, timestampPath)
nImgs = len(imgPathArr)

initTimestamp = radarImgPathToTimestamp(imgPathArr[0])
traj = Trajectory([initTimestamp], [[0,0,0]])
traj.plotTrajectory()

for imgNo in range(nImgs):
    currImg = getCartImageFromImgPaths(imgPathArr, imgNo)
    cv2.imshow("radar", currImg)

    blobIndices = getBlobsFromCart(currImg,
                                    min_sigma=0.01,
                                    max_sigma=10,
                                    num_sigma=3,
                                    threshold=.0005,
                                    method="doh")
    currFeatureInd = blobIndices[:, :2].astype(int)

    if imgNo:
        print(f"Computing affine transforms for {len(blobIndices)} blobs", end="... ", flush=True)
        start = tic()
        A, h = KLT(prevFeatureInd,
                    currFeatureInd,
                    currImg.shape,
                    cloud=True,
                    max_iters=20,
                    visual=False)
        print(f"Done in {toc(start):.5f} seconds.")
        timestamp = radarImgPathToTimestamp(imgPathArr[imgNo])
        traj.appendRelativePose(timestamp, A, h)
        traj.plotTrajectory()

    prevImg = np.copy(currImg)
    prevFeatureInd = np.copy(currFeatureInd)

print('Done.')
plt.show(block=True)