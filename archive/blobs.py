import os
import sys
import numpy as np
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from parseData import getRadarStreamPolar, convertPolarImageToCartesian
from utils import tic, toc

# Testing plot blob
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html

datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
dataPath = os.path.join("data", datasetName, "radar")
timestampPath = os.path.join("data", datasetName, "radar.timestamps")

streamArr = getRadarStreamPolar(dataPath, timestampPath)
nImgs = streamArr.shape[2]

# for i in range(nImgs):
for i in range(1):
    imgPolar = streamArr[:, :, i]
    imgCart = convertPolarImageToCartesian(imgPolar).astype(np.double)
    print(imgCart.shape)
    # imgCartBW = rgb2gray(imgCart)

    start = tic()
    blobs_log = blob_log(imgCart, min_sigma = 0.5, max_sigma=10, num_sigma=3, threshold=.1, log_scale=True)
    print(f'log: {toc(start):.5f} seconds')

    start = tic()
    blobs_dog = blob_dog(imgCart, min_sigma = 0.5, max_sigma=10, threshold=.1)
    print(f'dog: {toc(start):.5f} seconds')

    start = tic()
    blobs_doh = blob_doh(imgCart, min_sigma = 0.01, max_sigma=10, threshold=.0005, num_sigma=3, log_scale=False)
    print(f'doh: {toc(start):.5f} seconds')
    
    blobs_log[:, 2] = blobs_log[:, 2]
    blobs_dog[:, 2] = blobs_dog[:, 2]
    blobs_doh[:, 2] = blobs_doh[:, 2]
    blobs_list = [[], blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'yellow', 'lime', 'red']
    titles = ['None', 'Laplacian of Gaussian', 'Difference of Gaussian',
            'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 4, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(imgCart)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()