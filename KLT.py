from typing import Tuple
import numpy as np
import scipy.ndimage as sp
import cv2

import matplotlib.pyplot as plt
import os, sys
from getFeatures import getBlobsFromCart

from parseData import getCartImageFromImgPaths, getRadarImgPaths
'''
KLT: Kanade Lucas Tomasi Tracker
This tracker implements the KLT feature tracker, which tracks features between
two images without using descriptor matching.

The algorithm can be summarized as iterative least-squares across the gradient
of the difference image between two consecutive images.

This solver provides a solution in SE(2), assuming 2D data.

'''


def cloud_to_image(cloud, size):
    newImage = np.zeros(size)
    feature_r = cloud[:, 0]
    feature_c = cloud[:, 1]
    newImage[feature_r, feature_c] = 1
    return newImage


'''
KLT: Let G(x) be the new image, F(x) be the previous image.
We assume that the new image is formed from a rotation and translation
G(x) = F(Ax + h)

We want to find least-squares A, h. We can define error
E = sum[(F(Ax + h) - G(x))^2]

To solve, we use a linear approximation: x constant, A is linearization pt, 
F(x(A + delta_A) + (h + delta_h)) ~= F(Ax + h) + (delta_A x + delta_h) dF(x)/dx (evaluated at Ax + b)
F(Ax + h) ~= F(Ax + h) + (delta_A x + delta_h) dF(x)/dx (evaluated at Ax + b)

Therefore, the error term is:
E = sum[(F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x))^2]

We want to minimize this E:
dE/ddelta_A = d/ddelta_A sum[(F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x))^2]
= sum [ d/ddelta_A[(F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x))^2]]
= sum [2 * (F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x)) * 
      d/ddelta_A (F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x)) ]
= sum [2 * (F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x)) * 
      (x * dF(x)/dx)]
dE/ddelta_A = 0
=> sum [2 * (F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x)) * 
       (x * dF(x)/dx) ] = 0
=> sum [2 *  delta_A x * dF(x)/dx * (x * dF(x)/dx)] = sum[ 2 * (-F(Ax + h) - (delta_h) dF(x)/dx + G(x)) * 
       (x * dF(x)/dx)]
=> delta_A = sum[ 2 * (-F(Ax + h) - (delta_h) dF(x)/dx + G(x)) * (x * dF(x)/dx)]
            /sum [2 * x * dF(x)/dx * (x * dF(x)/dx)]

Similarly, to minimize E wrt h:
dE/ddelta_h = d/ddelta_h sum[(F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x))^2]
= sum [ d/ddelta_h[(F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x))^2]]
= sum [2 * (F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x)) * 
      d/ddelta_h (F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x)) ]
= sum [2 * (F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x)) * 
      (dF(x)/dx)]
dE/ddelta_h = 0
=> sum [2 * (F(Ax + h) + (delta_A x + delta_h) dF(x)/dx - G(x)) * (dF(x)/dx) ] = 0
=> sum [2 *  delta_h * dF(x)/dx * dF(x)/dx] = sum[ 2 * (-F(Ax + h) - (delta_A x) dF(x)/dx + G(x)) * 
       (dF(x)/dx)]
=> delta_h = sum[ 2 * (-F(Ax + h) - (delta_A x) dF(x)/dx + G(x)) * (dF(x)/dx)]
            /sum [2 * dF(x)/dx * dF(x)/dx)]

Inputs:
image0  - previous image. SD np array with binary features, or z x 2 pt cloud
image1  - current image. Same format as image0
size    - image size (M, N)
cloud   - optional parameter: if true, then image 1 is given as a z x 2 array of
          feature points
visual  - boolean to enable cv2 imshow visualizations per iteration
'''


def KLT(image0: np.ndarray,
        image1: np.ndarray,
        size: tuple[int, int],
        cloud: bool = False,
        max_iters: int = 20,
        visual: bool = True,
        verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    # Convert clouds to binary images
    if cloud:
        newImage0 = cloud_to_image(image0, size)
        newImage1 = cloud_to_image(image1, size)
    else:
        if verbose:
            assert (image0.shape == image1.shape)

        newImage0 = image0
        newImage1 = image1
    if visual:
        cv2.imshow("Prev Image", newImage0)
        cv2.imshow("Next Image", newImage1)
        diff = np.square(newImage1 - newImage0)
        cv2.imshow("Diff Img", diff)
        cv2.waitKey(0)

    A = np.eye(2)
    h = np.zeros((2, ))

    iters = 0
    rows = np.arange(size[0])
    cols = np.arange(size[1])
    r_mat, c_mat = np.meshgrid(rows, cols, indexing='ij')

    # Iterative least squares to find the optimal transform
    while iters < max_iters:
        indices = np.expand_dims(np.stack((r_mat, c_mat), axis=2), axis=3)

        # Apply affine transform F(Ax + b) to image F(x). M x N
        warped_image = sp.affine_transform(newImage0, A, h, mode='nearest')
        if visual:
            cv2.imshow(f"Iteration {iters} warp", warped_image)
            cv2.waitKey(0)
        loss = np.mean(np.square(warped_image - newImage1))

        if verbose:
            print(f"Iteration {iters} loss: {loss}")

        # Image gradient across x and y axes, M x N x 2
        dFx_dx = np.stack(np.gradient(warped_image, axis=(0, 1)), axis=2)

        # Compute the optimal delta A in the linearized system
        # Dimensions: 2 x 2 = [((M x N) - (M x N x 2)(2,) + (M x N)) * (M x N x 2) * (M x N x 2)]
        # Formula: delta_A = sum[ 2 * (-F(Ax + h) - (delta_h) dF(x)/dx + G(x)) * (x * dF(x)/dx)]
        #    /sum [2 * x * dF(x)/dx * (x * dF(x)/dx)]

        # (-F(Ax + h) - (delta_h) dF(x)/dx + G(x)): M x N x 1 x 1
        # should be newImage1 - warped_image - dFx_dx @ delta_h, but delta_h is 0
        pixel_grad_h = np.expand_dims(newImage1 - warped_image, axis=(2, 3))
        # (x * dF(x)/dx): M x N x 2 x 2
        inner_Jacobian = np.expand_dims(dFx_dx, axis=3) @ np.transpose(
            indices, axes=(0, 1, 3, 2))
        # (-F(Ax + h) - (delta_h) dF(x)/dx + G(x)) * (x * dF(x)/dx)): M x N x 2 x 2
        numer = np.sum(pixel_grad_h * inner_Jacobian, axis=(0, 1))
        inner_Jacobian_T = np.transpose(inner_Jacobian, axes=(0, 1, 3, 2))
        denom = np.sum(inner_Jacobian_T @ inner_Jacobian, axis=(0, 1))
        delta_A = numer @ np.linalg.inv(denom)

        # Compute the optimal delta_h vector
        # Formula: delta_h = sum[(-F(Ax + h) - (delta_A x) dF(x)/dx + G(x)) * (dF(x)/dx)]
        #    /sum [dF(x)/dx * dF(x)/dx)]

        # The following code block allows us to use the delta_A solved above to
        # improve the least squares solution for delta_h. It is removed for
        # debugging but can be added back in for faster convergence
        '''
        warped_x = delta_A @ indices
        A_delta = (np.expand_dims(dFx_dx, axis = 2) @ warped_x)[:, :, 0, 0]
        assert(A_delta.shape == size)
        '''
        warped_grad = np.expand_dims(
            newImage1 - warped_image,
            axis=2)  #replace with (newImage1 - warped_image - A_delta)
        numer = np.sum(warped_grad * dFx_dx, axis=(0, 1))
        dFx_dx = np.expand_dims(dFx_dx, axis=2)  # M x N x 1 x 2
        dFx_dx_T = np.transpose(dFx_dx, axes=(0, 1, 3, 2))
        denom = np.sum(dFx_dx_T @ dFx_dx, axis=(0, 1))
        delta_h = numer @ np.linalg.inv(denom)

        # Accumulate the new transform
        A = A + delta_A
        h = h + delta_h
        if verbose:
            print(f"Iteration {iters}:\ndA=\n{delta_A}\ndh:\n{delta_h}")

        iters += 1

    return A, h


def getTransformedFeatures(prevFeatureInd, A, h):
    h = h.reshape(2, 1)
    transformedSourceFeatures = (A @ (prevFeatureInd.T - h)).T

    return transformedSourceFeatures


def visualize_transform(prevImg, currImg, prevFeatureInd, currFeatureInd, A,
                        h, show=False):
    # Visualize
    print(f"Final Transform:\nA:\n{A}\nh:\n{h}")
    fit = sp.affine_transform(prevImg, A, h, mode='nearest')

    # Temporary display
    plt.subplot(1, 2, 1)
    plt.imshow(prevImg)
    plt.scatter(prevFeatureInd[:, 1],
                prevFeatureInd[:, 0],
                marker='.',
                color='red')
    plt.title("Old Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(currImg)
    plt.scatter(currFeatureInd[:, 1],
                currFeatureInd[:, 0],
                marker='.',
                color='red',
                label='Image 1 Features')

    # TODO: Remove, show feature points of old images
    plt.scatter(prevFeatureInd[:, 1],
                prevFeatureInd[:, 0],
                marker='^',
                color='green',
                label='Image 0 Features')

    # NOTE: A and h are inverse poses
    transformedSourceFeatures = getTransformedFeatures(prevFeatureInd, A, h)
    plt.scatter(transformedSourceFeatures[:, 1],
                transformedSourceFeatures[:, 0],
                marker='+',
                color='blue',
                label='Transformed Img 0 Features')

    plt.legend()
    plt.axis("off")
    plt.title("New Image")

    if show:
        plt.show()

if __name__ == "__main__":
    datasetName = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    dataPath = os.path.join("data", datasetName, "radar")
    timestampPath = os.path.join("data", datasetName, "radar.timestamps")

    # Incremental streaming
    imgPathArr = getRadarImgPaths(dataPath, timestampPath)
    nImgs = len(imgPathArr)

    for imgNo in range(nImgs):
        currImg = getCartImageFromImgPaths(imgPathArr, imgNo)

        # TODO: What are the values for num, min and max sigma
        blobIndices = getBlobsFromCart(currImg,
                                       min_sigma=0.01,
                                       max_sigma=10,
                                       num_sigma=3,
                                       threshold=.0005,
                                       method="doh")
        currFeatureInd = blobIndices[:, :2].astype(int)

        if imgNo:
            print("Computing affine transforms", end="... ", flush=True)
            A, h = KLT(prevFeatureInd,
                       currFeatureInd,
                       currImg.shape,
                       cloud=True,
                       visual=False)
            print("Done.")

            # np.savez("transform1.npz", A=A, h=h)
            # with np.load("transform1.npz") as data:
            #     A = data['A']
            #     h = data['h']

            visualize_transform(prevImg, currImg, prevFeatureInd,
                                currFeatureInd, A, h, show=False)
            # plt.pause(100)

        prevImg = np.copy(currImg)
        prevFeatureInd = np.copy(currFeatureInd)

    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     # Testing function here
#     data = np.load("data/KLT_test/aerialseq.npy")
#     print(data.shape)
#     # Two images to track
#     image0 = data[:, :, 0]
#     image1 = data[:, :, 5]
#     size = image0.shape
#     A, h = KLT(image0, image1, size, visual = False, max_iters = 50)

#     # Visualize
#     print(f"Final Transform:\nA:\n{A}\nh:\n{h}")
#     fit = sp.affine_transform(image0, A, h, mode = 'nearest')
#     cv2.imshow("Original New Image", image1)
#     cv2.imshow("Original Old Image", image0)
#     cv2.imshow("Best fit warp", fit)
#     diff = np.square(image1-fit)
#     cv2.imshow("Diff Img", diff)
#     cv2.waitKey(0)

#     loss = np.mean(diff)
#     print(f"Final loss: {loss}")
#     pass
