import numpy as np

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
'''
def KLT(image0, image1, size, cloud = False, max_iters = 20):
    assert(image0.shape == image1.shape)

    if cloud:
        newImage0 = cloud_to_image(image0, size)
        newImage1 = cloud_to_image(image1, size)

    A = np.eye(2)
    b = 0
    iters = 0
    rows = np.arange(size[0])
    cols = np.arange(size[1])
    r_mat, c_mat = np.meshgrid(rows, cols)

    while iters < max_iters:
        indices = np.expand_dims(np.stack(c_mat, r_mat, axis = 2), axis = 
        new_indices = A @ indices + b # Check that the indices are of correct dimensions
        iters += 1
        

