import numpy as np
import scipy.ndimage as sp
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

    # Convert clouds to binary images
    if cloud:
        newImage0 = cloud_to_image(image0, size)
        newImage1 = cloud_to_image(image1, size)
    else:
        newImage0 = image0
        newImage1 = image1

    A = np.eye(2)
    h = np.zeros((2,))
    iters = 0
    rows = np.arange(size[0])
    cols = np.arange(size[1])
    r_mat, c_mat = np.meshgrid(rows, cols)

    # Iterative least squares to find the optimal transform
    while iters < max_iters:
        indices = np.expand_dims(np.stack(c_mat, r_mat, axis = 2), axis = 3)
        #new_indices = A @ indices + b # Check that the indices are of correct dimensions
        
        # M x N
        warped_image = sp.affine_transform(newImage0, A, h, mode = 'constant')
        # M x N
        #dFx_dx = sp.gaussian_gradient_magnitude(warped_image, sigma = 1, 
        #                                        mode = 'constant')
        # Image gradient across x and y axes, M x N x 2
        dFx_dx = np.stack(np.gradient(warped_image, axis = (0, 1)), axis = 2)


        # 2 x 2 = [((M x N) - (M x N x 2)(2,) + (M x N)) * (M x N x 2) * (M x N x 2)]
        numer = np.sum((newImage0 - warped_image - dFx_dx @ h) * (x * dF(x)/dx))
        denom = np.sum(x * dF(x)/dx * (x * dF(x)/dx), axis = (0, 1))
        delta_A = numer / denom
        # (M x N) (M x N) (N x M)

        # Compute the optimal h transform vector
        # [((M x N) - (M x N x 2 x 1)(2 x 2))]
        #delta_h = sum[(-F(Ax + h) - (delta_A x) dF(x)/dx + G(x)) * (dF(x)/dx)]
        #    /sum [dF(x)/dx * dF(x)/dx)]
        warped_x = A @ indices
        A_delta = np.tensordot(warped_x, dFx_dx, axis = 1)
        assert(A_delta.shape == size)
        warped_grad = np.expand_dims(newImage1 - warped_image - A_delta, axis = 2)
        numer = np.sum(warped_grad * dFx_dx, axis = (0, 1))
        denom = np.sum(np.square(dFx_dx), axis = (0, 1))
        delta_h = numer / denom
        
        # Accumulate the new transform
        A = A + delta_A
        h = h + delta_h
        
        iters += 1

    return
        
if __name__ == '__main__':
    pass
