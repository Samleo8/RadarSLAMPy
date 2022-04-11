import numpy as np
import scipy.ndimage as sp
import cv2
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
def KLT(image0, image1, size, cloud = False, max_iters = 20, visual = True):
    assert(image0.shape == image1.shape)

    # Convert clouds to binary images
    if cloud:
        newImage0 = cloud_to_image(image0, size)
        newImage1 = cloud_to_image(image1, size)
    else:
        newImage0 = image0
        newImage1 = image1
    if visual:
        cv2.imshow("Prev Image", newImage0)
        cv2.imshow("Next Image", newImage1)
        diff = np.square(newImage1-newImage0)
        cv2.imshow("Diff Img", diff)
        cv2.waitKey(0)

    A = np.eye(2) #np.zeros((2, 2))
    h = np.zeros((2,))
    
    iters = 0
    rows = np.arange(size[0])
    cols = np.arange(size[1])
    r_mat, c_mat = np.meshgrid(rows, cols, indexing = 'ij')

    # Iterative least squares to find the optimal transform
    while iters < max_iters:
        indices = np.expand_dims(np.stack((r_mat, c_mat), axis = 2), axis = 3)
        #new_indices = A @ indices + b # Check that the indices are of correct dimensions
        
        # Apply affine transform F(Ax + b) to image F(x). M x N
        # Note: A is the inverse transform. It is used to warp the new image's
        # indices onto the old image, then the pixels are mapped, before sending
        # these pixels back to the new image. Might have to invert A here.
        A_inv = np.linalg.inv(A)
        warped_image = sp.affine_transform(newImage0, A_inv, h, mode = 'constant')
        if visual:
            cv2.imshow(f"Iteration {iters} warp", warped_image)
            cv2.waitKey(0)

        '''
        # Inner ALS optimization:
        inner_iters = 0
        max_inner_iters = 10
        delta_A = np.zeros((2, 2))
        delta_h = np.zeros((2,))
        while (inner_iters < max_inner_iters):
        '''
        # M x N
        #dFx_dx = sp.gaussian_gradient_magnitude(warped_image, sigma = 1, 
        #                                        mode = 'constant')
        # Image gradient across x and y axes, M x N x 2
        dFx_dx = np.stack(np.gradient(warped_image, axis = (0, 1)), axis = 2)

        # Compute the optimal delta A in the linearized system
        # 2 x 2 = [((M x N) - (M x N x 2)(2,) + (M x N)) * (M x N x 2) * (M x N x 2)]
        #delta_A = sum[ 2 * (-F(Ax + h) - (delta_h) dF(x)/dx + G(x)) * (x * dF(x)/dx)]
        #    /sum [2 * x * dF(x)/dx * (x * dF(x)/dx)]
        # M x N x 1 x 1
        pixel_grad_h = np.expand_dims(newImage1 - warped_image, axis = (2, 3)) #newImage1 - warped_image - dFx_dx @ delta_h
        # M x N x 2 x 2
        inner_Jacobian = np.expand_dims(dFx_dx, axis = 3) @ np.transpose(indices, axes = (0,1,3,2))
        numer = np.sum(pixel_grad_h * inner_Jacobian, axis = (0, 1)) # M x N x 2 x 2
        inner_Jacobian_T = np.transpose(inner_Jacobian, axes = (0, 1, 3, 2))
        denom = np.sum(inner_Jacobian_T @ inner_Jacobian, axis = (0, 1)) # could be element-wise square, or some other config?
        delta_A = numer @ np.linalg.inv(denom)
        # (M x N) (M x N) (N x M)

        # Compute the optimal h transform vector
        # [((M x N) - (M x N x 2 x 1)(2 x 2))]
        #delta_h = sum[(-F(Ax + h) - (delta_A x) dF(x)/dx + G(x)) * (dF(x)/dx)]
        #    /sum [dF(x)/dx * dF(x)/dx)]
        #warped_x = delta_A @ indices
        #print(warped_x.shape)
        #print(dFx_dx.shape)
        #A_delta = (np.expand_dims(dFx_dx, axis = 2) @ warped_x)[:, :, 0, 0]
        #assert(A_delta.shape == size)
        warped_grad = np.expand_dims(newImage1 - warped_image, axis = 2) #newImage1 - warped_image - A_delta
        numer = np.sum(warped_grad * dFx_dx, axis = (0, 1))
        dFx_dx = np.expand_dims(dFx_dx, axis = 2) # M x N x 1 x 2
        dFx_dx_T = np.transpose(dFx_dx, axes = (0, 1, 3, 2))
        denom = np.sum(dFx_dx_T @ dFx_dx, axis = (0, 1))
        delta_h = numer @ np.linalg.inv(denom)
        
        # Accumulate the new transform
        A = A + delta_A
        h = h + delta_h
        print(f"Iteration {iters}:\ndA=\n{delta_A}\ndh:\n{delta_h}")
        
        iters += 1

    return A, h
        
if __name__ == '__main__':
    # Testing function here
    data = np.load("data/KLT_test/aerialseq.npy")
    print(data.shape)
    image0 = data[:, :, 50]
    image1 = data[:, :, 50]
    size = image0.shape
    A, h = KLT(image0, image1, size, visual = False)
    print(f"Final Transform:\nA:\n{A}\nh:\n{h}")
    fit = sp.affine_transform(image0, np.linalg.inv(A), h, mode = 'constant')
    cv2.imshow("Original New Image", image1)
    cv2.imshow("Original Old Image", image0)
    cv2.imshow("Best fit warp", fit)
    diff = np.square(image1-fit)
    cv2.imshow("Diff Img", diff)
    cv2.waitKey(0)
    pass
