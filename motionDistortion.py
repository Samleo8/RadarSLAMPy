import numpy as np
import scipy as sp
from utils import *
import matplotlib.pyplot as plt
#sp.linalg.sqrtm
#sp.lin

'''
MotionDistortionSolver corrects observed radar feature points for motion
distortion. As the radar is scanning, the car is moving, which means the
radar image is captured at different time steps, one scanline at a time.

This solver explicitly models the distortion as a constant velocity manuever by
the car, and applies a transform to shift all points into the same time frame.
This provides a much better estimate of an instantaneous radar image.

The solver will also correct the naive transform from the previous frame to the
current one. This is a simultaneous optimization: the velocity is optimized wrt
the best fit transform. The best fit transform is improved using the undistorted
points using the velocity. The solver linearizes this system and uses LM to
solve for best fit v and T.

TODO: Run parse data to look at how the scanline moves. Adjust code to match
TODO: Verify code against math
TODO: Integration with the code. Adding mapping capability
TODO: Check arctan2 conventions for dT stuff. the image might be flipped because
images have the origin in the top left.

Plan:
Check out parse data
Create simulation data
Test on simulation data and compare with naive results
Debug

'''
RADAR_SCAN_FREQUENCY = 4 # 4 hz data
VERBOSE = False
class MotionDistortionSolver():
    def __init__(self, T_wj0, p_w, p_jt, T_wj, sigma_p, sigma_v, 
                 frequency = RADAR_SCAN_FREQUENCY):
        # e_p Parameters
        self.T_wj0 = T_wj0 # Prior transform, T_w,j0
        self.T_wj0_inv = np.linalg.inv(T_wj0)
        self.p_w = homogenize(p_w) # Estimated point world positions, N x 3
        self.p_jt = homogenize(p_jt) # Observed points at time t, N x 3
        self.T_wj_initial = T_wj

        # Radar Data Params
        self.total_scan_time = 1 / frequency

        # e_v Parameters
        self.v_j_initial = self.infer_velocity(self.T_wj0_inv @ T_wj)
        # Initial velocity guess (prior velocity/ velocity from SVD solution)

        # Optimization parameters
        self.sigma_p = sigma_p # Info matrix, point error, lamdba_p
        self.sigma_v = sigma_v # Info matrix, velocity, sigma_v
        # n_v = self.sigma_v.shape[0]
        # n_p = self.sigma_p.shape[0]
        # self.sigma_total = np.block([[sigma_v, np.zeros((n_v, n_p))],
        #                               [np.zeros((n_p, n_v)), sigma_p]])
        # self.info_sqrt = sp.linalg.sqrtm(np.linalg.inv(self.sigma_total))
        sigma_p_vector = np.tile(np.diag(sigma_p), p_jt.shape[0])
        sigma_v_vector = np.diag(sigma_v)
        sigma_vector = np.concatenate((sigma_p_vector, sigma_v_vector))
        self.info_vector = 1 / sigma_vector
        self.dT = MotionDistortionSolver.compute_time_deltas(self.total_scan_time, p_jt)
        pass

    def __init__(self, sigma_p, sigma_v, 
                 frequency = RADAR_SCAN_FREQUENCY):
        # Radar Data Params
        self.total_scan_time = 1 / frequency

        # Optimization parameters
        self.sigma_p = np.diag(sigma_p) # Info matrix, point error, lamdba_p
        self.sigma_v = np.diag(sigma_v) # Info matrix, velocity, sigma_v
        pass
    
    def update_problem(self, T_wj0, p_w, p_jt, T_wj, debug= False):
        # e_p Parameters
        self.T_wj0 = T_wj0 # Prior transform, T_w,j0
        self.T_wj0_inv = np.linalg.inv(T_wj0)
        self.p_w = homogenize(p_w) # Estimated point world positions, N x 3
        self.p_jt = homogenize(p_jt) # Observed points at time t, N x 3
        assert(p_w.shape == p_jt.shape)
        self.T_wj_initial = T_wj
        self.debug = debug

        # e_v Parameters
        self.v_j_initial = self.infer_velocity(self.T_wj0_inv @ T_wj)
        # Initial velocity guess (prior velocity/ velocity from SVD solution)
        self.dT = MotionDistortionSolver.compute_time_deltas(self.total_scan_time, p_jt)

        # Info matrix scales to number of points in the optimization problem
        sigma_p_vector = np.tile(self.sigma_p, p_jt.shape[0])
        sigma_v_vector = self.sigma_v
        sigma_vector = np.concatenate((sigma_p_vector, sigma_v_vector))
        self.info_vector = 1 / sigma_vector

    def infer_velocity(self, transform):
        dx = transform[0, 2]
        dy = transform[1, 2]
        dtheta = np.arctan2(transform[1, 0], transform[0, 0])
        return np.array([dx, dy, dtheta]) / self.total_scan_time

    @staticmethod
    def compute_time_deltas(period, points):
        '''
        Get the time deltas for each point. This depends solely on where the
        points are in scan angle. The further away from center, the greater the
        time displacement, and therefore the higher time delta. We use this time
        delta to help us transform the points into an undistorted frame. Note 
        that this is an estimate computed from distorted images. It is a good
        idea to re-run this function once an undistorted frame is obtained for
        better estimates.
        '''
        x = points[:, 0]
        y = points[:, 1]
        # scanline starts at positive x axis and moves clockwise (counter-clockwise?)
        # We take midpoint pi/2 as 0
        angles = np.arctan2(-y, -x) 
        dT = period * angles / (2 * np.pi)
        return dT
    
    @staticmethod
    def undistort(v_j, points, period = 1 / RADAR_SCAN_FREQUENCY, times = None):
        '''
        Computes a new set of undistorted observed points, based on the current
        best estimate of v_T, T_wj, dT
        '''
        # Turn points in homogeneous form if not already
        points = homogenize(points)

        # Get the time deltas for motion distortion 
        if times is None:
            assert(period > 0)
            times = MotionDistortionSolver.compute_time_deltas(period, points)
        
        v_j_column = np.expand_dims(v_j, axis = 1)
        displacement = v_j_column * times # 3 x N

        theta = displacement[2, :] # (N,)
        dx = displacement[0, :] # (N,)
        dy = displacement[1, :] # (N,)
        shape = theta.shape
        # Correction matrix for time drift, 3 x 3 x N
        T_j_jt = np.array([[np.cos(theta), -np.sin(theta), dx],
                           [np.sin(theta), np.cos(theta),  dy],
                           [np.zeros(shape), np.zeros(shape), np.ones(shape)]])
        p_jt_col = np.expand_dims(points, axis = 2) # N x 3 x 1
        undistorted = T_j_jt.transpose((2, 0, 1)) @ p_jt_col # N x 3 x 1
        return np.squeeze(undistorted) # N x 3

    def expected_observed_pts(self, T_wj):
        '''
        Returns the estimated positions of points based on their world location
        estimates and the current best fit transform
        '''
        return np.linalg.inv(T_wj) @ self.p_w.T

    def error_vector(self, params):
        '''
        Because we are optimizing over rotations, we choose to keep the rotation
        in a theta form, we have to do matrix exponential in here to convert
        into the SO(1) form, then augment to the rotation-translation transform
        '''
        theta = params[5]
        x = params[3]
        y = params[4]
        T = np.array([[np.cos(theta), -np.sin(theta), x],
                      [np.sin(theta),  np.cos(theta), y],
                      [0            ,  0            , 1]])
        #return self.info_sqrt @ self.error(params[:3], T)
        return self.info_vector * self.error(params[:3], T)

    def error(self, v_j, T_wj):
        '''
        Return the Cauchy robust error between the undistorted points and their
        estimated observed positions and the velocity error.
        '''
        # Compute point error
        undistorted = MotionDistortionSolver.undistort(v_j, self.p_jt, times=self.dT)
        expected = self.expected_observed_pts(T_wj)
        naive_e_p = expected - undistorted.T # 3 x N

        # Actual loss is the Cauchy robust loss, defined here:
        e_p_i = np.log(np.square(naive_e_p[:2, :]) / 2 + 1) # 2 x N
        e_p = e_p_i.flatten(order='F')
        #e_p = np.sum(e_p_i, axis = 1) # (2,)
        
        # Compute velocity error
        # Matrix log operation
        T_j_j1 = self.T_wj0_inv @ T_wj
        dx = T_j_j1[0, 2]
        dy = T_j_j1[1, 2]
        dtheta = np.arctan2(T_j_j1[1, 0], T_j_j1[0, 0])
        v_j_prior = np.array([dx, dy, dtheta]) / self.total_scan_time
        v_diff = (v_j - v_j_prior)
        v_diff[2] = normalize_angles(v_diff[2])
        e_v = v_diff * e_p_i.shape[1] # (3,)
        # ideally should warp2pi here on theta error
        e = np.hstack((e_p, e_v))

        return e

    def jacobian_vector(self, params):
        theta = params[5]
        x = params[3]
        y = params[4]
        T = np.array([[np.cos(theta), -np.sin(theta), x],
                      [np.sin(theta),  np.cos(theta), y],
                      [0            ,  0            , 1]])
        velocity = params[:3]
        #return self.info_sqrt @ self.jacobian(velocity, T)
        return np.expand_dims(self.info_vector, axis=1) * self.jacobian(velocity, T)
        
    def jacobian(self, v_j, T_wj):
        '''
        Compute the Jacobian. This has two parts, as defined by the RadarSLAM
        paper:
        J_p -   gradient of point error and velocity error wrt pose terms Tx,
                Ty, Ttheta
        J_v -   gradient of point error and velocity error wrt velocity terms
                vx, vy, vtheta
        '''
        undistorted = MotionDistortionSolver.undistort(v_j, self.p_jt, times=self.dT)
        expected = self.expected_observed_pts(T_wj)
        input = expected - undistorted.T # 3 x N
        naive_e_p = input[:2]
        cauchy_derivative = naive_e_p / (np.square(naive_e_p) / 2 + 1) # 3 x N

        # Compute J_p: derivative of errors wrt the point position
        c0 = self.T_wj0[0, 0]
        s0 = self.T_wj0[1, 0]
        c1 = T_wj[0, 0]
        s1 = T_wj[1, 0]
        Tx = T_wj[0, 2]
        Ty = T_wj[1, 2]
        pwx = self.p_w[:, 0] # 1 x N
        pwy = self.p_w[:, 1]
        ones = np.ones(pwx.shape)
        zeros = np.zeros(pwx.shape)

        # 2 x 3 x N
        J_p1 = np.array([[-c1 * ones, -s1 * ones, -pwx * s1 + pwy * c1 - c1 * Ty + s1 * Tx],
                        [s1 * ones,  -c1 * ones, -pwx * c1 - pwy * s1 + s1 * Ty + c1 * Tx]])
        J_p1 *= np.expand_dims(cauchy_derivative, axis = 1)
        J_p1 = np.squeeze(np.vstack(np.split(J_p1, J_p1.shape[2], axis = 2)))
        J_p2 = np.array([[ c0, s0, 0],
                         [-s0, c0, 0],
                         [0,   0,  1]]) / self.total_scan_time * pwx.shape[0]
        J_p = np.vstack((J_p1, J_p2))

        # Compute J_v: derivative of the errors wrt velocity
        points = self.p_jt # observed points
        x = points[:, 0]
        y = points[:, 1]
        displacement = np.expand_dims(v_j, axis = 1) * self.dT # 3 x N
        theta = displacement[2, :]
        zeros = np.zeros(theta.shape)
        J_v = np.array([[-self.dT, zeros, self.dT * (np.sin(theta) * x + np.cos(theta) * y) ],
                        [zeros, -self.dT, self.dT * (-np.cos(theta) * x + np.sin(theta) * y)]])
        J_v *= np.expand_dims(cauchy_derivative, axis = 1)
        J_v = np.squeeze(np.vstack(np.split(J_v, J_v.shape[2], axis = 2))) # 2N x 3
        J_v = np.vstack((J_v, np.eye(3) * x.shape[0]))
        
        # J = [J_v, J_p]
        J = np.hstack((J_v, J_p))
        return J

    def optimize(self, max_iters = 20):
        # Initialize v, T

        # while not converged:
            # Find e(v, T)

            # Find J

            # e_i = J d_v_T + e(v, T)

            # Minimize sum e_i.T lambda e_i

            # Equivalent to minimizing sqrt(lambda) @ (J d_v_T + e(v, T))
            # = (sqrt(lambda) @ J) d_v_T + (sqrt(lambda) @ e(v, T)))

            # d_v_T = least_squares(A = (sqrt(lambda) @ J), b = - (sqrt(lambda) @ e(v, T)))

            # v,T += d_v_T

        # return v, T

        pass

    def optimize_library(self):
        '''
        Optimize using the LM implementation in the scipy library.
        '''
        # Initialize v, T
        T0 = self.T_wj_initial
        T_params = np.array([T0[0, 2], T0[1, 2], np.arctan2(T0[1, 0], T0[0, 0])])
        initial_guess = np.hstack((self.v_j_initial, T_params))
        if VERBOSE:
            print(f"Initial v guess: {self.v_j_initial}")
            print(f"Initial T guess: {T_params}")

        result = sp.optimize.least_squares(self.error_vector, initial_guess, jac = '2-point', method = 'lm')
        # result = sp.optimize.least_squares(self.error_vector, initial_guess, jac = self.jacobian_vector, method = 'lm')
        # return v, T
        best_params = result.x
        num_evals = result.nfev # number of function evaluations: measure of how quickly we converged
        status = result.status
        status_dict = {-1 : "improper input parameters status returned from MINPACK.",
                        0 : "the maximum number of function evaluations is exceeded.",
                        1 : "gtol termination condition is satisfied",
                        2 : "ftol termination condition is satisfied",
                        3 : "xtol termination condition is satisfied",
                        4 : "Both ftol and xtol termination conditions are satisfied"}
        if VERBOSE:
            print(f"Final v: {best_params[:3]}")
            print(f"Final t: {best_params[3:]}")
            print(f"Used {num_evals} evaluations")
            print(f"Residuals were {result.fun}")
            print(status_dict[status])
        return best_params
