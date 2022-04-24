import numpy as np
import scipy as sp
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

Plan:
Check out parse data
Create simulation data
Test on simulation data and compare with naive results
Debug

'''
class MotionDistortionSolver():
    def __init__(self, T_wj0, p_w, p_jt, v_j0, T_wj, lambda_p, lambda_v):
        # e_p Parameters
        self.T_wj0 = T_wj0 # Prior transform, T_w,j0
        self.T_wj0_inv = np.linalg.inv(T_wj0)
        self.p_w = p_w # Estimated point world positions, N x 3
        self.p_jt = p_jt # Observed points at time t, N x 3

        # e_v Parameters
        self.v_j_initial = v_j0 # Initial velocity guess (prior velocity/ velocity from SVD solution)
        self.T_wj_initial = T_wj # Initial Transform guess (T from SVD solution)

        # Optimization parameters
        self.lambda_p = lambda_p # Info matrix, point error, lamdba_p
        self.lambda_v = lambda_v # Info matrix, velocity, lambda_v
        nv = self.lambda_p.shape[0]
        np = self.lambda_v.shape[0]
        self.lambda_total = np.block([[lambda_v, np.zeros((nv, np))],
                                      [np.zeros((np, nv)), lambda_p]])
        self.info_sqrt = sp.linalg.sqrtm(np.linalg.inv(self.lambda_total)) # 5 x 5
        self.dT = None
        self.T_wj_best = T_wj
        self.v_j_best = v_j0 # might not be good enough a guess, too far from optimal

        # Radar Data Params
        self.total_scan_time = 1 / 4 # assuming 4 Hz
        pass

    def compute_time_deltas(self):
        '''
        Get the time deltas for each point. This depends solely on where the
        points are in scan angle. The further away from center, the greater the
        time displacement, and therefore the higher time delta. We use this time
        delta to help us transform the points into an undistorted frame. Note 
        that this is an estimate computed from distorted images. It is a good
        idea to re-run this function once an undistorted frame is obtained for
        better estimates.
        '''
        points = self.undistort()#self.p_jt # provide in N x 3
        x = points[:, 0]
        y = points[:, 1]
        angles = np.arctan2(y, x) # big assumption here: scanline starts at positive x axis and moves counterclockwise
        dT = self.total_scan_time * angles / (2 * np.pi)
        dT -= self.total_scan_time / 2 # offset range, as defined in [-scan_time /2 , scan_time/2]
        self.dT = dT

    def undistort(self, v_j):
        '''
        Computes a new set of undistorted observed points, based on the current
        best estimate of v_T, T_wj, dT
        '''
        displacement = np.expand_dims(v_j, axis = 1) * self.dT # 3 x N
        assert(displacement.shape = (3,points.shape[0]))
        theta = displacement[2, :]
        dx = displacement[0, :]
        dy = displacement[1, :]
        # Correction matrix for time drift, 3 x 3 x N
        T_j_jt = np.array([[np.cos(theta), -np.sin(theta), dx],
                           [np.sin(theta), np.cos(theta),  dy],
                           [0,             0,              1]])
        
        p_jt_col = np.expand_dims(self.p_jt, axis = 1).transpose(axis = (2, 0, 1)) # N x 3 x 1
        undistorted = T_j_jt.transpose(axis = (2, 0, 1)) @ p_jt_col # N x 3 x 1
        return undistorted


    def expected_observed_pts(self, T_wj):
        '''
        Returns the estimated positions of points based on their world location
        estimates and the current best fit transform
        '''
        return np.linalg.inv(T_wj) @ self.p_w.T

    def error_vector(self, params):
        theta = params[2]
        x = params[0]
        y = params[1]
        T = np.array([[np.cos(theta), -np.sin(theta), x],
                      [np.sin(theta),  np.cos(theta), y],
                      [0            ,  0            , 1]])
        return self.info_sqrt @ self.error(params[:3], T)

    def error(self, v_j, T_wj):
        '''
        Return the Cauchy robust error between the undistorted points and their
        estimated observed positions and the velocity error.
        '''
        # Compute point error
        undistorted = self.undistort(v_j)
        expected = self.expected_observed_pts(self, T_wj)
        naive_e_p = expected - np.squeeze(undistorted).T # 3 x N
        # Actual loss is the Cauchy robust loss, defined here:
        e_p_i = np.log(np.square(naive_e_p[:2, :]) / 2 + 1)
        e_p = np.sum(e_p_i, axis = 1) # 2 x 1
        
        # Compute velocity error
        T_j_j1 = self.T_wj0_inv @ T_wj
        dx = T_j_j1[0, 2]
        dy = T_j_j1[1, 2]
        dtheta = np.arctan2(T_j_j1[1, 0], T_j_j1[0, 0])
        v_j_prior = np.array([dx, dy, dtheta]) / self.total_scan_time
        e_v = (v_j - v_j_prior) * e_p.shape[1] # 3 x 1

        e = np.vstack((e_v, e_p))
        return e

    def jacobian_vector(self, params):
        theta = params[2]
        x = params[0]
        y = params[1]
        T = np.array([[np.cos(theta), -np.sin(theta), x],
                      [np.sin(theta),  np.cos(theta), y],
                      [0            ,  0            , 1]])
        return self.info_sqrt @ self.jacobian(params[:3], T)

    def jacobian(self, v_j, T_wj):
        '''
        Compute the Jacobian. This has two parts, as defined by the RadarSLAM
        paper:
        J_p -   gradient of point error and velocity error wrt pose terms Tx,
                Ty, Ttheta
        J_v -   gradient of point error and velocity error wrt velocity terms
                vx, vy, vtheta
        '''
        undistorted = self.undistort(v_j)
        expected = self.expected_observed_pts(self, T_wj)
        input = expected - np.squeeze(undistorted).T # 3 x N
        cauchy_derivative = input / (np.square(input[:2, ]) / 2 + 1) # 2 x N

        # Compute J_p: derivative of e_p wrt
        c0 = self.T_wj0[0, 0]
        s0 = self.T_wj0[1, 0]
        c1 = T_wj[0, 0]
        s1 = T_wj[1, 0]
        Tx = T_wj[0, 2]
        Ty = T_wj[1, 2]
        pwx = self.p_w[:, 0] # 1 x N
        pwy = self.p_w[:, 1]
        ones = np.ones(pwx.shape)

        # 2 x 3 x N
        J_p1 = np.array([[-c1 * ones, -s1 * ones, -pwx * s1 + pwy * c1 - c1 * Ty + s1 * Tx],
                        [s1 * ones,  -c1 * ones, -pwx * c1 - pwy * s1 + s1 * Ty + c1 * Tx]])
        J_p1 *= np.expand_dims(cauchy_derivative, axis = 1)
        J_p2 = np.array([[ c0, s0, 0],
                         [-s0, c0, 0],
                         [0,   0,  1]]) / self.total_scan_time
        J_p = np.vstack((J_p1, J_p2))

        # Compute J_v: derivative of the errors wrt velocity
        points = self.p_jt # observed points
        x = points[:, 0]
        y = points[:, 1]
        displacement = np.expand_dims(v_j, axis = 1) * self.dT # 3 x N
        theta = displacement[2, :]
        J_v = np.array([[-self.dT, 0, np.sin(theta) * self.dT * x + np.cos(theta) * self.dT * y ],
                        [0, -self.dT, -np.cos(theta) * self.dT * x + np.sin(theta) * self.dT * y]])
        J_v *= np.expand_dims(cauchy_derivative, axis = 1)
        J_v = np.sum(J_v, axis = -1) # 3 x 2
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
        # Initialize v, T
        T0 = self.T_wj_initial
        T_params = np.array([T0[0, 2], T0[1, 2], np.arctan2(T0[1, 0], T0[0, 0])])
        initial_guess = np.hstack((self.v_j_initial, T_params))

        result = sp.optimize.least_squares(self.error_vector, initial_guess, jac = self.jacobian, method = 'lm')

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
        print(f"Final v: {best_params[:3]}, t: {best_params[3:]}")
        print(f"Used {num_evals} evaluations")
        print(status_dict[status])
        return best_params
