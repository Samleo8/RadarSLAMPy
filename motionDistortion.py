import numpy as np
import scipy as sp
#sp.linalg.sqrtm
#sp.lin

'''
TODO: Run parse data to look at how the scanline moves. Adjust code to match
TODO: Verify code against math
'''
class MotionDistortionSolver():
    def __init__(self, T_wj0, p_w, p_jt, v_j0, T_wj, lambda_p, lambda_v, dT):
        # Wanted parameters
        # Prior transform, T_w,j0
        self.T_wj0 = T_wj0
        self.T_wj0_inv = np.linalg.inv(T_wj0)
        # Estimated point world positions, p_w
        self.p_w = p_w # N x 3
        # Observed points at time t, p_jt 
        self.p_jt = p_jt # N x 3
        # Initial velocity guess (prior velocity/ velocity from SVD solution), v_j0
        self.v_j0 = v_j0
        # Initial Transform guess (T from SVD solution), T_w,j,initial
        self.T_wj_initial = T_wj
        # Info matrix, point error, lamdba_p
        self.lambda_p = lambda_p
        # Info matrix, velocity, lambda_v
        self.lamdba_v = lambda_v
        self.dT = dT
        pass

    def pure_distortion_error(self, v_j, T_wj):
        '''
        Returns the absolute difference between the estimated point position in
        the current robot frame and the undistorted point. The point
        '''
        # Get the time deltas for each point
        points = self.p_jt
        x = points[:, 0]
        y = points[:, 1]
        angles = np.arctan2(y, x) # big assumption here: scanline starts at positive x axis and moves counterclockwise
        total_scan_time = 1 / 4 # assuming 4 Hz
        dT = total_scan_time * angles / (2 * np.pi)
        dT -= total_scan_time / 2 # offset range, as defined in [-scan_time /2 , scan_time/2]

        # Compute the point errors: dist between corrected observed points and
        # estimated point position in this frame
        displacement = v_j * dT
        theta = displacement[2]
        dx = displacement[0]
        dy = displacement[1]
        # Correction matrix for time drift
        T_j_jt = np.array([[np.cos(theta), -np.sin(theta), dx],
                           [np.sin(theta), np.cos(theta),  dy],
                           [0,             0,              1]])
        naive_e_p = np.linalg.inv(T_wj) @ self.p_w.T - T_j_jt @ self.p_jt.T # 3 x N
        return naive_e_p

    def error(self, v_j, T_wj):
        '''
        Return the Cauchy robust error at the given point v
        '''
        # Get the time deltas for each point
        points = self.p_jt
        x = points[:, 0]
        y = points[:, 1]
        angles = np.arctan2(y, x) # big assumption here: scanline starts at positive x axis and moves counterclockwise
        total_scan_time = 1 / 4 # assuming 4 Hz
        dT = total_scan_time * angles / (2 * np.pi)
        dT -= total_scan_time / 2 # offset range, as defined in [-scan_time /2 , scan_time/2]

        # Compute the point errors: dist between corrected observed points and
        # estimated point position in this frame
        displacement = v_j * dT
        theta = displacement[2]
        dx = displacement[0]
        dy = displacement[1]
        # Correction matrix for time drift
        T_j_jt = np.array([[np.cos(theta), -np.sin(theta), dx],
                           [np.sin(theta), np.cos(theta),  dy],
                           [0,             0,              1]])
        naive_e_p = np.linalg.inv(T_wj) @ self.p_w.T - T_j_jt @ self.p_jt.T # 3 x N
        #naive_e_p = self.pure_distortion_error(self, v_j, T_wj) # 3 x N
        # Actual loss is the Cauchy robust loss, defined here:
        e_p_i = np.log(np.square(naive_e_p[:2, :]) / 2 + 1)
        e_p = np.sum(e_p_i, axis = 1) # 2 x 1
        
        T_j_j1 = self.T_wj0_inv @ T_wj
        v_j_prior = np.array([T_j_j1[0, 2], T_j_j1[1, 2], np.arctan2(T_j_j1[1, 0], T_j_j1[0, 0])]) / total_scan_time
        e_v = (v_j - v_j_prior) * e_p.shape[1] # 3 x 1

        e = np.vstack((e_v, e_p))
        return e

    def jacobian(self, v_j, T_wj):
        # Get the time deltas for each point
        points = self.p_jt
        x = points[:, 0]
        y = points[:, 1]
        angles = np.arctan2(y, x) # big assumption here: scanline starts at positive x axis and moves counterclockwise
        total_scan_time = 1 / 4 # assuming 4 Hz
        dT = total_scan_time * angles / (2 * np.pi)
        dT -= total_scan_time / 2 # offset range, as defined in [-scan_time /2 , scan_time/2]

        # Compute the point errors: dist between corrected observed points and
        # estimated point position in this frame
        displacement = v_j * dT
        theta = displacement[2]
        dx = displacement[0]
        dy = displacement[1]
        # Correction matrix for time drift
        T_j_jt = np.array([[np.cos(theta), -np.sin(theta), dx],
                           [np.sin(theta), np.cos(theta),  dy],
                           [0,             0,              1]])
        input = np.linalg.inv(T_wj) @ self.p_w.T - T_j_jt @ self.p_jt.T # 3 x N
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
                         [0,   0,  1]]) / total_scan_time
        J_p = np.vstack((J_p1, J_p2))

        # Compute J_v: derivative of the errors wrt velocity
        J_v = np.array([[-dT, 0, np.sin(theta) * dT * x + np.cos(theta) * dT * y ],
                        [0, -dT, -np.cos(theta) * dT * x + np.sin(theta) * dT * y]])
        J_v *= np.expand_dims(cauchy_derivative, axis = 1)
        J_v = np.vstack((J_v, np.eye(3)))
        
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
