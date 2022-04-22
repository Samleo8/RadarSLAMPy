from utils import *
from trajectoryPlotting import Trajectory, plotGtAndEstTrajectory
import numpy as np
from getTransformKLT import calculateTransformSVD

# dx, dy, dth_deg in robot coords
commands = np.array([
    [1, 0, 0],
    [2, .1, 0],
    [3, -.2, 0],
    [3, -.2, 1],
    [3, .2, 2],
    [3, .2, 5],
    [3, -.2, 7],
    [5, -.2, 8],
    [5, .2, 0],
    [5, .2, 0],
    [3, 0, 0],
    [2, 0, 0],
    [-.5, 0, -1],
    [-.5, 0, -2],
    [-2, 0, -2],
    [-1, 0, -2],
    [-.5, 0, -2],
    [-.5, .1, -2],
    [-1, -.1, -1],
    [-1, .1, -1],
    [-3, 0, -3],
    [-5, 0, 0],
])
commands[:,2] = np.deg2rad(commands[:,2])

# integrate command to get gt poses
poses = [[0,0,0]]
x, y, th = 0, 0, 0
for row in commands:
    dx = float(row[0])
    dy = float(row[1])
    dth = float(row[2])
    x += dx * np.cos(th) + dy * -np.sin(th)
    y += dx * np.sin(th) + dy * np.cos(th)
    th += dth
    poses.append([x,y,th])
poses = np.array(poses)
pose_transforms = convertPoseToTransform(poses)
times = [*range(len(poses))]
T = len(times)

# calculate xlims and ylims of trajectory
xlims = [-100, 100]
ylims = [-100, 100]

# generate landmarks
gtLandmarks = (np.random.rand(200,2) - .5) * 100
plt.title('ground truth')
plt.scatter(gtLandmarks[:,0], gtLandmarks[:,1])
quiver(poses)
plt.xlim(xlims[0],xlims[1])
plt.ylim(ylims[0],ylims[1])
plt.show(block=True)

# calculate sensed landmarks by frame
relLandmarksOverTime = []
for t in range(T):
    # l_x_wrld, l_y_wrld is landmark wrt to world
    # A is transformation matrix of robot wrt world
    # l_x_rob, l_y_rob is landmark wrt to robot
    relLandmarks = []
    for l_x_wrld,l_y_wrld in gtLandmarks:
        A = pose_transforms[t]
        xy = np.linalg.inv(A) @ [l_x_wrld,l_y_wrld,1]
        l_x_rob,l_y_rob = xy[:2]
        relLandmarks.append([l_x_rob,l_y_rob])
    relLandmarksOverTime.append(relLandmarks)
relLandmarksOverTime = np.array(relLandmarksOverTime)
noise = np.random.randn(T,len(relLandmarksOverTime[0]),2) * .5
relLandmarksOverTime += noise

# plot trajectory and landmarks
for t in range(T):
    plt.clf()
    quiver([[0,0,0]])
    scat = plt.scatter(relLandmarksOverTime[t,:,0], relLandmarksOverTime[t,:,1])
    plt.xlim(xlims[0],xlims[1])
    plt.ylim(ylims[0],ylims[1])
    if t == 0:
        plt.show(block=False)
    plt.pause(0.01)

# create trajectory objects
gtTraj = Trajectory(times,poses)
estTraj = Trajectory([0],[[0,0,0]])

for t in range(1,T):
    # estimate transform
    good_old = relLandmarksOverTime[t-1]
    good_new = relLandmarksOverTime[t]
    R, h = calculateTransformSVD(good_old, good_new)
    R_th = np.arctan2(R[1, 0], R[0, 0]) * 180 / np.pi
    dx_rob_est, dy_rob_est, dth_rob_est = -R_th, -float(h[0]), -float(h[1])
    print(f"[Calculated]: d_th={dth_rob_est:.2f} \t h=[{dx_rob_est:.2f},{dy_rob_est:.2f}]")
    
    # get actual transform
    dx_rob_gt, dy_rob_gt, dth_rob_gt = commands[t-1]
    dth_rob_gt = np.rad2deg(dth_rob_gt)
    print(f"    [Actual]: d_th={dth_rob_gt:.2f} \t h={[dx_rob_gt,dy_rob_gt]}")
    estTraj.appendRelativeTransform(t, R, h)

plotGtAndEstTrajectory(gtTraj, estTraj)
plt.show(block=True)