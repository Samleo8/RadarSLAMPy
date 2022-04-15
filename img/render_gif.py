import imageio
import glob

# cant run shell scripts
# truly a windows moment
images = []
for fname in sorted(
    glob.glob("track_klt_thresholding/full_seq_1_traj/*.jpg"), key=lambda x: int(x.split("\\")[1].split(".")[0])
):
    print(fname)
    images.append(imageio.imread(fname))
imageio.mimsave("track_klt_thresholding/full_seq_1_traj.gif", images, fps=10)