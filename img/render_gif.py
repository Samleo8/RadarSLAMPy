import imageio
import glob

# cant run shell scripts
# truly a windows moment
images = []
for fname in sorted(
    glob.glob("roam/full_seq_1_traj/*.jpg"), key=lambda x: int(x.split("\\")[1].split(".")[0])
):
    print(fname)
    images.append(imageio.imread(fname))
imageio.mimsave("roam/full_seq_1_traj_1.gif", images, fps=10)