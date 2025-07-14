import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

data = np.load('../train_data_26_100_3.npy')  # (26, 100, 3)

n_trajectories, timesteps, _ = data.shape
n_rows, n_cols = 5, 6
labels = [chr(ord('a') + i) for i in range(n_trajectories)]

fig = plt.figure(figsize=(20, 15))
axes = [fig.add_subplot(n_rows, n_cols, i+1, projection='3d') for i in range(n_trajectories)]
lines = []
points = []

for i in range(n_trajectories):
    ax = axes[i]
    ax.set_xlim(np.min(-data[:, :, 0]), np.max(-data[:, :, 0]))
    ax.set_ylim(np.min(-data[:, :, 1]), np.max(-data[:, :, 1]))
    ax.set_zlim(np.min(data[:, :, 2]), np.max(data[:, :, 2]))
    ax.set_title(f"Trajectory {labels[i]}", fontsize=10)

    line, = ax.plot([], [], [], lw=1)
    point, = ax.plot([], [], [], 'o')
    lines.append(line)
    points.append(point)

def update(frame):
    for i in range(n_trajectories):
        traj = data[i]
        x, y, z = traj[:frame+1].T
        lines[i].set_data(-x, -y)
        lines[i].set_3d_properties(z)
        points[i].set_data(-x[-1:], -y[-1:])
        points[i].set_3d_properties(z[-1:])
    return lines + points

ani = FuncAnimation(fig, update, frames=timesteps, interval=100, blit=False)

plt.tight_layout()
plt.show()
