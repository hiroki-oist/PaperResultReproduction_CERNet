import numpy as np
import matplotlib.pyplot as plt

from utils.x1_path_utils import resolve_path

data = np.load(resolve_path('Data/train_data_26_100_3.npy'))

z_threshold = -0.38

x_all = data[:, :, 0]
y_all = data[:, :, 1]
x_min, x_max = x_all.min(), x_all.max()
y_min, y_max = y_all.min(), y_all.max()

fig, axes = plt.subplots(5, 6, figsize=(18, 12))
axes = axes.flatten()

for i in range(26):
    ax = axes[i]
    x = data[i, :, 0]
    y = -data[i, :, 1]  
    z = data[i, :, 2]

    # Z < -0.38 threshold (table hight)
    mask = z < z_threshold
    x_filtered = x[mask]
    y_filtered = y[mask]

    ax.plot(y_filtered, x_filtered)
    ax.set_xlim(-y_max, -y_min)
    ax.set_ylim(x_min, x_max)
    ax.axis('off')

for j in range(26, len(axes)):
    axes[j].axis('off')

fig.suptitle("2D Trajectories Filtered by Z < -0.38", fontsize=16)
fig.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()
