import gstools as gs
from matplotlib import pyplot as plt
import numpy as np


# x = y = range(100)
x = np.arange(0, 100, 1)
y = np.arange(0, 100, 1)
xx, yy = np.meshgrid(x, y, indexing="ij")
x = xx[:, 0]
y = yy[0, :]
# print(x)
# Create the covariance model
n = 5
seed = 20170519
fields = []
for len_scale in range(n):
    model = gs.Gaussian(
        dim=2,
        len_scale=len_scale + 0.0001,
    )

    srf = gs.SRF(model, seed=seed)

    srf.structured([x, y])
    fie = srf.transform("binary")
    fie_binary = np.where(fie == -1.0, 0.0, fie)
    fields.append(fie)
print(fields[0].shape)
srf.plot()
# plot
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(n):
    ax[i].imshow(fields[i], origin="lower")
plt.show()
