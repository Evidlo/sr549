#!/bin/env python3
# Evan Widloski - 2020-05-09

import numpy as np
import matplotlib.pyplot as plt
from sr549.forward import forward
from sr549.data import scene

# %% weights

hr_size = (500, 500)
lr_size = (50, 50)
num_frames = 10

scene = scene[:500, :500]

w = forward(
    drift_angle=45,
    drift_velocity=3,
    frame_rate=4,
    num_frames=num_frames,
    lr_size=lr_size,
    hr_size=hr_size,
    factor=4
)

# %% frames

frames = w @ scene.flatten()

frames = frames.reshape(num_frames, *lr_size)

plt.imshow(frames[0])
