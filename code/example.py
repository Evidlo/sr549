#!/bin/env python3
# Evan Widloski - 2020-05-09

import numpy as np
import matplotlib.pyplot as plt
from sr549.forward import Forward, add_noise
from sr549.data import scene
from sr549.registration import registration, shift_and_sum

# %% forward

f = Forward(
    drift_angle=45,
    drift_velocity=1,
    frame_rate=4,
    num_frames=40,
    lr_size=(100, 100),
    hr_size=(500, 500),
    factor=4
)

# %% register

# crop scene to proper input size
scene = scene[:f.hr_size[0], :f.hr_size[1]]

# propagate scene through system and reshape
frames = f.forward @ scene.flatten()
frames = frames.reshape(f.num_frames, *f.lr_size)

# add noise to frames
frames_noisy = add_noise(frames, dbsnr=-10)

est_drift = registration(frames) * f.factor
print('true_drift:', f.true_drift, 'px')
print('est_drift:', est_drift, 'px')
print('err:', np.linalg.norm(est_drift - f.true_drift), 'px')

recon = shift_and_sum(frames, f.true_drift / f.factor)

# %% plot

plt.close()
plt.subplot(1, 3, 1)
plt.imshow(frames[0])
plt.title('Clean Frame')
plt.subplot(1, 3, 2)
plt.imshow(frames_noisy[0])
plt.title('Noisy Frame')
plt.subplot(1, 3, 3)
plt.imshow(recon)
plt.title('Coadded Registered Frames')

plt.tight_layout()
plt.show()
