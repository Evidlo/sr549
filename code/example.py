#!/bin/env python3
# Evan Widloski - 2020-05-09

import numpy as np
import matplotlib.pyplot as plt
from sr549.forward import Forward, add_noise
from sr549.data import scene
from sr549.registration import registration, shift_and_sum
from sr549.misc import crop, compare_ssim, compare_psnr, upsample
from skimage.transform import rescale

# %% forward

f = Forward(
    drift_angle=45,
    drift_velocity=1,
    frame_rate=4,
    num_frames=40,
    lr_size=np.array((100, 100)),
    hr_size=np.array((500, 500)),
    factor=4
)

# %% register

# crop scene to proper input size
scene = scene[:f.hr_size[0], :f.hr_size[1]]

# propagate scene through system and reshape
frames = f.forward @ scene.flatten()
frames = frames.reshape(f.num_frames, *f.lr_size)
# FIXME: weight matrix is transposed
frames = frames.transpose(0, 2, 1)

# add noise to frames
frames_noisy = add_noise(frames, dbsnr=-10)

est_drift = registration(frames_noisy) * f.factor

recon = shift_and_sum(
    upsample(frames_noisy, factor=4),
    est_drift,
    mode='first'
)

# ----- Evaluation -----
# %% eval

# drift error
print('true_drift:', f.true_drift)
print('est_drift:', est_drift)
print('err:', np.linalg.norm(est_drift - f.true_drift))

# crop and normalize recon and scene
recon_norm = recon / recon.max()
scene_norm = scene[:f.lr_size[0] * f.factor, :f.lr_size[1] * f.factor]
scene_norm = scene_norm / scene_norm.max()

print('PSNR:', compare_psnr(truth=scene_norm, estimate=recon_norm))
print('SSIM:', compare_ssim(truth=scene_norm, estimate=recon_norm))

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
