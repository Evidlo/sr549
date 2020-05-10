import numpy as np
import matplotlib.pyplot as plt
from sr549.forward import Forward, add_noise
from sr549.data import scene
from sr549.registration import registration, shift_and_sum
from sr549.misc import crop, compare_ssim, compare_psnr, upsample
from sr549.hmrf import hmrf
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

# %%
scene = scene[:f.hr_size[0], :f.hr_size[1]]
frames = f.forward @ scene.flatten()
frames = frames.reshape(f.num_frames, *f.lr_size)

# add noise to frames
frames_noisy = add_noise(frames, dbsnr=-10)

# %% Output is normalized. size:lr_size * factor
z = hmrf(frames_noisy,f,dbsnr=-10)