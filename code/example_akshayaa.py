import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from sr549.forward import Forward
from sr549.data import scene
from scipy import ndimage
from sr549.forward import Forward, add_noise
from sr549.registration import registration, shift_and_sum
from sr549.misc import crop, compare_ssim, compare_psnr
from sr549.MAP_est import MAP_est


num_frames = 40
M = 100
N = 500
f = Forward(
    drift_angle=45,
    drift_velocity=1,
    frame_rate=4,
    num_frames= num_frames,
    lr_size=(M, M),
    hr_size=(N, N),
    factor=4
)
scene = scene[:500, :500]
# crop scene to proper input size
scene = scene[:f.hr_size[0], :f.hr_size[1]]

# propagate scene through system and reshape
frames = (f.forward @ scene.flatten())/16.0
frames = frames.reshape(f.num_frames, *f.lr_size) 

# %%
frames_noisy = add_noise(frames, dbsnr= -10)

# Output is 400 x 400 normalized 
recon_norm = MAP_est(frames_noisy, f, -10)
scene_norm = scene[:f.lr_size[0] * f.factor, :f.lr_size[1] * f.factor]
scene_norm = scene_norm / scene_norm.max()

print('PSNR:', compare_psnr(truth=scene_norm, estimate=recon_norm))
print('SSIM:', compare_ssim(truth=scene_norm, estimate=recon_norm))