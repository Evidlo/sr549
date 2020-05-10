# Call MAP_est(frames, f, dbsnr)

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from sr549.forward import Forward
from sr549.data import scene
from scipy import ndimage
from sr549.forward import Forward, add_noise
from sr549.registration import registration, shift_and_sum
from sr549.misc import crop, compare_ssim, compare_psnr

def DownSample(img, factor):
    m = np.shape(img)[0]
    n = np.shape(img)[1]
    down_img = np.zeros((int(m/factor), int(n/factor)))
    for i in range(0,int(m/factor)):
        for j in range(0,int(n/factor)):
            Xs = i*factor, i*factor + factor
            Ys = j*factor, j*factor + factor
            down_img[i][j] = np.mean(img[slice(*Xs), slice(*Ys)])
    return down_img

def UpSample(img, factor):
    id = np.ones((factor, factor))
    up_img = np.kron(img, id)
    return up_img/factor**2

# %%
def interpol(img, factor):
    
    x = np.arange(0, np.shape(img)[0])*4
    y = np.arange(0, np.shape(img)[1])*4
    f = interpolate.interp2d(x,y,img, kind = 'cubic')
    x_new = np.arange(0, np.shape(img)[0]*factor)
    y_new = np.arange(0, np.shape(img)[1]*factor)
    interp_img = f(x_new, y_new)
    return interp_img
# %%
def find_shift(frames, z, N , M, num_frames):
    max_vel = 5
    min_vel = 1
    w = Forward(
        factor = 4, hr_size = (N,N), lr_size = (M,M), 
        num_frames = num_frames, frame_rate = 4, 
        drift_velocity = min_vel, drift_angle = 0)
    min_cost =  np.linalg.norm(frames.flatten() - w.forward/16.0 @ z.flatten())**2
    vels = np.linspace(1, max_vel, 5)
    for vel in vels:
        print(vel)
        w = Forward(
        factor = 4, hr_size = (N,N), lr_size = (M,M), 
        num_frames = num_frames, frame_rate = 4, 
        drift_velocity = vel, drift_angle = 0)
        cost = np.linalg.norm(frames.flatten() - w.forward/16.0 @ z.flatten())**2
        print(cost)
        if cost < min_cost:
            min_cost = cost
            min_vel = vel
    
    return min_vel

# %%

def find_kernel(lamda):
    kernel = np.zeros((5,5))
    kernel[0,2] = 1.0/(16*lamda)
    kernel[2,0] = 1.0/(16*lamda)
    kernel[2,4] = 1.0/(16*lamda)
    kernel[4,2] = 1.0/(16*lamda)
    kernel[1,1] = 1.0/(8*lamda)
    kernel[1,3] = 1.0/(8*lamda)
    kernel[3,1] = 1.0/(8*lamda)
    kernel[3,3] = 1.0/(8*lamda)
    kernel[1,2] = -1.0/(2*lamda)
    kernel[2,1] = -1.0/(2*lamda)
    kernel[2,3] = -1.0/(2*lamda)
    kernel[3,2] = -1.0/(2*lamda)
    kernel[2,2] = 5.0/(4*lamda)
    return kernel


# %%

def find_g(w, z, frames, sig, lamda):
    pred_err =  (1.0/sig**2)*(w.T @ (w @ z.flatten() - frames.flatten())) 
    im_grad = ndimage.filters.convolve(z, find_kernel(lamda))
    g = pred_err + im_grad.flatten()
    return g

# %%
def compute_step(w, z, N, M, sig, lamda, frames):
    
    kernel = find_kernel(lamda)
    g = find_g(w,z,frames,sig,lamda)
    im_grad = ndimage.filters.convolve(z, kernel)
    num_1 = (1/sig**2)*((w @ g) @ (w @ z.flatten() - frames.flatten()))
    num_2 = g @ im_grad.flatten()
    g_grad = ndimage.filters.convolve(np.reshape(g,(N,N)), kernel)
    den_2 = g @ g_grad.flatten()
    proj_grad = w @ g
    den_1 = (1/sig**2)*(proj_grad @ proj_grad)
    step = (num_1 + num_2)/(den_1 + den_2)
    return step


def MAP_est(frames, f, snr):
    w = f.forward/(f.factor**2)
    N = int(np.sqrt(np.shape(w)[1]))
    M = np.shape(frames)[1]    
    hr_size = (N, N)
    lr_size = (M, M)
    num_frames = np.shape(frames)[0]
    iter = 500
    lamda = 200
    sig = np.sqrt(100*10**(-snr/10))
    z = interpol(frames[0], N/M)
    #z = z.T
    for i in range(iter):
    #drift = find_shift(frames, z, N, M, num_frames)
        g = find_g(w, z, frames, sig, lamda)        
        step = compute_step(w, z, N, M, sig, lamda, frames)
        z = z.flatten() - step*g
        z = z.reshape(N,N)
    
    recon_norm = z[:f.lr_size[0] * f.factor, :f.lr_size[1] * f.factor]
    recon_norm = recon_norm/recon_norm.max()
    return recon_norm