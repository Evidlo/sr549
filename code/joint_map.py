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
    
# %%    
N = 500
M = 100
hr_size = (N, N)
lr_size = (M, M)
num_frames = 40

scene = scene[:500, :500]

f = Forward(
    drift_angle=45,
    drift_velocity=1,
    frame_rate=4,
    num_frames= num_frames,
    lr_size=(M, M),
    hr_size=(N, N),
    factor=4
)

# crop scene to proper input size
scene = scene[:f.hr_size[0], :f.hr_size[1]]

# propagate scene through system and reshape
frames = (f.forward @ scene.flatten())/16.0
frames = frames.reshape(f.num_frames, *f.lr_size) 

# %%
frames_noisy = add_noise(frames, dbsnr=-10)
frames = frames_noisy



# %%

iter = 10
lamda = 100
sig = np.sqrt(1000)
z = interpol(frames[0], 5)
z = z.T

# %%
w_est = Forward(
        drift_angle=45,
        drift_velocity=1,
        frame_rate=4,
        num_frames= num_frames,
        lr_size=(M, M),
        hr_size=(N, N),
        factor=4
    )
w = w_est.forward/16

# %%

for i in range(500):
    #drift = find_shift(frames, z, N, M, num_frames)
    

    g = find_g(w, z, frames, sig, lamda)        
    step = compute_step(w, z, N, M, sig, lamda, frames)
    z = z.flatten() - step*g
    z = z.reshape(N,N)
    plt.imshow(z)
    plt.title('%i' %i)

# %%

est_drift = registration(frames) * f.factor
print('true_drift:', f.true_drift, 'px')
print('est_drift:', est_drift, 'px')
print('err:', np.linalg.norm(est_drift - f.true_drift), 'px')

recon = shift_and_sum(frames, f.true_drift / f.factor)

# %%

scene_norm = scene[:f.lr_size[0] * f.factor, :f.lr_size[1] * f.factor]
scene_norm = scene_norm / scene_norm.max()
recon_norm = z[:400, :400]
recon_norm = recon_norm/recon_norm.max()

print('PSNR:', compare_psnr(truth=scene_norm, estimate=recon_norm))
print('SSIM:', compare_ssim(truth=scene_norm, estimate=recon_norm))