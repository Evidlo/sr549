# -*- coding: utf-8 -*-
"""
Created on Sun May 10 06:52:20 2020

@author: aditya, akshayaa
"""

#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sr549.forward import Forward, add_noise
from sr549.data import scene
from sr549.registration import registration, shift_and_sum
from scipy import ndimage
from scipy.sparse import csc_matrix
from scipy.sparse import identity, block_diag
from scipy import signal
from scipy.optimize import minimize
from scipy import interpolate
from sr549.misc import crop, compare_ssim, compare_psnr
#def interpol(img, factor):
#    
#    x = np.arange(0, np.shape(img)[0])*4
#    y = np.arange(0, np.shape(img)[1])*4
#    f = interpolate.interp2d(x,y,img, kind = 'cubic')
#    x_new = np.arange(0, np.shape(img)[0]*factor)
#    y_new = np.arange(0, np.shape(img)[1]*factor)
#    interp_img = f(x_new, y_new)
#    return interp_img

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
frames = 1/f.factor**2 * f.forward @ scene.flatten()
frames = frames.reshape(f.num_frames, *f.lr_size)

# add noise to frames
frames_noisy = add_noise(frames, dbsnr= -10)

est_drift = registration(frames) * f.factor
print('true_drift:', f.true_drift, 'px')
print('est_drift:', est_drift, 'px')
print('err:', np.linalg.norm(est_drift - f.true_drift), 'px')

# %% HMRF
A = f.forward/f.factor**2
A0 = A[:10000,:]
A_p = A[10000:,]
y = frames_noisy.flatten()
y0 = y[:10000]
y_p = y[10000:]
q = f.factor
niter = 100
alpha = 10
beta = 1
P = identity(250000)-q**2 * A0.T @ A0
z = q**2 * A0.T @ y0
N = 500

I = identity(10000)
t=()
for i in range(1,f.num_frames):
   t = t + (np.sqrt(beta/i)*I,)
L = block_diag(t)

y_p = L @ y_p
A_p = L @ A_p

def interpol(img, factor):
    
    x = np.arange(0, np.shape(img)[0])*4
    y = np.arange(0, np.shape(img)[1])*4
    f = interpolate.interp2d(x,y,img, kind = 'cubic')
    x_new = np.arange(0, np.shape(img)[0]*factor)
    y_new = np.arange(0, np.shape(img)[1]*factor)
    interp_img = f(x_new, y_new)
    return interp_img
  

def block_avg(I,q):
    m = np.shape(I)[0]
    n = np.shape(I)[1]
    down_I = np.zeros((int(m/q), int(n/q)))
    for i in range(0,int(m/q)):
        for j in range(0,int(n/q)):
            Xs = i*q, i*q + q
            Ys = j*q, j*q + q
            down_I[i][j] = np.mean(I[slice(*Xs), slice(*Ys)])
    return down_I

def block_kron(I, q):
    id = np.ones((q, q))
    up_I = np.kron(I, id)
    return up_I/q**2

def image_shift(I, x_shift, y_shift):
    return ndimage.shift(I, (x_shift,y_shift))

def rho_dot(x):
    y = x
    y[np.abs(x)<=alpha] *= 2 
    y[x < -alpha] = -2*alpha
    y[x > alpha] = 2*alpha
    return y
    
def d_dot_z(m,n,r):
    if r==1:
        return z[m*N+n-1]-2*z[m*N+n]+z[m*N+n+1]
    elif r==2:
        return 0.5*z[(m+1)*N+n-1]-z[m*N+n]+0.5*z[(m-1)*N+n+1]
    elif r==3:
        return z[(m-1)*N+n]-2*z[m*N+n]+z[(m+1)*N+n]
    else:
        return 0.5*z[(m-1)*N+n-1]-z[m*N+n]+0.5*z[(m+1)*N+n+1]
    
    
def rho(x):
    y = x
    y[np.abs(x)<=alpha] = y[np.abs(x)<=alpha]**2
    y[x < -alpha] = -2*alpha*x -alpha**2
    y[x > alpha] = 2*alpha*x - alpha**2
    return y

def my_objective(z):
    d1z = signal.convolve2d(z.reshape(N,N), np.array([[1, -2, 1]]), mode='same',boundary = 'symm')
    d2z = signal.convolve2d(z.reshape(N,N), np.array([[0, 0, 0.5],[0,-1,0],[0.5,0,0]]), mode='same',boundary = 'symm')
    d3z = signal.convolve2d(z.reshape(N,N), np.array([[1, -2, 1]]).T, mode='same',boundary = 'symm')
    d4z = signal.convolve2d(z.reshape(N,N), np.array([[0.5, 0, 0],[0,-1,0],[0,0,0.5]]), mode='same',boundary = 'symm')
    
    return np.sum(d1z) + np.sum(d2z) + np.sum(d3z) + np.sum(d4z) + np.linalg.norm(y_p-A_p @ z)

def my_gradient(z):
    d1z = signal.convolve2d(z.reshape(N,N), np.array([[1, -2, 1]]), mode='same',boundary = 'symm')
    d2z = signal.convolve2d(z.reshape(N,N), np.array([[0, 0, 0.5],[0,-1,0],[0.5,0,0]]), mode='same',boundary = 'symm')
    d3z = signal.convolve2d(z.reshape(N,N), np.array([[1, -2, 1]]).T, mode='same',boundary = 'symm')
    d4z = signal.convolve2d(z.reshape(N,N), np.array([[0.5, 0, 0],[0,-1,0],[0,0,0.5]]), mode='same',boundary = 'symm')
    
    d1z = rho_dot(d1z)
    d2z = rho_dot(d2z)
    d3z = rho_dot(d3z)
    d4z = rho_dot(d4z)
    #print(np.sum(d1z) + np.sum(d2z) + np.sum(d3z) + np.sum(d4z) + np.linalg.norm(y_p-A_p @ z))
    d1z = signal.convolve2d(d1z, np.array([[1, -2, 1]]), mode='same',boundary = 'symm')
    d2z = signal.convolve2d(d2z, np.array([[0, 0, 0.5],[0,-1,0],[0.5,0,0]]), mode='same',boundary = 'symm')
    d3z = signal.convolve2d(d3z, np.array([[1, -2, 1]]).T, mode='same',boundary = 'symm')
    d4z = signal.convolve2d(d4z, np.array([[0.5, 0, 0],[0,-1,0],[0,0,0.5]]), mode='same',boundary = 'symm')
    
    g = d1z.flatten()+d2z.flatten()+d3z.flatten()+d4z.flatten()
    g = g - 2*A_p.T @ (y_p-A_p @ z)
    return g



# %% PROJECTED GRADIENT DESCENT

z = interpol(frames[0],5)
z = z.flatten()
#z = q**2 A0.T @ y0

for i in range(niter):
    print(i)    
    g = my_gradient(z)   
    p = P @ g
    z = z - 0.0001*p
plt.imshow(z.reshape(N,N))


# %% EVALUATION
scene_norm = scene[:f.lr_size[0] * f.factor, :f.lr_size[1] * f.factor]
scene_norm = scene_norm / scene_norm.max()
z = z.reshape(N,N)
my_recon = z[:400,:400]
my_recon = my_recon / my_recon.max()
print('PSNR:', compare_psnr(truth=scene_norm, estimate=my_recon))
print('SSIM:', compare_ssim(truth=scene_norm, estimate=my_recon))

# %% ML recon
#import scipy
#z_ml = scipy.sparse.linalg.inv(A.T @ A) @ A.T @ y