from numpy.fft import fftshift as ffts

# ----- Evan Figure Generation -----

# frames
for n in (0, 1, 2, 3,  len(frames) - 1):
    plt.imsave(f'frame{n}.png', frames[n])
    plt.imsave(f'frame_noisy{n}.png', frames_noisy[n])

# phase correlations
for time_diff in (1, 21):
    frames_freq = np.fft.fftn(frames_noisy, axes=(1, 2))
    products = frames_freq[:-time_diff] * frames_freq[time_diff:].conj()
    # CC PC comparison
    pcs = np.fft.ifftn(products / np.abs(products), axes=(1, 2)).real
    ccs = np.fft.ifftn(products, axes=(1, 2)).real
    for n in (0, 1, 2, len(products) - 1):
        plt.imsave(f'pc{time_diff}-{n}.png', crop(ffts(pcs[n]), width=64))
        plt.imsave(f'cc{time_diff}-{n}.png', crop(ffts(ccs[n]), width=64))

# guizar diagram
plt.imsave(f'cc{time_diff}-0-zoom.png', rescale(ccs[0, :8, :8], scale=2))

corr_sum = correlate_and_sum(frames_noisy, mode='PC')
corr_sum_cc = correlate_and_sum(frames_noisy, mode='CC')

# corrsums
for n in (0, 10, 20, 30, 37):
    plt.imsave(f'corrsum{n}.png', ffts(np.abs(corr_sum[n])[:64, :64]))

plt.figure(figsize=(5, 2))
plt.plot(ffts(pcs[0, 3]) / np.max(pcs[0, 3]))
plt.savefig(f'pc{time_diff}-1D.png', dpi=200)
plt.close()
plt.figure(figsize=(5, 2))
plt.plot(ffts(ccs[0, 3]) / np.max(ccs[0, 3]))
plt.savefig(f'cc{time_diff}-1D.png', dpi=200)
plt.close()
