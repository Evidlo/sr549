import numpy as np

@np_gpu(np_args=[0])
def correlate_and_sum(frames, mode='CC', np=np):
    """Correlate all frame combinations and sum each group

    Args:
        frames (ndarray): input images
        mode (str, default='PC'): type of correlation to use. ('PC', 'CC')

    Returns:
        ndarray: axes (group, corr_x_coord, corr_y_coord)
    """

    frames_freq = np.fft.fftn(frames, axes=(1, 2))

    product_sums = np.zeros(
        (len(frames) - 1, frames.shape[1], frames.shape[2]),
        dtype='complex128'
    )
    for time_diff in tqdm(range(1, len(frames_freq)), desc='Correlation', leave=None):
        products = frames_freq[:-time_diff] * frames_freq[time_diff:].conj()
        if mode.upper() == 'PC':
            product_sums[time_diff - 1] = np.sum(products / np.abs(products), axis=0)
        elif mode.upper() == 'CC':
            product_sums[time_diff - 1] = np.sum(products, axis=0)
        else:
            raise Exception('Invalid mode {}'.format(mode.upper()))

    return np.fft.ifftn(np.array(product_sums), axes=(1, 2))


def roll(x, shift):
    shift = np.round(shift).astype(int)
    return np.roll(
        np.roll(
            x,
            shift[0],
            axis=0
        ),
        shift[1],
        axis=1
    )

def shift_and_sum(frames, drift, mode='full', shift_method='roll'):
    """Coadd frames by given shift

    Args:
        frames (ndarray): input frames to coadd
        drift (ndarray): drift between adjacent frames (cartesian)
        mode (str): zeropad before coadding ('full') or crop to region of
            frame overlap ('crop')
        shift_method (str): method for shifting frames ('roll', 'fourier')
        pad (bool): zeropad images before coadding

    Returns:
        (ndarray): coadded images
    """

    pad = np.ceil(drift * (len(frames) - 1)).astype(int)
    pad_x = (0, pad[0]) if drift[0] > 0 else (-pad[0], 0)
    pad_y = (pad[1], 0) if drift[1] > 0 else (0, -pad[1])
    frames = np.pad(frames, ((0, 0), pad_y, pad_x), mode='constant')

    summation = np.zeros(frames[0].shape, dtype='complex128')

    for time_diff, frame in enumerate(frames):
        shift = np.array(drift) * (time_diff + 1)
        if shift_method == 'roll':
            integer_shift = np.round(shift).astype(int)
            shifted = roll(frame, (-integer_shift[1], integer_shift[0]))
        elif shift_method == 'fourier':
            shifted = np.fft.ifftn(fourier_shift(
                np.fft.fftn(frame),
                (-shift[1], shift[0])
            ))
        else:
            raise Exception('Invalid shift_method')
        summation += shifted

    if mode == 'crop':
        summation = size_equalizer(
            summation,
            np.array(frames[0].shape).astype(int) -
            2 * np.ceil(xy2rc(drift) * (len(frames)-1)).astype(int)
        )
    elif mode == 'full':
        pass
    else:
        raise Exception('Invalid mode')

    return summation.real
