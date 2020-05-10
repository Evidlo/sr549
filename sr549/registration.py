import numpy as np
from sr549.misc import np_gpu, downsample2, roll
from tqdm import tqdm
from skimage.transform import rescale
import numpy.ma as ma

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


def shift_and_sum(frames, drift, mode='full', shift_method='roll'):
    """Coadd frames by given shift

    Args:
        frames (ndarray): input frames to coadd
        drift (ndarray): drift between adjacent frames
        mode (str): zeropad before coadding ('full') or crop to region of
            frame overlap ('crop')
        shift_method (str): method for shifting frames ('roll', 'fourier')
        pad (bool): zeropad images before coadding

    Returns:
        (ndarray): coadded images
    """

    pad = np.ceil(drift * (len(frames) - 1)).astype(int)
    pad_r = (0, pad[0]) if drift[0] > 0 else (-pad[0], 0)
    pad_c = (0, pad[1]) if drift[1] > 0 else (-pad[1], 0)
    frames_ones = np.pad(
        np.ones(frames.shape),
        ((0, 0), pad_r, pad_c),
        mode='constant'
    )
    frames = np.pad(frames, ((0, 0), pad_r, pad_c), mode='constant')

    summation = np.zeros(frames[0].shape, dtype='complex128')
    summation_scale = np.copy(summation)

    # import ipdb
    # ipdb.set_trace()

    for time_diff, (frame, frame_ones) in enumerate(zip(frames, frames_ones)):
        shift = np.array(drift) * (time_diff + 1)
        if shift_method == 'roll':
            integer_shift = np.round(shift).astype(int)
            shifted = roll(frame, (integer_shift[0], integer_shift[1]))
            shifted_ones = roll(frame_ones, (integer_shift[0], integer_shift[1]))
        elif shift_method == 'fourier':
            shifted = np.fft.ifftn(fourier_shift(
                np.fft.fftn(frame),
                (shift[0], shift[1])
            ))
            shifted_ones = np.fft.ifftn(fourier_shift(
                np.fft.fftn(frame_ones),
                (shift[0], shift[1])
            ))
        else:
            raise Exception('Invalid shift_method')
        summation += shifted
        summation_scale += shifted_ones

    if mode == 'crop':
        summation = size_equalizer(
            summation,
            np.array(frames[0].shape).astype(int) -
            2 * np.ceil(drift * (len(frames)-1)).astype(int)
        )
    elif mode == 'full':
        summation /= ma.masked_where(summation_scale == 0, summation_scale)
    else:
        raise Exception('Invalid mode')

    return summation.real


def registration(frames):
    """
    Compute drift vector

    Args:
        frames (ndarray): frames to register (frame #, frame width, frame height)

    Returns:
        tuple: interframe drift vector on HR grid
    """

    # %% coarse

    corr_sum = correlate_and_sum(frames, mode='PC')

    max_timediff = len(corr_sum)
    scale_factor = np.array(corr_sum[0].shape) // len(corr_sum)

    coarse_list = []
    for n, cs in enumerate(corr_sum, 1):
        coarse_list.append(
            downsample2(
                cs[:scale_factor[0] * n, :scale_factor[1] * n],
                n
            )
        )

    coarse = abs(np.array(coarse_list).sum(axis=0))

    coarse_est = np.unravel_index(np.argmax(coarse), coarse.shape)

    # %% fine

    fine_list = []
    fine_argmaxes = []
    # FIXME - corr_sum must be real for rescale()
    corr_sum = abs(corr_sum)
    cropped = []
    for n, cs in enumerate(corr_sum, 1):
        sfx, sfy = scale_factor
        c = cs[:sfx * n, :sfy * n]
        cropped.append(c)
        rescaled = rescale(
            c,
            float(max_timediff) / n)[
                :sfx * max_timediff,
                :sfy * max_timediff
            ]
        fine_list.append(rescaled)
        fine_argmaxes.append(np.unravel_index(np.argmax(rescaled), rescaled.shape))

    fine = np.array(fine_list)

    weights = np.arange(1, max_timediff + 1)**2
    weights = weights / weights.sum()
    weights[-2:] = 0

    result = np.sum(weights[:, np.newaxis, np.newaxis] * fine, axis=0)

    fine_est = np.array(np.unravel_index(np.argmax(result), result.shape)) / max_timediff

    # import ipdb
    # ipdb.set_trace()

    return fine_est
