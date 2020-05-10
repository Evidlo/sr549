from sr549.misc import np_gpu, roll, store_kwargs, vectorize
from skimage.draw import line
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, vstack
from cachalot import Cache
from tqdm import tqdm

class Forward(object):

    @store_kwargs
    def __init__(
            self, drift_angle=45, drift_velocity=1, frame_rate=4, num_frames=40,
            lr_size=(50, 50), hr_size=(500, 500), factor=4, start=(0, 0),
            motion_blur=True,
        ):
        """Generate forward matrix

        Args:
            drift_angle (float): direction of drift in image coordinate system (deg)
            drift_velocity (float): velocity of drift in HR coordinates (pix / s)
            frame_rate (float): frames per second
            num_frames (int): total number of frames
            lr_size (int): width of LR image
            hr_size (int): width of HR image
            factor (int): downsample factor
            start (tuple): top left point of first LR image (in HR coordinate system)
        """

        self.exp_time = num_frames / frame_rate
        self.true_drift = np.array((
            np.cos(np.deg2rad(drift_angle)),
            np.sin(np.deg2rad(drift_angle)),
        )) * drift_velocity / frame_rate

        # ----- motion kernel start points -----

        start = np.array(start)
        # top left point of last frame
        total_dx, total_dy = self.true_drift * num_frames
        end = start + (total_dx, total_dy)

        # import ipdb
        # ipdb.set_trace()

        assert (
            0 <= start[0] <= hr_size[0] and 0 <= start[1] <= hr_size[1]
        ), "start coordinates out of bounds of image"
        assert (
            0 <= (end[0] + factor * lr_size[0]) <= hr_size[0] and
            0 <= (end[1] + factor * lr_size[1]) <= hr_size[1]
        ), "end coordinates out of bounds of image"

        points = np.linspace(start, end, num_frames + 1)
        starts = np.round(points[:-1]).astype(int)
        # ends = np.round(points[1:]).astype(int)

        # ----- create motion kernel -----

        kernel = np.zeros(starts[1] - starts[0] + (factor, factor))
        if motion_blur:
            line_points = line(*starts[0], *starts[1])
        else:
            line_points = line(*starts[0], *starts[0])
        for r, c in zip(*line_points):
            kernel[r:r + factor, c:c + factor] = np.ones((factor, factor))

        kernel_r, kernel_c = np.where(kernel)

        # ----- copy motion kernel to all positions -----

        rows = []
        for sc, sr in tqdm(starts, desc='frame'):
            for c in np.arange(0, factor * lr_size[1], factor):
                for r in np.arange(0, factor * lr_size[0], factor):
                    pixel_segments = csr_matrix(
                        (
                            np.ones(len(kernel_r)),
                            (kernel_r + r + sr, kernel_c + c + sc)
                        ),
                        shape=hr_size
                    )
                    rows.append(pixel_segments.reshape(1, -1))

        self.forward = vstack(rows)


@vectorize
def add_noise(signal, dbsnr=None):
    """
    Add noise to the given signal at the specified level.

    Args:
        signal (ndarray): noise-free input signal
        dbsnr (float): signal to noise ratio in dB: for Gaussian noise model, it is
            defined as the ratio of variance of the input signal to the variance of
            the noise. For Poisson model, it is taken as the average snr where snr
            of a pixel is given by the square root of its value.
        max_count (int): Max number of photon counts in the given signal

    Returns:
        ndarray: noisy version of input
    """
    if dbsnr is None:
        return signal
    else:
        var_sig = np.var(signal)
        var_noise = var_sig / 10**(dbsnr / 10)
        return np.random.normal(loc=signal, scale=np.sqrt(var_noise))
