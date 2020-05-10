import numpy as np
from skimage.draw import line
from skimage.transform import rescale
from scipy.ndimage import rotate
from tqdm import tqdm
from sr549.misc import crop, vectorize


class Video(object):

    """A class for holding video state

    Args:
        scene (ndarray): simulation high-resolution image
        exp_time (int): total experiment time (s)
        drift_angle (float): angle of drift (degrees)
        drift_velocity (float): velocity of drift (m / s)
        max_count (int): maximum photon rate
        noise_model (function): function which takes 'frames', 'frame_rate',
            'dark_current', 'background', 'read_noise' and returns noisy frames

        ccd_size (tuple): resolution of physical CCD

        resolution_ratio (int): subpixel scale factor
            (ccd pixel size / simulation pixel size)
        fov_ratio (int): (simulation FOV / CCD FOV)

        num_strands (int): number of nanoflares

        diameter (float): sieve diameter (m)
        smallest_hole_diameter (float): smallest sieve hole diam (m)
    """

    def __init__(
            self,
            scene=None, # HR image
            # experiment parameters
            exp_time=10, # s
            drift_angle=-45, # degrees
            drift_velocity=4, # pix / s
            angle_velocity=0, # deg / s
            noise_level=-10, # dB SNR
            # CCD parameters
            frame_rate=4, # Hz
            ccd_size=np.array((100, 100)),
            start=(500, 500),
            # simulation subpixel parameters
            resolution_ratio=2, # HR -> LR downsample factor
            fov_ratio=2, # simulated FOV / CCD FOV
    ):

        self.scene = scene
        self.exp_time = exp_time
        self.drift_angle = drift_angle
        self.drift_velocity = drift_velocity
        self.angle_velocity = angle_velocity
        self.noise_level = noise_level
        self.frame_rate = frame_rate
        self.ccd_size = ccd_size
        self.start = start
        self.resolution_ratio = resolution_ratio
        self.fov_ratio = fov_ratio

        # use provided scene or load default
        if scene is not None:
            scene = np.copy(scene)
        else:
            from sr549.data import scene

        # apply blur to scene
        # FIXME

        self.frames_clean, self.topleft_coords = video(
            scene=scene,
            frame_rate=frame_rate,
            exp_time=exp_time,
            drift_angle=drift_angle,
            drift_velocity=drift_velocity,
            angle_velocity=angle_velocity,
            ccd_size=ccd_size,
            resolution_ratio=resolution_ratio,
            start=start
        )

        # add noise to the frames
        self.frames = add_noise(self.frames_clean, noise_level)

        self.true_drift = drift_velocity / frame_rate * np.array([
            np.cos(np.deg2rad(drift_angle)),
            np.sin(np.deg2rad(drift_angle)) # use image coordinate system
        ])


def video(*, scene, resolution_ratio, frame_rate, exp_time, drift_angle,
          drift_velocity, angle_velocity, ccd_size, start,):

    """
    Get video frames from input scene, applying appropriate motion blur

    Args:
        scene (ndarray): high resolution input scene
        resolution_ratio (float): downsample factor to low resolution images
        frame_rate (float): video frame rate
        exp_time (float): experiment duration
        drift_angle (float): linear drift direction (deg)
        drift_velocity (float): linear drift velocity (pix / s)
        angle_velocity (float): camera rotation rate (deg / s)
        ccd_size (int): size of square detector ccd (pixels)
        start (tuple): start location of detector in scene
    """

    num_frames = exp_time * frame_rate

    def coord(k):
        return np.array((
            start[0] - k * drift_velocity * np.sin(np.deg2rad(drift_angle)) *
            resolution_ratio / (frame_rate),
            start[1] + k * drift_velocity * np.cos(np.deg2rad(drift_angle)) *
            resolution_ratio / (frame_rate)
        )).astype(int).T

    # FIXME check box bounds correctly, need to account for rotation
    assert (
        0 <= coord(0)[0] < scene.shape[0] and 0 <= coord(0)[1] < scene.shape[1] and
        0 <= coord(num_frames)[0] < scene.shape[0] and 0 <= coord(num_frames)[1] < scene.shape[1]
    ), f"Frames drift outside of scene bounds \
    ({coord(0)[0]}, {coord(0)[1]}) -> ({coord(num_frames)[0]}, {coord(num_frames)[1]})"

    # calculate the middle points for all frames
    mid = coord(np.arange(num_frames + 1))

    # initialize frame images
    frames = np.zeros((num_frames, ccd_size[0], ccd_size[1]))

    # calculate each frame by integrating high resolution image along the drift
    # direction
    for frame in tqdm(range(num_frames), desc='Frames', leave=None, position=1):
        hr_size = np.array(ccd_size) * resolution_ratio
        hr_frame = np.zeros(hr_size)
        # calculate middle coordinates for the shortest line connecting the
        # middle coordinates of the consecutive frames
        path_rows, path_cols = line(
            mid[frame][0],
            mid[frame][1],
            mid[frame+1][0],
            mid[frame+1][1]
        )
        total_rotation = exp_time * angle_velocity
        angles = total_rotation * np.sqrt((path_rows - mid[0][0])**2 + (path_cols - mid[0][1])**2) / np.linalg.norm(mid[-1] - mid[0])
        if len(path_rows) > 1:
            path_rows, path_cols = path_rows[:-1], path_cols[:-1]
        for row, col, angle in zip(path_rows, path_cols, angles):
            # accelerate algorithm by not rotating if angle_velocity is 0
            if angle_velocity == 0:
                slice_x = slice(row - hr_size[0] // 2, row + (hr_size[0] + 1) // 2)
                slice_y = slice(col - hr_size[1] // 2, col + (hr_size[1] + 1) // 2)
                hr_frame += scene[slice_x, slice_y]
            else:
                # diameter of circumscribing circle
                circum_diam = int(np.ceil(np.linalg.norm(hr_size)))
                slice_x = slice(row - circum_diam // 2, row + (circum_diam + 1) // 2)
                slice_y = slice(row - circum_diam // 2, row + (circum_diam + 1) // 2)
                unrotated = scene[slice_x, slice_y]
                hr_frame += crop(rotate(unrotated, angle, reshape='largest'), width=hr_size)
        # scale collected energy of subframes
        hr_frame /= frame_rate * len(path_rows)
        frames[frame] = rescale(hr_frame, 1 / resolution_ratio, anti_aliasing=False)

    return frames, mid

@vectorize
def add_noise(signal, dbsnr=None):
    """
    Add noise to the given signal at the specified level.

    Args:
        (ndarray): noise-free input signal
        dbsnr (float): signal to noise ratio in dB: for Gaussian noise model, it is
        defined as the ratio of variance of the input signal to the variance of
        the noise. For Poisson model, it is taken as the average snr where snr
        of a pixel is given by the square root of its value.
        max_count (int): Max number of photon counts in the given signal
        model (string): String that specifies the noise model. The 2 options are
        `Gaussian` and `Poisson`
        no_noise (bool): (default=False) If True, return the clean signal

    Returns:
        ndarray that is the noisy version of the input
    """
    if dbsnr is None:
        return signal
    else:
        var_sig = np.var(signal)
        var_noise = var_sig / 10**(dbsnr / 10)
        return np.random.normal(loc=signal, scale=np.sqrt(var_noise))
