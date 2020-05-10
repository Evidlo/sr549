from sr549.misc import np_gpu, roll
from skimage.draw import line
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, vstack
from cachalot import Cache
from tqdm import tqdm

# @Cache(size=1)
def forward(
        factor=2, start=(0, 0), motion_blur=True, *,
        drift_angle, drift_velocity, frame_rate, num_frames,
        lr_size, hr_size
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

    exp_time = num_frames / frame_rate

    # ----- motion kernel start points -----

    total_dx = np.cos(drift_angle) * drift_velocity * exp_time
    total_dy = np.sin(drift_angle) * drift_velocity * exp_time
    start = np.array(start)
    # top left point of last frame
    end = start + (total_dx, total_dy)

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

    return vstack(rows)
