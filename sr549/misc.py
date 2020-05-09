import numpy as np

def crop(im, center=None, *, width):
    """
    Return a cropped rectangle from an input image

    Args:
        im (ndarray): input image
        center (tuple): coordinate pair of center of cropped rectangle.
            defaults to image center
        width (int, tuple): length of each axis of cropped rectangle.
            returns a square region if integer

    Returns:
        cropped rectangle of input image
    """

    if type(width) is int:
        width = (width, width)

    if center is None:
        center = (im.shape[0] // 2, im.shape[1] // 2)

    assert (
        (0 <= center[0] - width[0]) and
        (0 <= center[1] - width[1]) and
        (im.shape[0] >= center[0] + width[0]) and
        (im.shape[1] >= center[1] + width[1])
    ), "Cropped region falls outside image bounds"

    crop_left = (im.shape[0] - width[0] + 1) // 2
    crop_right = crop_left + width[0]
    crop_top = (im.shape[1] - width[1] + 1) // 2
    crop_bottom = crop_top + width[1]

    return im[crop_left:crop_right, crop_top:crop_bottom]


def _vectorize(signature='(m,n)->(i,j)', included=[0]):
    """Decorator to make a 2D functions work with higher dimensional arrays
    Last 2 dimensions are taken to be images
    Iterate over first position argument.

    Args:
        signature (str): override mapping behavior
        included (list): list of ints and strs of position/keyword arguments to iterate over

    Returns:
        function: decorator which can be applied to nonvectorized functions

    Signature examples:

        signature='(),()->()', included=[0, 1]
            first two arguments to function are vectors. Loop through each
            element pair in these two vectors and put the result in a vector of
            the same size. e.g. if args 0 and 1 are of size (5, 5), the output
            from the decorated function will be a vector of size (5, 5)

        signature='(m,n)->(i,j)', included=[0]
            input argument is a vector with at least two dimensions. Loop
            through all 2d vectors (using last 2 dimensions) in this input
            vector, process each, and return a 2d vector for each. e.g. if arg
            0 is a vector of size (10, 5, 5), loop through each (5, 5) vector
            and return a (10, 5, 5) vector of all the results

        signature='(m,n)->()', included=[0]
            input argument is a vector with at least two dimensions. Loop
            through each 2d image and return a 1d vector. e.g. if arg 0 is a
            vector of size (10, 5, 5), return a vector of size (10)
    """
    def decorator(func):

        def new_func(*args, **kwargs):
            nonlocal signature

            # exclude everything except included
            excluded = set(range(len(args))).union(set(kwargs.keys()))
            excluded -= set(included)

            # allow signature override
            if 'signature' in kwargs.keys():
                signature = kwargs['signature']
                kwargs.pop('signature')

            return np.vectorize(func, excluded=excluded, signature=signature)(*args, **kwargs)

        return new_func

    return decorator

vectorize = _vectorize()



def downsample(x, factor=2):
    """
    Downsample an image by averaging factor*factor sized patches.  Discards remaining pixels
    on bottom and right edges

    Args:
        x (ndarray): input image to downsample
        factor (int): factor to downsample image by

    Returns:
        ndarray containing downsampled image
    """

    return fftconvolve(
        x,
        np.ones((factor, factor)) / factor**2,
        mode='valid'
    )[::factor, ::factor]

def downsample2(x, factor=2):
    """
    Downsample an image by average factor*factor sized patches.  

    Args:
        x (ndarray): input image to downsample
        factor (int): factor to downsample image by

    Returns:
        ndarray containing downsampled image
    """

    X = np.fft.fftn(x)
    kern = size_equalizer(np.ones((factor, factor)) / factor**2, x.shape)
    kern = np.fft.fftshift(kern)
    kern = np.fft.fftn(kern)
    # -factor to prevent overlap from first kernel pos to last kernel pos
    # return np.fft.ifftn(kern * X)[:-factor:factor, :-factor:factor]
    return np.fft.ifftn(kern * X)[::factor, ::factor]

@vectorize
def size_equalizer(x, ref_size, mode='center'):
    """
    Crop or zero-pad a 2D array so that it has the size `ref_size`.
    Both cropping and zero-padding are done such that the symmetry of the
    input signal is preserved.
    Args:
        x (ndarray): array which will be cropped/zero-padded
        ref_size (list): list containing the desired size of the array [r1,r2]
        mode (str): ('center', 'topleft') where x should be placed when zero padding
    Returns:
        ndarray that is the cropper/zero-padded version of the input
    """
    assert len(x.shape) == 2, "invalid shape for x"

    if x.shape[0] > ref_size[0]:
        pad_left, pad_right = 0, 0
        crop_left = 0 if mode == 'topleft' else (x.shape[0] - ref_size[0] + 1) // 2
        crop_right = crop_left + ref_size[0]
    else:
        crop_left, crop_right = 0, x.shape[0]
        pad_right = ref_size[0] - x.shape[0] if mode == 'topleft' else (ref_size[0] - x.shape[0]) // 2
        pad_left = ref_size[0] - pad_right - x.shape[0]
    if x.shape[1] > ref_size[1]:
        pad_top, pad_bottom = 0, 0
        crop_top = 0 if mode == 'topleft' else (x.shape[1] - ref_size[1] + 1) // 2
        crop_bottom = crop_top + ref_size[1]
    else:
        crop_top, crop_bottom = 0, x.shape[1]
        pad_bottom = ref_size[1] - x.shape[1] if mode == 'topleft' else (ref_size[1] - x.shape[1]) // 2
        pad_top = ref_size[1] - pad_bottom - x.shape[1]

    # crop x
    cropped = x[crop_left:crop_right, crop_top:crop_bottom]
    # pad x
    padded = np.pad(
        cropped,
        ((pad_left, pad_right), (pad_top, pad_bottom)),
        mode='constant'
    )

    return padded
