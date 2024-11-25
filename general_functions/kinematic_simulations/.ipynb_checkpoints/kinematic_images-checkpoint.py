'''
Script to simulate STEM images of big objects using the kinematic approximation
'''
import numpy as np
from scipy.signal import fftconvolve
from .volumes import Point, Volume

def kinematic_haadf(volumes, z_element, z_factor, img_size, resolution, center=None):
    '''
    Function that convert the list of objects into a kinematic STEM HAADF image
    '''
    if center is None:
        center = [0,0]

    if not isinstance(volumes, (Volume, tuple, list)):
        raise TypeError("The volumes defined doesn't have the right type. It must be of type "
                        "volumes.Volume, list or tuple")
    if isinstance(volumes, Volume):
        volumes = [volumes,]

    try:
        if len(img_size) != 2:
            raise ValueError("Img_size must contain the height and width of the image")
    except TypeError:
        raise TypeError("Img_size must be an array-like object with 2 parameters") from None

    try:
        if len(center) != 2:
            raise ValueError("Center must contain the x and y coordinate")
    except TypeError:
        raise TypeError("Center must be an array-like object with 2 coordinates") from None

    simu_img = np.zeros(img_size)
    x_img = (np.arange(img_size[1]) - img_size[1]//2)*resolution + center[1]
    y_img = (np.arange(img_size[0]) - img_size[0]//2)*resolution + center[0]

    count = 1
    total = len(volumes)

    for volume in volumes:
        print(f"KinSimul : {count}/{total}")
        vol_min = np.min(volume.positions, axis=0)
        vol_max = np.max(volume.positions, axis=0)
        for y_id in np.nonzero((y_img >= vol_min[1])*(y_img <= vol_max[1]))[0]:
            for x_id in np.nonzero((x_img >= vol_min[0])*(x_img <= vol_max[0]))[0]:
                simu_img[y_id,x_id] += volume.thickness(Point((x_img[x_id],y_img[y_id])))
        count += 1

    simu_img *= z_element**z_factor

    return simu_img

def gaussian_probe(gauss_width, px_dim):
    '''
    Create a gaussian probe
    '''
    x_arr = np.arange(0, px_dim) - (px_dim//2)
    y_arr = np.arange(0, px_dim) - (px_dim//2)
    x_mat, y_mat = np.meshgrid(x_arr, y_arr)

    gauss_probe = ((1/(2*np.pi*gauss_width*gauss_width)) *
                  (np.exp(- ((x_mat**2)/(2*(gauss_width**2))) - ((y_mat**2)/(2*(gauss_width**2))))))

    return gauss_probe

def probe_convolution(img, gauss_width, resolution, img_size=None):
    '''
    Apply a convolution of the probe to the given img
    '''
    if img_size is None:
        probe_img = gaussian_probe(gauss_width/resolution, img.shape[0])
        return fftconvolve(img, probe_img, mode="same")
    probe_img = gaussian_probe(gauss_width/resolution, img_size)
    return fftconvolve(img, probe_img, mode="same")
