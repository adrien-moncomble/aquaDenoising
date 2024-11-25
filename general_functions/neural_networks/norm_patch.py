'''
Functions to normalize images, create patches and recombine the patches.
'''
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

def normalization(img):
    '''
    Normalize the image(s).

    Parameters:
    img: Array containing one or multiple images. If multiple images then the shape should be
    (nb images, size in 1st direction, size in 2nd direction).
    
    Return:
    Array of normalized image(s).
    '''
    img_shape = img.shape

    if len(img_shape)==3:
        min_img = np.tile(np.min(img, axis=(1,2))[:, np.newaxis, np.newaxis],
                          (1,img_shape[1],img_shape[2]))
        max_img = np.tile(np.max(img, axis=(1,2))[:, np.newaxis, np.newaxis],
                          (1,img_shape[1],img_shape[2]))
    else:
        min_img = np.min(img)
        max_img = np.max(img)

    return (img-min_img)/(max_img-min_img+1e-8)



def robustnorm(img, weight):
    '''
    Apply a robust normalization on the image(s).

    Parameters:
    img: Array containing one or multiple images. If multiple images then the shape should be
    (nb images, size in 1st direction, size in 2nd direction).
    weight: Weighting parameters to reduce the impact of outliers.
    
    Return:
    Array of normalized image(s).
    '''
    img_shape = img.shape

    if len(img_shape)==3:
        min_img = np.tile(np.min(img, axis=(1,2))[:, np.newaxis, np.newaxis],
                          (1,img_shape[1],img_shape[2]))
        max_img = np.tile(np.max(img, axis=(1,2))[:, np.newaxis, np.newaxis],
                          (1,img_shape[1],img_shape[2]))
    else:
        min_img = np.min(img)
        max_img = np.max(img)

    prenorm_img = (img-((1-weight)*min_img)-(weight*max_img))/((max_img-min_img+1e-8)*(1-2*weight))
    tail1 = np.where(prenorm_img <= 0, 0, prenorm_img)
    return np.where(tail1 >= 1, 1, tail1)



def evenpatch(full_img, patch_size=128, step=32):
    '''
    Transform an image into an array of smaller one.
    The patches have a size defined by patch_size px and are separated by step px.
    
    Parameters:
    full_img: Image to transform into patches.
    patch_size: Size of the patches.
    step: Distance between each patches.
    
    Return:
    Array of patch of the input image.
    '''

    patches_img = []

    img_height, img_width = full_img.shape[:2]
    j_end_old = -1
    k_end_old = -1

    for j in range(0, img_height, step):
        for k in range(0, img_width, step):
            j_end = min(j + patch_size, img_height)
            k_end = min(k + patch_size, img_width)
            j_start = j_end - patch_size
            k_start = k_end - patch_size

            if j_end!=j_end_old and k_end!=k_end_old:
                patch = full_img[j_start:j_end, k_start:k_end]
                patches_img.append(patch)

            k_end_old = k_end
        j_end_old = j_end

    patches_list = np.array(patches_img, dtype=np.float32)
    return patches_list



def evenreconstruct(original_img, p_list, step=32, border=16):
    '''
    Transform an array of small patches into a larger image with the size of the original image.
    To reposition each patches we follow the same algorithm as the creation of evenly distributed
    patches.
    
    Parameters:
    original_img: Reference image with the same size as the final reconstruction.
    p_list: List of patches containing the intensities in the images.
    step: Distance between each patches.
    border: Number of pixels on the edges of the patches that will not be considered for the 
            reconstruction.
    
    Return:
    Reconstructed image from the patches in patches_list.
    '''
    recon_norm = np.zeros_like(original_img)
    recon_img = np.zeros_like(original_img, dtype="float32")
    img_height, img_width = original_img.shape[:2]
    patch_size = p_list.shape[1]
    index = 0
    j_end_old = -1
    k_end_old = -1

    for j in range(0, img_height, step):
        for k in range(0, img_width, step):
            j_end = min(j + patch_size, img_height)
            k_end = min(k + patch_size, img_width)
            j_start = j_end - patch_size
            k_start = k_end - patch_size

            if j_end!=j_end_old and k_end!=k_end_old:
                recon_norm[j_start+border:j_end-border,
                           k_start+border:k_end-border] += 1
                recon_img[j_start+border:j_end-border,
                          k_start+border:k_end-border] += p_list[index,border:patch_size-border,
                                                                 border:patch_size-border,0]
                index += 1

            k_end_old = k_end
        j_end_old = j_end

    recon_norm = np.where(recon_norm==0, 1, recon_norm)
    img = recon_img/recon_norm

    return img



def randompatch(full_img, nb_patch, patch_size=128, seed=0, write_xy=True):
    '''
    Transform an image into an array of smaller one.
    The patches have a size defined by patch_size px and there will nb_patch different patches
    generated.
    
    Parameters:
    full_img: Image to transform into patches.
    nb_patch: Number of different patches generated.
    patch_size: Size of the patches.
    seed: Random seed to reproduce results.
    
    Return:
    Array of patch of the input image and two arrays to reconstruct the full images.
    '''
    patches_img = extract_patches_2d(full_img, (patch_size,patch_size), max_patches=nb_patch,
                                     random_state=seed)
    patches_list = np.array(patches_img, dtype=np.float32)

    if write_xy:
        x = np.arange(full_img.shape[1])
        y = np.arange(full_img.shape[0])
        y_grid, x_grid = np.meshgrid(x,y)
        patches_x = extract_patches_2d(x_grid, (patch_size,patch_size), max_patches=nb_patch,
                                    random_state=seed)[:,0,0]
        patches_y = extract_patches_2d(y_grid, (patch_size,patch_size), max_patches=nb_patch,
                                    random_state=seed)[:,0,0]

        return patches_list, patches_x, patches_y
    return patches_list



def randomreconstruct(original_img, p_list, p_x, p_y, border=16):
    '''
    Transform an array of small patches into a larger image with the size of the original image.
    To reposition each patches at the right position, it is necessary to give the patches
    coordinates in x and y.
    
    Parameters:
    original_img: Reference image with the same size as the final reconstruction.
    p_list: List of patches containing the intensities in the images.
    p_x: List of position patches along x direction, the order must be the same as p_list.
    p_y: List of position patches along y direction, the order must be the same as p_list.
    border: Number of pixels on the edges of the patches that will not be considered for the 
            reconstruction.
    
    Return:
    Reconstructed image from the patches in p_list.
    '''
    recon_norm = np.zeros_like(original_img)
    recon_img = np.zeros_like(original_img, dtype="float32")
    p_size = p_list.shape[1]

    for i in range(len(p_list)):
        recon_norm[p_x[i]+border:p_x[i]+p_size-border,
                   p_y[i]+border:p_y[i]+p_size-border] += 1
        recon_img[p_x[i]+border:p_x[i]+p_size-border,
                  p_y[i]+border:p_y[i]+p_size-border] += p_list[i,border:p_size-border,
                                                                border:p_size-border,0]

    recon_norm = np.where(recon_norm==0, 1, recon_norm)
    img = recon_img/recon_norm

    return img
