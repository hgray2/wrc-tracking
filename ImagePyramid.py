"""\
Image pyramiding module which is used for downsampling
images for more efficient computation with traditionally
brute force template matching algorithms such as NCC.    

__author__: Hayden Gray
"""


### Imports ###
import numpy as np
from skimage.filters import gaussian
from scipy import ndimage
###############

def calcDimension(img_dimension):
    dimension = 2
    while(dimension * 2 < img_dimension):
        dimension = dimension * 2
    
    return dimension

def blur(img):
    blurred_img = np.ndarray(shape=img.shape)
    for i in range(3):
        blurred_img[:,:,i] = gaussian(img[:,:,i],sigma=1)
    return blurred_img

def downSample(img):
    return img[:img.shape[0]:2,:img.shape[1]:2, :] 

def interp(img):
    interp_rows = (img.shape[0] * 2)

    interp_cols = (img.shape[1] * 2)

    interp_image = np.zeros((interp_rows, interp_cols, 3))

    horizontal_avg_kernel = np.array([[1/2,0,1/2]])
    vert_avg_kernel = np.array([[1/2],[0],[1/2]])
    corner_avg_kernel = np.array([[1/4,0,1/4],
                                    [0,0,0],
                                    [1/4,0,1/4]])
    
    for i in range(3): # RGB
        interp_image[::2, ::2, i] = img[:,:,i]

        horizontal_avg = ndimage.convolve(interp_image[:,:,i], horizontal_avg_kernel, mode='nearest')
        vertical_avg = ndimage.convolve(interp_image[:,:,i], vert_avg_kernel, mode = 'nearest')
        corner_avg = ndimage.convolve(interp_image[:,:,i], corner_avg_kernel, mode = 'nearest')

        average_img = horizontal_avg + vertical_avg + corner_avg

        interp_image[1::2,:, i] = average_img[1::2,:]
        interp_image[:,1::2, i] = average_img[:,1::2]

    return interp_image

def getImagePyramid(img: np.array):
    """
    Returns the image pyramid for the given image.

    Usage:
        imgPyr[0] = (smallest image)
        imgPyr[1] = (middle image)
        imgPyr[2] = (full size image)
    """
    # Number of levels
    N = 3

    if(img.shape[2] == 4):
        img = img[:,:,:3]

    # Crop the image
    height = calcDimension(img.shape[0])
    width = calcDimension(img.shape[1])

    row_diff = img.shape[0] - height
    col_diff = img.shape[1] - width

    left_row_diff = right_row_diff = 0
    lower_col_diff = upper_col_diff = 0

    if(row_diff %2 == 0):
        left_row_diff = right_row_diff = int(row_diff/2)
    else:
        left_row_diff = int(row_diff/2)
        right_row_diff = int(row_diff/2) + 1

    if(col_diff %2 == 0):
        lower_col_diff = upper_col_diff = int(col_diff/2)
    else:
        upper_col_diff = int(col_diff/2)
        lower_col_diff = int(col_diff/2) + 1

    img = img[left_row_diff : img.shape[0] - right_row_diff, upper_col_diff : img.shape[1] - lower_col_diff, :]

    ### GAUSSIAN PYRAMID ###
    level3_img = img / 255.0

    level2_img = downSample(blur(level3_img)) 

    level1_img = downSample(blur(level2_img)) 

    level0_img = downSample(blur(level1_img)) 
    #########################

    ### LAPLACIAN PYRAMID ###
    level1_diff_img = level1_img - interp(level0_img)

    level2_diff_img = level2_img - interp(level1_img)

    level3_diff_img = level3_img - interp(level2_img)
    #########################

    ### RECONSTRUCTION ###
    level1_recon = interp(level0_img) + level1_diff_img

    level2_recon = interp(level1_recon) + level2_diff_img

    level3_recon = interp(level2_recon) + level3_diff_img
    ######################

    return(level1_recon, level2_recon, level3_recon)
