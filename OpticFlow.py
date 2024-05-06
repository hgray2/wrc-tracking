"""\
A specialized optic flow module which is used
primarily for determining the direction of motion
of a moving camera.

__author__: Hayden Gray
"""

### Imports ###
import numpy as np
from skimage.filters import gaussian
from scipy.ndimage import convolve
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from skimage.color import rgb2gray
from tqdm import tqdm
################

END_POINT_MULTIPLIER = 100

def df_dt(curr_img, prev_img):
    gaussian(curr_img, sigma=3, output = curr_img)
    gaussian(prev_img, sigma=3, output = prev_img)
    return curr_img - prev_img

def df_dy(img):
    sobel_y = np.array([[-1,-2,-1],
                         [0,0,0],
                         [1,2,1]]) / 8
    return convolve(img, sobel_y, mode='nearest')


def df_dx(img):
    sobel_x = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]]) / 8
    return convolve(img, sobel_x, mode='nearest') 

def _get_neighbors(row, col, radius):
    neighbors = []

    start_col = col - radius
    start_row = row - radius
    for i in range (start_row, start_row + (radius*2)+1):
        for j in range(start_col, start_col + (radius*2)+1):
            neighbors.append((i,j))
    
    return neighbors

def calculate_velocity(row, col, df_dx_img, df_dy_img, df_dt_img, radius) -> np.ndarray:
    neighbors = _get_neighbors(row,col,radius)

    sum_fx_squared = 0
    sum_fy_squared = 0
    sum_fx_fy = 0
    sum_fx_ft = 0
    sum_fy_ft = 0

    ### Calculation of matrix A and vector b ##
    for neighbor in neighbors:
        df_dx_neighbor = df_dx_img[neighbor[0], neighbor[1]]
        df_dy_neighbor = df_dy_img[neighbor[0], neighbor[1]]
        df_dt_neigbor = df_dt_img[neighbor[0], neighbor[1]]

        sum_fx_squared += df_dx_neighbor ** 2
        sum_fy_squared += df_dy_neighbor ** 2
        sum_fx_fy += df_dx_neighbor * df_dy_neighbor
        sum_fx_ft += df_dx_neighbor * df_dt_neigbor
        sum_fy_ft += df_dy_neighbor * df_dt_neigbor

    A = np.ndarray((2,2))
    A[0,0] = sum_fx_squared
    A[0,1] = sum_fx_fy
    A[1,0] = sum_fx_fy
    A[1,1] = sum_fy_squared

    b = np.ndarray((2,1))
    b[0,0] = -1 * sum_fx_ft
    b[1,0] = -1 * sum_fy_ft

    return np.linalg.lstsq(A, b, rcond=None)[0]

def _findCameraDirection(img1, img2):
    vectors = calculateFlow(img1, img2)
    dir = np.mean(vectors, axis=0)
    return dir 

def camDirArrFromImageArr(imageArr, velocity_scale = 1):
    """
    'Camera Direction Array From Image Array'

    Accepts an image array and returns an array with 
    the direction of the cameras velocity for each frame 
    in the image array.

    @param 
        velocity_scale: 
            Scales the resulting velocity vectors.
    """

    dirArr = []

    grayImageArr = rgb2gray(imageArr)

    for i in tqdm(range(1, len(imageArr))):
        camera_dir = _findCameraDirection(grayImageArr[i-1], grayImageArr[i])
        camera_dir = camera_dir * velocity_scale
        dirArr.append(camera_dir)

    dirArr = np.array(dirArr)
    dirArr = gaussian(dirArr, 2) 
    
    return np.array(dirArr)


def calculateFlow(img1, img2):
    df_dx_img = df_dx(img2)
    df_dy_img = df_dy(img2)
    df_dt_img = df_dt(img2,img1)

    # for velocity calculation
    radius = 2

    rows = img1.shape[0]
    cols = img1.shape[1]

    stride = 30

    vectors = []
    for row in range(radius+2, rows-radius-2, stride):
        for col in range(radius+2, cols-radius-2, stride):
            velocity = calculate_velocity(row, col, df_dx_img, df_dy_img, df_dt_img, radius)

            x_comp = velocity[0,0]
            y_comp = velocity[1,0]

            if(x_comp > 100 or y_comp > 100 or x_comp < -100 or y_comp < -100):
                # DO NOTHING
                pass
            else:
                vectors.append((x_comp, y_comp))

    return vectors
