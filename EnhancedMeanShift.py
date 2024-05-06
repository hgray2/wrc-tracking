#!/usr/bin/env python3
"""\
This script performs enhanced mean shift tracking on an input
video and outputs an annotated video of the tracked object.
    
Usage: EnhancedMeanShift.py <number>

__author__: Hayden Gray
"""

### Imports ###
import sys

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from tqdm import tqdm
import cv2

from Util.TrackerUtils import videoToArray, arrayToVideo
from OpticFlow import camDirArrFromImageArr 
from CovarianceTracking import CovarianceTracker as CT
from MeanShiftTracking import MeanShiftTracker as MST
from ImagePyramid import getImagePyramid
###############

ms_weight = 0.7
of_weight = 0.3

# Used just for displaying the vectors to the screen (most results are very small)
vector_multiplier = 25

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("ERROR: Please provide a number corresponding to the video data to be processed. Example: python3 EnhancedMeanShift 3")
        exit(1)

    video_number = sys.argv[1]

    in_video_file = "safari_{}.mp4".format(video_number)
    out_video_file = "Out/tracked_" + in_video_file
    imageArr = videoToArray("Data/" + in_video_file)

    # Initialize tracker objects
    covTracker = CT()
    meanShiftTracker = MST()

    # Load template and generate image pyramid
    templateImg = io.imread("Data/safari_{}_target.png".format(video_number))
    pyr0 = getImagePyramid(imageArr[0])
    templatePyr = getImagePyramid(templateImg)

    # Calculate search radius from template size
    search_radius = int((max(templatePyr[1].shape)/2) * .4) # radius is 75 percent of half of the largest dimension of the template.
    draw_radius = int((max(templatePyr[1].shape)/2) * .4)

    # Use covariance tracking to find the location of the vehicle in the first frame
    print("Finding initial location of vehicle...")
    x, y = covTracker.covarianceLocate(templatePyr[1], pyr0[1])
    
    print("Finding camera velocity vectors...")
    camera_velocities = camDirArrFromImageArr(imageArr)

    outArr = []

    print("Running Mean-Shift...")
    for i in tqdm(range(1, len(imageArr))):
        # Generate pyramids for the previous and current frame
        prev_frame_pyr = getImagePyramid(imageArr[i-1])
        curr_frame_pyr = getImagePyramid(imageArr[i])
        prev_frame = (prev_frame_pyr[1] * 255).astype(np.uint8)
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2BGR)
        curr_frame = (curr_frame_pyr[1] * 255).astype(np.uint8)
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR)

        # New position from mean shift
        ms_x, ms_y = meanShiftTracker.meanShiftLocate(prev_frame, curr_frame, (x,y), search_radius)
        ms_mag = np.linalg.norm((ms_x - x,ms_y - y))

        # Make sure that the camera velocity and the meanshift vector share the same magnitude (normalize using ms_mag)
        camera_velocity = camera_velocities[i-1]
        camera_velocity = camera_velocity / np.linalg.norm(camera_velocity)
        camera_velocity = camera_velocity * ms_mag 

        # New location is a linear combintation of the mean shift vector and the camera velocity vector
        new_x = ms_weight * ms_x + of_weight * (x + camera_velocity[0])
        new_y = ms_weight * ms_y + of_weight * (y + camera_velocity[1])

        ms_x_diff = (ms_x - x)
        ms_y_diff = (ms_y - y)

        # Mean shift vector
        ms_vector = np.array([int(ms_x_diff), int(ms_y_diff)])

        # Avoid division by 0 for very small mean shift results
        if (ms_mag != 0):
            ms_vector = (ms_vector / ms_mag) * vector_multiplier
            camera_velocity = (camera_velocity / ms_mag) * vector_multiplier

        ### Draw detection and vectors to the current frame ###
        new_vector = (ms_weight * ms_vector) + (of_weight * camera_velocity)

        # Detection boundary
        frame = cv2.circle(curr_frame, (int(x),int(y)), draw_radius, color=(0,255,0), thickness=3)

        # Mean shift vector (blue)
        frame = cv2.arrowedLine(curr_frame, (int(x),int(y)), (int(x) + int(ms_vector[0]), int(ms_y) + int(ms_vector[1])), color=(0,0,255), thickness = 4)

        # Camera velocity vector (red)
        frame = cv2.arrowedLine(curr_frame, (int(x),int(y)), (int(x) + int(camera_velocity[0]) , int(y) + int(camera_velocity[1])), color=(255,0,0), thickness = 4)
        
        # Result vector (magenta)
        frame = cv2.arrowedLine(curr_frame, (int(x),int(y)), (int(x) + int(new_vector[0]), int(y) + int(new_vector[1])) , color=(255,0,255), thickness = 4)

        outArr.append(frame)

        x = new_x
        y = new_y

    # Export out array to a video file
    arrayToVideo(outArr, out_video_file)

