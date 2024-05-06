"""\
This file contains helper methods and properties used throughout the project.

__author__: Hayden Gray
__author__: Suyash Talekar
"""

### Imports ###
import av
import cv2
import numpy as np
from matplotlib import pyplot as plt
###############


def videoToArray(fileName: str) -> np.ndarray:
    """
    Takes a video file path and converts it into an image sequence 
    in the form of a numpy array.

    Keyword arguments:
    fileName: file name of the video to be converted.

    __author__: Hayden Gray
    """
    video = av.open(fileName)

    frames = []

    for packet in video.demux(video=0): # video = 0 -> only want video, discard audio.
        for frame in packet.decode():
            img = frame.to_image()  # PIL/Pillow image
            frames.append(np.asarray(img))

    return np.array(frames)


def arrayToVideo(outArr: np.array, fileName: str) -> None:
    """Takes an np.array and renders it to a video.

    Keyword arguments:
    array: the array to be rendered
    fileName: the name of the video file to be generated

    __author__: Sayush Talekar
    """
    # Set the video codec

    # plt.imshow(outArr[1])
    # plt.show()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the height and width from the shape of the first frame
    if(len(outArr[0].shape) == 3):

        height, width, _ = outArr[0].shape
    else:
        height, width = outArr[0].shape

    # Create a VideoWriter object
    out = cv2.VideoWriter(fileName, fourcc, 20.0, (width, height))

    # Write each frame to the video file
    for frame in outArr:
        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object
    out.release()
