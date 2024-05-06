"""\
Mean shift tracking class which allows the client
track an object given its pixel coordinates within
a larger image.

__author__: Hayden Gray
"""

### Imports ###
import math

import numpy as np
from skimage import io

from CovarianceTracking import CovarianceTracker
################

class MeanShiftTracker():

    CV = None

    BINS = 20
    ITERATIONS = 25

    def __init__(self):
        self.CV = CovarianceTracker()

    def _createFeatureVector(self,img,col,row):
        x_val = col
        y_val = row
        R = img[row,col,0]
        G = img[row,col,1]
        B = img[row,col,2]
        return [x_val,y_val,R,G,B]

    def _eKernel(self,r):

        if(r < 1):
            return 1 - r
        else:
            return 0

    def circularNeighbors(self,img, x, y, radius):
        neighbors = []

        rows, cols, _ = img.shape

        for col in range((int(x) - radius), (int(x) + radius)):

            col_centered = col - x
                
            for row in range((int(y) - radius), (int(y) + radius)):
                row_centered = row - y

                if(np.sqrt(col_centered**2 + row_centered**2) <= radius):
                    if((col < cols and col > 0) and (row < rows and row > 0)):
                        feature_vect = self._createFeatureVector(img, col, row)
                        neighbors.append(feature_vect)

        return np.array(neighbors)

    def colorHistogram(self, X, bins, x, y, h):
        hist = np.zeros(shape=(bins,bins,bins)) # R, G, B

        interval = 256/bins

        # Build the histogram

        for vect in X:
            # Figure out which bin this one goes in 
            R = math.floor((vect[2] / interval))
            G = math.floor((vect[3] / interval))
            B = math.floor((vect[4] / interval))
            vect_x = vect[0]
            vect_y = vect[1]

            dist = np.sqrt((x - vect_x) ** 2 + (y - vect_y) ** 2)

            # Before we add it to the bin, we have to weight it using the kernel

            r = (dist / h) ** 2
            val = self._eKernel(r)

            hist[R,G,B] += val

        # Normalize

        hist = hist / np.sum(hist)
        return hist

    def meanshiftWeights(self, X, q_model, p_test, bins):

        interval = 256/bins
        weights = np.zeros(shape=(X.shape[0]))
        
        for i, neighbor in enumerate(X): # every w_i
            # Figure out which bin this belongs to
            R = math.floor((neighbor[2] / interval))
            G = math.floor((neighbor[3] / interval))
            B = math.floor((neighbor[4] / interval))

            if(p_test[R,G,B] != 0):
                weights[i] = np.sqrt(q_model[R,G,B]/p_test[R,G,B])

        return weights

    def bestLocation(self, X, weights):

        numerator = np.zeros(shape=(2))
        denom = 0
        for i, x in enumerate(X):
            pixel = np.array([x[0], x[1]]) # x and y position of this x_i
            numerator += pixel * weights[i]

            denom += weights[i]
        
        return numerator/denom

    def meanShiftLocate(self, prev_frame: np.array, curr_frame: np.array, prev_loc: (int,int), radius) -> (int,int):
        """
        Track the target centered at prev_loc into the current frame

        prev_frame: The previous frame in the image sequence
        curr_frame: The current frame in the image sequence
        prev_loc: Where we detected the target last

        Returns: The location of the target in the current frame.
        """
        center_x, center_y = prev_loc

        h = radius * .75 
        
        q_neighbors = self.circularNeighbors(prev_frame, center_x, center_y, radius)
        q_model = self.colorHistogram(q_neighbors, self.BINS, center_x, center_y, h)

        # q_model (target)
        for i in range(self.ITERATIONS):
            
            # p_test (at 'current' frame)
            p_neighbors = self.circularNeighbors(curr_frame, center_x, center_y, radius)
            p_test = self.colorHistogram(p_neighbors, self.BINS, center_x, center_y, h)

            weights = self.meanshiftWeights(p_neighbors, q_model, p_test, self.BINS)

            y_1 = self.bestLocation(p_neighbors, weights)
            candidate_x = y_1[0]
            candidate_y = y_1[1]

            center_x = candidate_x
            center_y = candidate_y
        
        return (math.ceil(center_x), math.ceil(center_y))
        