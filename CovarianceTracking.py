"""\
Covariance tracking class which allows the client
to calculate the covariance of an input target and 
then locate that target within a larger image.

__author__: Hayden Gray
"""
### Imports ###
import math

import numpy as np
from scipy.linalg import eig
from tqdm import tqdm
################

NUM_FEATURES = 5

class CovarianceTracker():

    target_cov_matrix = None

    def calculateCovariance(self,region: np.array):
        region_reshaped = region.reshape(region.shape[0]*region.shape[1],5)
        region_reshaped = np.transpose(region_reshaped)
        return np.cov(region_reshaped,bias=True)

    def distance(self,model: np.array, candidate: np.array):
        eig_vals = eig(model, candidate)[0]
        eig_vals_real = np.real(eig_vals)

        sum = 0
        for i in range(len(eig_vals_real)):
            if(eig_vals_real[i] != 0):
                sum += (math.log(eig_vals_real[i]) ** 2)

        return np.sqrt(sum)

    def generateFeatureVector(self,row, col, img):

        x = col
        y = row
        R = img[row,col][0]
        G = img[row,col][1]
        B = img[row,col][2]

        return np.array([x,y,R,G,B])

    def covarianceLocate(self, target: np.array, frame: np.array):
        """
        Returns the location of the target in the given frame.
        
        target: the target to be located.
        frame: The frame in which to search for the target.
        """
        P, Q, _ = target.shape
        M, N, _ = frame.shape


        # We only need to compute this once
        if(self.target_cov_matrix is None):
            ### Create array of feature vectors ###
            featureArray = np.zeros(shape=(P, Q, 5))
            for i in range(P): # Rows
                for j in range(Q): # Cols
                    featureArray[i,j] = self.generateFeatureVector(i,j,target)
            self.target_cov_matrix = self.calculateCovariance(featureArray)


        ### Create array of feature vectors ###
        featureArray = np.zeros(shape=(M, N, 5))
        for i in range(frame.shape[0]): # Rows
            for j in range(frame.shape[1]): # Cols
                featureArray[i,j] = self.generateFeatureVector(i,j,frame)


        best_candidate_dist = float("inf")
        best_candidate_pos = (0,0)

        ### Check candidates at every pixel ###
        for i in tqdm(range(M-P)):
            for j in range(N-Q):
                '''
                Region is of size P x Q
                so we slice the image starting
                from the current row up to P
                and from the current col up to Q
                '''
                candidate = self.calculateCovariance(region=featureArray[i:i+P, j:j+Q])
                candidate_distance = self.distance(self.target_cov_matrix, candidate)
                if(candidate_distance < best_candidate_dist):
                    best_candidate_dist = candidate_distance

                    # Add offset to candidate position (position should be in the center of the detection, not the corner)
                    best_candidate_pos = (j + Q/2,i + P/2)

        
        return best_candidate_pos
                