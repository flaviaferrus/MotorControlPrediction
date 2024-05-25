#########################################
##          GENERAL IMPORTS            ##
#########################################

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import scipy.io
import os
import random
from sklearn.cluster import KMeans
from typing import Tuple, List
import warnings

#########################################
## DATA LOADING AND CLEANING FUNCTIONS ##
#########################################

def load_data(path = 'dataTrajectories-25.mat') -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # Navigate one step up (to the parent directory)
    parent_dir = os.path.dirname(current_dir)
    
    # Enter the 'data' folder
    data_folder = os.path.join(parent_dir, 'data')
    
    # Construct the full file path
    full_path = os.path.join(data_folder, path)
    
    # Load the data from the file
    mat = scipy.io.loadmat(full_path)
    dfx = pd.DataFrame(mat["X"].T)
    dfy = pd.DataFrame(mat["Y"].T)
    dfx = dfx.reset_index().drop(columns="index")
    dfy = dfy.reset_index().drop(columns="index")
    dfx1 = dfx.dropna()
    dfy1 = dfy.dropna()
    
    return dfx1, dfy1

def point_to_segment(cluster_points : List, n_clusters = 4) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    
    segments = []
    for cluster in range(n_clusters):
         # Get the corresponding point for the current cluster
        pt = cluster_points[cluster]
        
        if (pt[0] < 0 and pt[1] > 0): #cluster == 3: # Top left pt2
            pt0 = (pt[0] + 1, pt[1] + 1.75)
            pt1 = (pt[0] - 1, pt[1] - 1.75)
            pt2 = (pt0[0] - 1.25, pt0[1] + 1)
            pt3 = (pt1[0] - 1.25, pt1[1] + 1)
            r = (pt1[0], pt1[1])
            s = (pt0[0], pt0[1])
            
        elif (pt[0] < 0 and pt[1] < 0): #cluster == 2: # Bottom right 1
            pt0 = (pt[0] - 1, pt[1] + 1.75)
            pt1 = (pt[0] + 1, pt[1] - 1.75)
            pt2 = (pt0[0] - 1.25, pt0[1] - 1 )
            pt3 = (pt1[0] - 1.25, pt1[1] - 1)
            r = (pt1[0], pt1[1])
            s = (pt0[0], pt0[1])
            
        elif (pt[0] > 0 and pt[1] > 0): #cluster == 0: # Top right 2, 0
            pt0 = (pt[0] - 1, pt[1] + 1.75)
            pt1 = (pt[0] + 1, pt[1] - 1.75)
            pt2 = (pt0[0] + 1.25, pt0[1] + 1 )
            pt3 = (pt1[0] + 1.25, pt1[1] + 1)
            s = (pt1[0], pt1[1])
            r = (pt0[0], pt0[1])
            
    return segments


def on_segment(p, r, s, tol = 1e-3) -> bool:
    x_max = max(r[0], s[0])
    x_min = min(r[0], s[0])
    y_max = max(r[1], s[1])
    y_min = min(r[1], s[1])
    if (p[0] <= x_max and p[0] >= x_min and 
        p[1] <= y_max and p[1] >= y_min and 
        (p[0] - r[0]) / (p[1] - r[1] ) - (s[0] - r[0]) / (s[1] - r[1]) < tol ):
            return True
    return False


def cleaning_multiple_data(dfx : pd.DataFrame, dfy : pd.DataFrame, 
                  segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Tuple[pd.DataFrame, pd.DataFrame, list]: 
    
    print('Dataset shape before cleaning:')
    print(dfx.shape)
        
    idxrule = []
    
    for dx in dfx.index:
        t_intersect = None  # Initialize t_intersect
        for t in range(len(dfx.columns) - 1):
            p = (dfx.loc[dx, dfx.columns[t]], dfy.loc[dx, dfx.columns[t]])
            for r, s in segments:
                if on_segment(p, r, s):
                    # If intersects, keep trajectory up to the intersecting point
                    t_intersect = t + 1
                    idxrule.append(t_intersect)
                    dfx.loc[dx, dfx.columns[t_intersect:]] = p[0]
                    dfy.loc[dx, dfx.columns[t_intersect:]] = p[1]
                    # And we stop looking for the intersection
                    break
            if t_intersect is not None:
                break

        if t_intersect is None:
            # No intersection found, drop the row
            #print('Dropping row: ', dx)
            dfx = dfx.drop(axis=0, index=dx)
            dfy = dfy.drop(axis=0, index=dx)

    index1 = dfy[(dfy < -10).any(axis=1)].index
    index2 = dfx[(dfx < -15).any(axis=1)].index
    index3 = dfx[(dfx > 15).any(axis=1)].index
    index4 = dfx[(dfy > 10).any(axis=1)].index
    index0 = index1.union(index2)
    index01 = index0.union(index3)
    index = index01.union(index4)
    
    dfx = dfx.drop(axis=0, index=index)
    dfy = dfy.drop(axis=0, index=index)
    dfx.reset_index(drop=True, inplace=True)
    dfy.reset_index(drop=True, inplace=True)
    
    print('Dataset shape after cleaning:')
    print(dfx.shape)
    
    return dfx, dfy, idxrule
