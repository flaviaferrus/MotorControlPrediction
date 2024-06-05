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

def saving_processed_mult_data(cleaned_data_dict : list, 
                               folder_name = 'cleaned_multiple_data') -> None: 
    
    print('Saving the cleaned data...')
    # Get the current directory
    current_dir = os.getcwd()
    
    # Navigate one step up (to the parent directory)
    parent_dir = os.path.dirname(current_dir)
    
    # Enter the 'data' folder
    data_folder = os.path.join(parent_dir, folder_name)
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        
    for key, df in cleaned_data_dict.items():
        # Define the file path
        file_path = os.path.join(data_folder, f"{key}.csv")
        
        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        
    print("CSV files have been saved successfully.")
       
       
def load_processed_mult_data(folder_path: str) -> list: 
    
    print('Reading the cleaned files...')
    
    # Initialize an empty dictionary to store the DataFrames
    loaded_data_dict = {}
    
    # Get the current directory
    current_dir = os.getcwd()
    # Navigate one step up (to the parent directory)
    parent_dir = os.path.dirname(current_dir)
    # Enter the 'data' folder
    data_folder = os.path.join(parent_dir, folder_path)
    
    # Iterate over each file in the directory
    for filename in os.listdir(file_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(data_folder, filename)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            
            # Remove the '.csv' extension from the filename to use as the dictionary key
            key = filename[:-4]
            
            # Store the DataFrame in the dictionary
            loaded_data_dict[key] = df

    print("CSV files have been loaded successfully.")
    
    return loaded_data_dict

#########################################
##          PLOTTING DATA              ##
#########################################

def plot_data(dfx : pd.DataFrame, dfy : pd.DataFrame, 
              cluster_labels: list, 
              cluster = 2, 
              n_clusters = 4,
              plotting_target = False, 
              saving_plot = False,
              pt = (0,0),
              pic_name = 'cluster', 
              ax = None
              ) -> Tuple[np.ndarray, np.ndarray]: 
    
    if ax is None:
        ax = plt.gca()
       
    if len(cluster_labels) == 0:
        for i in range(len(dfx)):
            ax.plot(dfx.iloc[i], dfy.iloc[i])
    else: 
        for i in range(len(dfx)):
            if (cluster_labels[i] == cluster):
                ax.plot(dfx.loc[i], dfy.loc[i], color=plt.cm.jet(cluster_labels[i] / n_clusters), alpha=0.5)
    
    if plotting_target:
        if (pt[0] < 0 and pt[1] > 0): #cluster == 3: # Top left pt2
            pt0 = (pt[0] + 1, pt[1] + 1.75)
            pt1 = (pt[0] - 1, pt[1] - 1.75)
            pt2 = (pt0[0] - 1.25, pt0[1] + 1)
            pt3 = (pt1[0] - 1.25, pt1[1] + 1) 
        elif (pt[0] < 0 and pt[1] < 0): #cluster == 2: # Bottom right 1
            pt0 = (pt[0] - 1, pt[1] + 1.75)
            pt1 = (pt[0] + 1, pt[1] - 1.75)
            pt2 = (pt0[0] - 1.25, pt0[1] - 1 )
            pt3 = (pt1[0] - 1.25, pt1[1] - 1)
        elif (pt[0] > 0 and pt[1] > 0): #cluster == 0: # Top right 2, 0
            pt0 = (pt[0] - 1, pt[1] + 1.75)
            pt1 = (pt[0] + 1, pt[1] - 1.75)
            pt2 = (pt0[0] + 1.25, pt0[1] + 1 )
            pt3 = (pt1[0] + 1.25, pt1[1] + 1)
        else: # bottom left 1 (notebook 3)
            pt0 = (pt[0] + 1, pt[1] + 1.75)
            pt1 = (pt[0] - 1, pt[1] - 1.75)
            pt2 = (pt0[0] + 1.25, pt0[1] - 1 )
            pt3 = (pt1[0] + 1.25, pt1[1] - 1)
            
                    
        rectx,recty=np.array([pt0[0], pt2[0], pt1[0], pt3[0]]), np.array([pt0[1], pt2[1], pt1[1], pt3[1]])
        ax.scatter(rectx,recty)
        ax.plot([rectx[0], rectx[2]], [recty[0], recty[2]], color = 'red', alpha = 0.5)
    else: 
        rectx,recty=np.array([0,0,0,0]), np.array([0,0,0,0])
     
    ax.set_title('Trajectories in Cluster {}'.format(cluster))
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if saving_plot:
        # Check if the 'pics' folder exists, if not, create it
        if not os.path.exists('pics'):
            os.makedirs('pics')
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}{cluster}.png'
        filepath = os.path.join('pics', filename)
        plt.savefig(filepath)
    
    if ax is None:
        plt.show()
        
    return rectx, recty