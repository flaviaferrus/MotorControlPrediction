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
from typing import Tuple
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

def data_clustering(dfx : pd.DataFrame, dfy : pd.DataFrame) -> Tuple[list, np.ndarray]: 
    #np.random.seed()
    # Combine dfx1 and dfy1 into a single feature matrix
    features = np.hstack((dfx.values, dfy.values))
    # Number of clusters
    n_clusters = 4
    # Set a fixed random seed
    random_seed = 12
    # Initialize KMeans with random seed
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    # Fit KMeans clustering model
    with warnings.catch_warnings():  # Suppress the warning temporarily
        warnings.simplefilter("ignore")
        kmeans.fit(features)
    # Get cluster labels
    cluster_labels = kmeans.labels_
    # Create empty lists to hold trajectories for each cluster
    cluster_dfx = [[] for _ in range(n_clusters)]
    cluster_dfy = [[] for _ in range(n_clusters)]
    # Split trajectories into separate lists for each cluster
    for i, label in enumerate(cluster_labels):
        cluster_dfx[label].append(dfx.iloc[i])
        cluster_dfy[label].append(dfy.iloc[i])
    # Concatenate trajectories within each cluster to create datasets
    cluster_datasets = []
    for i in range(n_clusters):
        cluster_dfx[i] = pd.concat(cluster_dfx[i], axis=1).T.reset_index(drop=True)
        cluster_dfy[i] = pd.concat(cluster_dfy[i], axis=1).T.reset_index(drop=True)
        cluster_datasets.append((cluster_dfx[i], cluster_dfy[i]))
    # Print the number of trajectories in each cluster
    for i in range(n_clusters):
        print(f"Cluster {i + 1} has {len(cluster_datasets[i][0])} trajectories for x and {len(cluster_datasets[i][1])} trajectories for y.")
        
    return cluster_datasets, cluster_labels

def plot_data(dfx : pd.DataFrame, dfy : pd.DataFrame, 
              cluster_labels: list, 
              cluster = 2, 
              n_clusters = 4,
              plotting_target = False, 
              saving_plot = False,
              pt = (0,0),
              pic_name = 'cluster'
              ) -> Tuple[np.ndarray, np.ndarray]: 
    
    plt.figure(figsize=(10, 6))
    
    if len(cluster_labels) == 0:
        for i in range(len(dfx)):
            plt.plot(dfx.iloc[i], dfy.iloc[i])
    else: 
        for i in range(len(dfx)):
            if (cluster_labels[i] == cluster):
                plt.plot(dfx.loc[i], dfy.loc[i], color=plt.cm.jet(cluster_labels[i] / n_clusters), alpha=0.5)
    
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
        plt.scatter(rectx,recty)
        plt.plot([rectx[0], rectx[2]], [recty[0], recty[2]], color = 'red', alpha = 0.5)
    else: 
        rectx,recty=np.array([0,0,0,0]), np.array([0,0,0,0])
     
    plt.title('Trajectories in Cluster {}'.format(cluster))
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    if saving_plot:
        # Check if the 'pics' folder exists, if not, create it
        if not os.path.exists('pics'):
            os.makedirs('pics')
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}{cluster}.png'
        filepath = os.path.join('pics', filename)
        plt.savefig(filepath)
    
    plt.show()
        
    return rectx, recty

def plot_trajectory(x: np.ndarray, y: np.ndarray,
                    showing = True, via = True, plot_title = 'Mean Trajectory'): 
    plt.plot(x,y,color='blue', label=plot_title, alpha = 1)
    if via:
        angle=math.pi*7/24
        T_1=.2
        plt.plot(np.cos(angle*(T_1-1)),np.sin(angle*(T_1-1)),marker='o',markersize=35)
    if showing: 
        plt.show()
    
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

def cleaning_data(dfx : pd.DataFrame, dfy : pd.DataFrame, 
                  rectx : np.ndarray, recty : np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, list]: 
    
    print('Dataset shape before cleaning:')
    print(dfx.shape)
    
    ## Points defining the target segment
    ## We check the trajectories direction:
    if rectx[0] > 0 :
        r = (rectx[0], recty[0])
        s = (rectx[2], recty[2])
    else: 
        s = (rectx[0], recty[0])
        r = (rectx[2], recty[2])
        
    idxrule = []
    
    for dx in dfx.index:
        t_intersect = None  # Initialize t_intersect
        for t in range(len(dfx.columns) - 1):
            p = (dfx.loc[dx, dfx.columns[t]], dfy.loc[dx, dfx.columns[t]])
            if on_segment(p, r, s):
                # If intersects, keep trajectory up to the intersecting point
                t_intersect = t + 1
                idxrule.append(t_intersect)
                dfx.loc[dx, dfx.columns[t_intersect:]] = p[0]
                dfy.loc[dx, dfx.columns[t_intersect:]] = p[1]
                # And we stop looking for the intersection
                break

        if t_intersect is None:
            # No intersection found, drop the row
            #print('Dropping row: ', dx)
            dfx = dfx.drop(axis=0, index=dx)
            dfy = dfy.drop(axis=0, index=dx)

    if (rectx[0] > 0):
        index1 = dfy[(dfy < -2).any(axis=1)].index
        index2 = dfx[(dfx < -5).any(axis=1)].index
        index3 = dfx[(dfx > 15).any(axis=1)].index
        index0 = index1.union(index2)
        index = index0.union(index3)
    else:
        index1 = dfy[(dfy < -8).any(axis=1)].index
        index2 = dfx[(dfx < -13).any(axis=1)].index
        index3 = dfx[(dfy > 3).any(axis=1)].index
        index0 = index1.union(index2)
        index = index0.union(index3)
    dfx = dfx.drop(axis=0, index=index)
    dfy = dfy.drop(axis=0, index=index)
    dfx.reset_index(drop=True, inplace=True)
    dfy.reset_index(drop=True, inplace=True)
    
    print('Dataset shape after cleaning:')
    print(dfx.shape)
    
    return dfx, dfy, idxrule

def linear_transf(dfx : pd.DataFrame, dfy : pd.DataFrame, 
                  rectx : np.ndarray, recty : np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    model_target = np.array((1,0))
    if rectx[0] > 0 and recty[-1] > 0:
        screen_target = np.array(((rectx[3]+rectx[0])/2,(recty[3]+recty[0])/2))
    elif rectx[0] < 0 and recty[-1] > 0: 
        dfx = -1 * dfx
        screen_target = np.array(((-rectx[3]-rectx[0])/2,(recty[3]+recty[0])/2))
    elif rectx[0] > 0 and recty[-1] < 0:
        dfy = -1 * dfy
        screen_target = np.array(((rectx[3]+rectx[0])/2,(-recty[3]-recty[0])/2)) 
    else: 
        dfx = -1 * dfx
        dfy = -1 * dfy
        screen_target = np.array(((-rectx[3]-rectx[0])/2,(-recty[3]-recty[0])/2))
        
    model_origin = np.array((np.cos(-math.pi*7/24),np.sin(-math.pi*7/24))) 
    screen_origin = np.array((0,0))

    v_model=model_target-model_origin
    v_model_ort=np.array((v_model[1],-v_model[0]))

    v_screen=screen_target-screen_origin
    v_screen_ort=np.array((v_screen[1],-v_screen[0]))

    model_M=np.vstack((np.append(model_origin,1),np.append(model_target,1),np.append(v_model_ort,0))).T
    screen_M=np.vstack((np.append(screen_origin,1),np.append(screen_target,1),np.append(v_screen_ort,0))).T

    M=np.dot(model_M,np.linalg.inv(screen_M))
    A=M[:2,:2]
    b=M[:2,-1:].flatten()
    
    dfx_=A[0,0]*dfx+A[0,1]*dfy+b[0]
    dfy_=A[1,0]*dfx+A[1,1]*dfy+b[1]
    
    return dfx_, dfy_

def experimental_velocity(dfx : pd.DataFrame, dfy : pd.DataFrame) -> pd.DataFrame:
    # Computing velocity of each point along the trajectories
    dfx1=dfx.iloc[:,1:]
    dfx1.columns = range(dfx1.shape[1])
    dfvx=dfx1-dfx.iloc[:,:-1]

    dfy1=dfy.iloc[:,1:]
    dfy1.columns = range(dfy1.shape[1])
    dfvy=dfy1-dfy.iloc[:,:-1]

    dfv=np.sqrt(np.square(dfvx)+np.square(dfvy))
    
    return dfv

def plot_velocity(dfv : pd.DataFrame, saving_plot = False, pic_name = 'Velocity') -> None:
     
    plt.figure(figsize=(10, 6))
    for i in range(len(dfv)):
        plt.plot(dfv.iloc[i], color='gray', alpha=0.5)
    plt.plot(dfv.mean(axis = 0), label='Mean Velocity')
    plt.title(pic_name)
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.grid(True)
    if saving_plot:
        # Check if the 'pics' folder exists, if not, create it
        if not os.path.exists('pics'):
            os.makedirs('pics')
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}.png'
        filepath = os.path.join('pics', filename)
        plt.savefig(filepath)
    plt.show()


def saving_processed_data(df : pd.DataFrame, folder_name = 'processed_data', file_name = 'processed_dfx'): 
    # Get the current directory
    current_dir = os.getcwd()
    
    # Navigate one step up (to the parent directory)
    parent_dir = os.path.dirname(current_dir)
    
    # Enter the 'data' folder
    data_folder = os.path.join(parent_dir, folder_name)
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # Save the DataFrame to a file
    filename = f'{file_name}.csv'
    file_path = os.path.join(data_folder, filename)   
    df.to_csv(file_path, index=False) 
    
def load_processed_data(folder_path: str, file_name: str) -> pd.DataFrame:
    # Get the current directory
    current_dir = os.getcwd()
    
    # Navigate one step up (to the parent directory)
    parent_dir = os.path.dirname(current_dir)
    
    # Enter the 'data' folder
    data_folder = os.path.join(parent_dir, folder_path)
    
    # Load the DataFrame from the file
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)  # Change the format as per your requirement
    return df

