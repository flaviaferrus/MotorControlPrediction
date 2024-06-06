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
from typing import Tuple, List, Dict
import warnings



#########################################
## DATA LOADING AND SAVING FUNCTIONS   ##
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

def load_multiple_data(first_subj = 25, last_subj = 37, 
                       file_names = 'dataTrajectories') -> list: 
    # Initialize an empty dictionary to store the datasets
    data_dict = {}

    # Loop over subjects
    for subject in range(first_subj, last_subj): 
        # Loop over motivations
        for motivation in range(1, 4):  # Motivation 1 to 3 inclusive
            # Loop over playing modes
            for mode in range(1, 3):  # Playing mode 1 to 2 inclusive
                # Construct the file name
                file_name = f'{file_names}-{subject}-M{motivation}-C{mode}.mat'
                
                # Load the data using the load_data function
                dfx, dfy = load_data(path=file_name)
                
                # Construct a key for the dictionary
                key_x = f'dfx_{subject}_{motivation}{mode}'
                key_y = f'dfy_{subject}_{motivation}{mode}'
                
                # Store the datasets in the dictionary
                data_dict[key_x] = dfx
                data_dict[key_y] = dfy
    return data_dict

def saving_processed_mult_data(cleaned_data_dict: dict, 
                               folder_name: str = 'cleaned_multiple_data') -> None:
    
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
        
    for subject, subject_data in cleaned_data_dict.items():
        subject_folder = os.path.join(data_folder, f'subject_{subject}')
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
        
        for key, df in subject_data.items():
            # Define the file path
            file_path = os.path.join(subject_folder, f"{key}.csv")
            
            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
        
    print("CSV files have been saved successfully.")
    
    def load_processed_mult_data(folder_name: str = 'cleaned_multiple_data') -> dict:
        print('Loading the cleaned data...')
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # Navigate one step up (to the parent directory)
    parent_dir = os.path.dirname(current_dir)
    
    # Enter the 'data' folder
    data_folder = os.path.join(parent_dir, folder_name)
    
    # Check if the folder exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"The folder {data_folder} does not exist.")
        
    cleaned_data_dict = {}
    
    # Iterate through each subject's folder
    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if os.path.isdir(subject_path):
            subject_key = int(subject_folder.split('_')[1])
            cleaned_data_dict[subject_key] = {}
            
            # Iterate through each CSV file in the subject's folder
            for file_name in os.listdir(subject_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(subject_path, file_name)
                    df = pd.read_csv(file_path)
                    key = file_name.split('.csv')[0]
                    cleaned_data_dict[subject_key][key] = df
    
    print("CSV files have been loaded successfully.")
    return cleaned_data_dict

def get_cluster_data(cleaned_data_dict, subject, motivation, mode, cluster):
    cluster_key_x = f'dfx_{subject}_{motivation}{mode}_cluster_{cluster}'
    cluster_key_y = f'dfy_{subject}_{motivation}{mode}_cluster_{cluster}'
    
    if subject in cleaned_data_dict:
        subject_data = cleaned_data_dict[subject]
        if cluster_key_x in subject_data and cluster_key_y in subject_data:
            dfx = subject_data[cluster_key_x]
            dfy = subject_data[cluster_key_y]
            return dfx, dfy
        else:
            raise ValueError(f"Cluster {cluster} not found for subject {subject} with motivation {motivation} and mode {mode}.")
    else:
        raise ValueError(f"Subject {subject} not found in the dataset.")



############################################
## DATA CLEANING AND CLUSTERING FUNCTIONS ##
############################################

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
            
        else: # bottom left 1 (notebook 3)
            pt0 = (pt[0] + 1, pt[1] + 1.75)
            pt1 = (pt[0] - 1, pt[1] - 1.75)
            pt2 = (pt0[0] + 1.25, pt0[1] - 1 )
            pt3 = (pt1[0] + 1.25, pt1[1] - 1)
            s = (pt1[0], pt1[1])
            r = (pt0[0], pt0[1])
        
        segments.append((r, s))  
                
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

def cleaning_clustering_data(dfx : pd.DataFrame, dfy : pd.DataFrame, 
                  segments: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                  printing = False) -> Tuple[pd.DataFrame, pd.DataFrame, list]: 
    
    print('Dataset shape before cleaning:')
    print(dfx.shape)
    
    # Dictionary to hold dataframes for each segment
    segment_data = {i: (pd.DataFrame(columns=dfx.columns), pd.DataFrame(columns=dfy.columns)) for i in range(len(segments))}
    idxrules = {i: [] for i in range(len(segments))}
    
    for dx in dfx.index:
        segment_index = None  # Initialize segment index
        t_intersect = None  # Initialize t_intersect
        for t in range(len(dfx.columns) - 1):
            p = (dfx.loc[dx, dfx.columns[t]], dfy.loc[dx, dfx.columns[t]])
            for i, (r, s) in enumerate(segments):
                if on_segment(p, r, s):
                    # If intersects, keep trajectory up to the intersecting point
                    t_intersect = t + 1
                    segment_index = i
                    dfx.loc[dx, dfx.columns[t_intersect:]] = p[0]
                    dfy.loc[dx, dfy.columns[t_intersect:]] = p[1]
                    # And we stop looking for the intersection
                    break
            if t_intersect is not None:
                break
        
        if t_intersect is not None:
            # Add the trajectory to the corresponding segment's dataframe
            segment_data[segment_index][0].loc[dx] = dfx.loc[dx]
            segment_data[segment_index][1].loc[dx] = dfy.loc[dx]
            idxrules[segment_index].append(t_intersect)
        else:
            # No intersection found, drop the row
            dfx = dfx.drop(axis=0, index=dx)
            dfy = dfy.drop(axis=0, index=dx)
    
    # Drop trajectories that exceed the specified bounds
    index1 = dfy[(dfy < -10).any(axis=1)].index
    index2 = dfx[(dfx < -15).any(axis=1)].index
    index3 = dfx[(dfx > 15).any(axis=1)].index
    index4 = dfx[(dfy > 10).any(axis=1)].index
    index0 = index1.union(index2)
    index01 = index0.union(index3)
    index = index01.union(index4)
    
    for key in segment_data.keys():
        segment_dfx, segment_dfy = segment_data[key]
        segment_dfx = segment_dfx.drop(axis=0, index=index, errors='ignore')
        segment_dfy = segment_dfy.drop(axis=0, index=index, errors='ignore')
        segment_dfx.reset_index(drop=True, inplace=True)
        segment_dfy.reset_index(drop=True, inplace=True)
        segment_data[key] = (segment_dfx, segment_dfy)
    
    if printing:
        print('Dataset shape after cleaning:')
        for key in segment_data.keys():
            print(f'{key} shape: {segment_data[key][0].shape}')
        
    return segment_data, idxrules


def cleaning_clustering_multiple_data(data_dict: dict, 
                                      segments: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                                      first_subj: int = 25, last_subj: int = 37,
                                      save_dir: str = 'subject_plots') -> Tuple[Dict[int, Dict[str, pd.DataFrame]], Dict[int, Dict[str, List[int]]]]:
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cleaned_data_dict = {}
    idxrule_dict = {}

    # Loop over subjects
    for subject in range(first_subj, last_subj):
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows, 2 columns of subplots
        fig.suptitle(f'Subject {subject} Clustered Trajectories', fontsize=16)
        subplot_index = 0
        
        colors = ['r', 'g', 'b', 'c']  # Add more colors if needed
        #markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']  # Add more markers if needed
        
        for motivation in range(1, 4):
            for mode in range(1, 3):
                
                ax = axes[subplot_index // 2, subplot_index % 2]
                key_x = f'dfx_{subject}_{motivation}{mode}'
                key_y = f'dfy_{subject}_{motivation}{mode}'
                pic_name = f'Cleaned Trajectories-{subject}-M{motivation}-C{mode}'
                
                print('Cleaning and clustering subject data...', pic_name)
                df, idxrule = cleaning_clustering_data(data_dict[key_x], data_dict[key_y], segments)
                
                # Initialize the dictionaries if not already
                if subject not in cleaned_data_dict:
                    cleaned_data_dict[subject] = {}
                if subject not in idxrule_dict:
                    idxrule_dict[subject] = {}
                
                # Store the datasets in the dictionary
                for cluster, (dfx, dfy) in df.items():
                    cluster_key_x = f'{key_x}_cluster_{cluster}'
                    cluster_key_y = f'{key_y}_cluster_{cluster}'
                    cleaned_data_dict[subject][cluster_key_x] = dfx
                    cleaned_data_dict[subject][cluster_key_y] = dfy
                    idxrule_dict[subject][cluster_key_x] = idxrule[cluster]
                    
                    # Plotting the data
                    color = colors[cluster % len(colors)]
                    #marker = markers[cluster % len(markers)]
                    for i in range(dfx.shape[0]):
                        ax.plot(dfx.iloc[i, :], dfy.iloc[i, :], color=color, linestyle='-', linewidth=1, markersize=2,
                                label=f'Cluster {cluster}' if i == 0 else "")
                        # Mark the intersection point
                        if idxrule[cluster]:
                            ax.plot(dfx.iloc[i, idxrule[cluster][i]], dfy.iloc[i, idxrule[cluster][i]], 'kx')
                
                subplot_index += 1
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.savefig(os.path.join(save_dir, f'Subject_{subject}_clustered_trajectories.png'))
        plt.close(fig)
    
    return cleaned_data_dict, idxrule_dict


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

def plot_multiple_data(data_dict : list, 
                       first_subj = 25, last_subj = 37, 
                       save_dir = 'subject_plots', 
                       saving = True):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loop over subjects
    for subject in range(first_subj, last_subj): 
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows, 2 columns of subplots
        fig.suptitle(f'Subject {subject} Trajectories', fontsize=16)
        subplot_index = 0
        for motivation in range(1, 4):
            for mode in range(1, 3):
                ax = axes[subplot_index // 2, subplot_index % 2]
                key_x = f'dfx_{subject}_{motivation}{mode}'
                key_y = f'dfy_{subject}_{motivation}{mode}'
                pic_name = f'dataTrajectories-{subject}-M{motivation}-C{mode}'
                
                plot_data(
                    data_dict[key_x], 
                    data_dict[key_y], 
                    cluster_labels=[],  
                    cluster=None,     
                    n_clusters=None,  
                    plotting_target=False, 
                    saving_plot=False, 
                    pic_name=pic_name,
                    ax=ax  # Pass the subplot axis to plot_data
                )
                ax.set_title(f'Motivation {motivation}, Mode {mode}')
                subplot_index += 1
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save the figure
        if saving: 
            save_path = os.path.join(save_dir, f'Subject_{subject}_Trajectories.png')
            plt.savefig(save_path)
        
        plt.show()