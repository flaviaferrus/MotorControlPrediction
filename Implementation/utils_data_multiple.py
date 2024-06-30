#########################################
##          GENERAL IMPORTS            ##
#########################################

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.io
import os
import json
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

def saving_processed_mult_data(cleaned_data_dict: dict = None, idxrule_dict: dict = None, 
                               folder_name: str = 'cleaned_multiple_data') -> None:
    
    if cleaned_data_dict is None and idxrule_dict is None:
        print("No data to save.")
        return
    
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
        
    if cleaned_data_dict is not None:
        for subject, subject_data in cleaned_data_dict.items():
            subject_folder = os.path.join(data_folder, f'subject_{subject}')
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)
            
            for key, df in subject_data.items():
                # Define the file path for the DataFrame
                file_path = os.path.join(subject_folder, f"{key}.csv")
                
                # Save the DataFrame to a CSV file
                df.to_csv(file_path, index=False)
        
    if idxrule_dict is not None:
        for subject, idxrules in idxrule_dict.items():
            subject_folder = os.path.join(data_folder, f'subject_{subject}')
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)
            
            # Save the idxrule_dict for the subject
            idxrule_path = os.path.join(subject_folder, f"idxrule_{subject}.json")
            
            # Convert np.array objects to lists before saving to JSON
            idxrules_list = {key: value.tolist() for key, value in idxrules.items()}
            
            with open(idxrule_path, 'w') as f:
                json.dump(idxrules_list, f)
        
    print("CSV and/or idxrule JSON files have been saved successfully.")

def load_processed_mult_data(folder_name: str = 'cleaned_multiple_data') -> Tuple[dict, dict]:
    
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
    idxrule_dict = {}
    
    # Iterate through each subject's folder
    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if os.path.isdir(subject_path):
            subject_key = int(subject_folder.split('_')[1])
            cleaned_data_dict[subject_key] = {}
            
            # Iterate through each file in the subject's folder
            for file_name in os.listdir(subject_path):
                file_path = os.path.join(subject_path, file_name)
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    key = file_name.split('.csv')[0]
                    cleaned_data_dict[subject_key][key] = df
                elif file_name.startswith('idxrule_') and file_name.endswith('.json'):
                    with open(file_path, 'r') as f:
                        idxrule_dict[subject_key] = json.load(f)
    
    print("CSV and idxrule JSON files have been loaded successfully.")
    return cleaned_data_dict, idxrule_dict

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

def get_cluster_idxrule(idxrule_dict, subject, motivation, mode, cluster):
    idxrule_key = f'dfx_{subject}_{motivation}{mode}_cluster_{cluster}'
    res_key = f'{subject}_{motivation}{mode}_cluster_{cluster}'
   
    if subject in idxrule_dict:
        subject_idxrules = idxrule_dict[subject]
        if idxrule_key in subject_idxrules:
            idxrule = subject_idxrules[idxrule_key]
            return idxrule
        elif res_key in subject_idxrules:
            idxrule = subject_idxrules[res_key]
            return idxrule
        else:
            raise ValueError(f"Index rule for cluster {cluster} not found for subject {subject} with motivation {motivation} and mode {mode}.")
    else:
        raise ValueError(f"Subject {subject} not found in the idxrule dataset.")
    

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
                                      saving = True, 
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
                
                #print('Cleaning and clustering subject data...', pic_name)
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
                
                ax.set_title(f'Motivation {motivation}, Mode {mode}')
    
                subplot_index += 1
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])  
        
        if saving:
            plt.savefig(os.path.join(save_dir, f'Subject_{subject}_clustered_trajectories.png'))
        
        plt.close(fig)
        
    return cleaned_data_dict, idxrule_dict



#########################################
##          PROCESSING DATA            ##
#########################################

def linear_transf(dfx : pd.DataFrame, dfy : pd.DataFrame, 
                  rectx : np.ndarray, recty : np.ndarray, 
                  inverse = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    
    if inverse:
        
        if rectx[0] > 0 and recty[-1] > 0:
            model_origin = np.array((0.5,0.2))
            screen_origin = np.array(((rectx[1]+rectx[0])/2,(recty[1]+recty[0])/2))
            model_target = np.array((np.cos(-math.pi*12/24),np.sin(-math.pi*12/24))) 

        elif rectx[0] < 0 and recty[-1] > 0: 
            model_origin = np.array((0.8,-0.8))
            screen_origin = np.array(((-rectx[1]-rectx[0])/2,(recty[1]+recty[0])/2))
            model_target = np.array((np.cos(-math.pi*12/24),np.sin(-math.pi*12/24))) 
            
        elif rectx[0] > 0 and recty[-1] < 0:
            model_origin = np.array((0.9,-0.8))
            screen_origin = np.array(((rectx[1]+rectx[0])/2,(-recty[1]-recty[0])/2)) 
            model_target = np.array((np.cos(-math.pi*12/24),np.sin(-math.pi*12/24))) 
            
        else: 
            model_origin = np.array((0.5,0.2))
            screen_origin = np.array(((-rectx[1]-rectx[0])/2,(-recty[1]-recty[0])/2))
            model_target = np.array((np.cos(-math.pi*12/24),np.sin(-math.pi*12/24))) 
    
        screen_target = np.array((0,0)) 
        
    else:
    
        if rectx[0] > 0 and recty[-1] > 0:
            model_target = np.array((0.5,1))
            screen_target = np.array(((rectx[1]+rectx[0])/2,(recty[1]+recty[0])/2))
            model_origin = np.array((np.cos(-math.pi*12/24),np.sin(-math.pi*12/24))) 

        elif rectx[0] < 0 and recty[-1] > 0: 
            model_target = np.array((1.5,0))
            dfx = -1 * dfx
            screen_target = np.array(((-rectx[1]-rectx[0])/2,(recty[1]+recty[0])/2))
            model_origin = np.array((np.cos(-math.pi*12/24),np.sin(-math.pi*12/24))) 
            
        elif rectx[0] > 0 and recty[-1] < 0:
            model_target = np.array((1.5,0))
            screen_target = np.array(((rectx[1]+rectx[0])/2,(-recty[1]-recty[0])/2)) 
            model_origin = np.array((np.cos(-math.pi*12/24),np.sin(-math.pi*12/24))) 
            
        else: 
            model_target = np.array((0.5,1))
            screen_target = np.array(((-rectx[1]-rectx[0])/2,(-recty[1]-recty[0])/2))
            model_origin = np.array((np.cos(-math.pi*12/24),np.sin(-math.pi*12/24))) 
    
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
    
    if inverse: 
        # Compute the inverse of the affine transformation
        A= np.linalg.inv(A)
        b = -np.dot(A, b)
    
    dfx_=A[0,0]*dfx+A[0,1]*dfy+b[0]
    dfy_=A[1,0]*dfx+A[1,1]*dfy+b[1]
    
    if inverse: 
        if rectx[0] > 0 and recty[-1] > 0:
            return dfx_, dfy_
        elif rectx[0] < 0 and recty[-1] > 0: 
            dfx_ = -1 * dfx_
            return dfx_, dfy_
        elif rectx[0] > 0 and recty[-1] < 0:
            dfy_ = -1 * dfy_
            return dfx_, dfy_
        else: 
            dfx_ = -1 * dfx_
            dfy_ = -1 * dfy_
            return dfx_, dfy_
    
    else: 
        return dfx_, dfy_
    
def multiple_linear_transf(cleaned_data_dict: dict, idxrule_dict: dict, 
                            segments: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                            first_subj: int = 25, last_subj: int = 37,
                            n_clusters = 4,
                            saving = True, 
                            save_dir: str = 'subject_plots'): 
    
    velocity_dict = {}
    results_dict = {}
    scaled_data_dict = {}
    
    for subject in range(first_subj, last_subj):
        print('Rotating and scaling the trajectories for subject ', subject )
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows, 2 columns of subplots
        fig.suptitle(f'Subject {subject} Rotated Trajectories', fontsize=16)
        subplot_index = 0
        pic_name = f'Scaled Trajectories-{subject}'

        colors = ['r', 'g', 'b', 'c']
        
        for motivation in range(1, 4):
            
            for mode in range(1, 3):
                ax = axes[subplot_index // 2, subplot_index % 2]
                
                for cluster in range(n_clusters):
                    key_ = f'{subject}_{motivation}{mode}_cluster_{cluster}'
                    key_x = f'dfx_{subject}_{motivation}{mode}_cluster_{cluster}'
                    key_y = f'dfy_{subject}_{motivation}{mode}_cluster_{cluster}'
                
                    dfx, dfy = get_cluster_data(cleaned_data_dict, subject, motivation, mode, cluster)
                    idxrule = get_cluster_idxrule(idxrule_dict, subject, motivation, mode, cluster)
                    
                    dfx_, dfy_ = linear_transf(dfx, dfy, segments[cluster][0], segments[cluster][1])
                    dfv = experimental_velocity(dfx_, dfy_)
                    
                    if len(idxrule) > 0:
                        mean_idxrule = sum(idxrule) / len(idxrule)
                    else: 
                        mean_idxrule = 0
                        
                    T = mean_idxrule / 1000
                    vel=dfv.T.max().mean()
                    
                    # Initialize the dictionaries if not already
                    if subject not in velocity_dict:
                        velocity_dict[subject] = {}
                    if subject not in results_dict:
                        results_dict[subject] = {}
                    if subject not in scaled_data_dict:
                        scaled_data_dict[subject] = {}
                        
                    velocity_dict[subject][key_] = dfv
                    results_dict[subject][key_] = np.array([ T, vel ])
                    scaled_data_dict[subject][key_x] = dfx_
                    scaled_data_dict[subject][key_y] = dfy_ 
                    
                    # Plotting the data
                    color = colors[cluster % len(colors)]
                    for i in range(dfx.shape[0]):
                        ax.plot(dfx_.iloc[i, :], dfy_.iloc[i, :], color=color, linestyle='-', linewidth=1, markersize=2,
                                label=f'Cluster {cluster}' if i == 0 else "")
                        
                subplot_index += 1
                
        fig.tight_layout(rect=[0, 0, 1, 0.96])  
        
        if saving:
            plt.savefig(os.path.join(save_dir, pic_name))
        
        plt.close(fig)

    return scaled_data_dict, velocity_dict, results_dict 


#########################################
##          COMPUTING VELOCITY         ##
#########################################

def experimental_velocity(dfx : pd.DataFrame, dfy : pd.DataFrame) -> pd.DataFrame:
    # Computing velocity of each point along the trajectories
    dfx1=dfx.iloc[:,1:]
    dfx1.columns = range(dfx1.shape[1])
    dfvx=dfx1.values-dfx.iloc[:,:-1].values

    dfy1=dfy.iloc[:,1:]
    dfy1.columns = range(dfy1.shape[1])
    dfvy=dfy1.values-dfy.iloc[:,:-1].values

    dfv=np.sqrt(np.square(dfvx)+np.square(dfvy))
    dfv = pd.DataFrame(dfv, index=dfx1.index, columns=dfx1.columns)
    
    return dfv



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
    #ax.grid(True)
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
                       saving = True, style_label ='seaborn-v0_8-white'):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.style.use(style_label)  
    
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


