#########################################
##          GENERAL IMPORTS            ##
#########################################

import random
import pandas as pd
import numpy as np
from typing import Tuple

# Load the custom functions from .py file  
from utils_data import load_data, data_clustering, plot_data, cleaning_data, linear_transf
from utils_data import experimental_velocity, plot_velocity, saving_processed_data, load_processed_data
from utils_model import generate_trajectory, plot_simulation, generate_trajectory_vel, optimize_Sigma, plot_multiple_trajectories



##############################################
## GENERAL PROCESSING AND FITTING FUNCTIONS ##
##############################################


def processing_data(n_clusters = 4) -> Tuple[pd.DataFrame, list, list]:
    
    dfx, dfy = load_data()
    
    random.seed(10)
    # n_clusters = 4
    
    cluster_datasets, cluster_labels = data_clustering(dfx, dfy)
    
    cluster_points = [
        (10.5, 4.85),   # pt2
        (-11, -5),       # pt3  
        (10.5, 0),      # pt1    
        (-10.5, 1.25)  # pt0
    ]
    rect_df = pd.DataFrame(columns=['cluster', 'rectx', 'recty', 'idxrule', 'mean_T', 'max_vel'])
    
    truncated_dfx = [[] for _ in range(n_clusters)]
    truncated_dfy = [[] for _ in range(n_clusters)]   
    rotated_dfx = [[] for _ in range(n_clusters)]
    rotated_dfy = [[] for _ in range(n_clusters)]
    dfv = [[] for _ in range(n_clusters)] 
   
    # Iterate over each cluster
    for cluster in range(n_clusters):
        # Get the corresponding point for the current cluster
        pt = cluster_points[cluster]
        
        # Call the plot_data function for the current cluster
        rectx, recty = plot_data(dfx, dfy, cluster_labels, 
                cluster, n_clusters, 
                plotting_target=True, saving_plot=False, pt=pt)
        truncated_dfx[cluster] , truncated_dfy[cluster], idxrule = cleaning_data(cluster_datasets[cluster][0], 
                                                               cluster_datasets[cluster][1], 
                                                               rectx, recty)
        # Plotting truncated data
        _ = plot_data(truncated_dfx[cluster] , truncated_dfy[cluster], cluster_labels = [], 
                      cluster = cluster, n_clusters = n_clusters, 
                      plotting_target= True, saving_plot= False, 
                      pt = pt, pic_name= 'Truncated')
        
        # Linear transformation
        rotated_dfx[cluster] , rotated_dfy[cluster] = linear_transf(truncated_dfx[cluster] , truncated_dfy[cluster], 
                                                        rectx, recty)
        
        # Plotting rotated data
        _ = plot_data(rotated_dfx[cluster] , rotated_dfy[cluster], cluster_labels = [], 
                        cluster = cluster, n_clusters = n_clusters, 
                        plotting_target= False, saving_plot= False, 
                        pt = pt, pic_name = 'Translated')
        
        # Computing the velocity 
        dfv[cluster] = experimental_velocity(rotated_dfx[cluster], rotated_dfy[cluster])
        plot_velocity(dfv[cluster], saving_plot = True, pic_name = 'Velocity{}'.format(cluster))
        
        # Compute the mean of the list idxrule: this is the stopping time average (in ms)
        mean_idxrule = sum(idxrule) / len(idxrule)
        T = mean_idxrule / 1000
        vel=dfv[cluster].T.max().mean()
        
        # Concatenate the data to rect_df
        rect_df = pd.concat([rect_df, pd.DataFrame({'cluster': [cluster], 'rectx': [rectx], 'recty': [recty],
                                                    'idxrule': [idxrule], 'mean_T': T, 'max_vel': vel})],
                            ignore_index=True) 
        
        saving_processed_data(rotated_dfx[cluster], folder_name = 'processed_data', file_name = 'cluster{}_dfx'.format(cluster))
        saving_processed_data(rotated_dfy[cluster], folder_name = 'processed_data', file_name = 'cluster{}_dfy'.format(cluster))
    
    saving_processed_data(rect_df, folder_name = 'processed_data', file_name = 'results')
    
    return rect_df, rotated_dfx, rotated_dfy  

def fitParamaters(results : pd.DataFrame, 
                  dfx : list, dfy : list, 
                  n_clusters = 4) -> Tuple[list, list]:
    
    new_params = [[] for _ in range(n_clusters)]
    opt_sigma = [[] for _ in range(n_clusters)]
    
    for cluster in range(n_clusters): 
        
        print('Computing trajectory with optimized velocity for cluser: ', cluster)
        
        ## Generate the optimal trajectory by optimizing the Functional in terms of the time T 
        x, y, T = generate_trajectory(plotting = False)
        plot_simulation(x, y, dfx[cluster], dfy[cluster], 
                    cluster = cluster, pic_name = 'Trajectories_optFunctional', 
                    saving_plot = True)
        
        ## Generate the optimal trajectory with the time provided from optimizing the Functional 
        # by optimizing the velocity in terms of the parameters (alpha, epsilon, gamma)
        x_, y_, new_params[cluster] = generate_trajectory_vel(plotting = False, 
                                 T = T,
                                 vel = results[results['cluster'] == cluster].max_vel.values[0])
        plot_simulation(x_, y_, dfx[cluster], dfy[cluster], 
                    cluster = cluster, pic_name = 'Trajectories_optVel', 
                    saving_plot = True)
        
        ## Generate the optimal trajectory with the optimum stopping time and parameters
        # by optimizing the Kolmogorov Sirnov estimate in terms of the sigma
            # Converting idxrule to array from string
        idxr = results[results['cluster'] == cluster].idxrule.values[0]
        idxrule = np.fromstring(idxr[1: -1], dtype = int, sep = ', ')
        
        x__, y__, opt_sigma[cluster] = optimize_Sigma(dfx[cluster] , dfy[cluster],
                                            idxrule = idxrule, 
                                            new_params = new_params[cluster])
        plot_simulation(x__, y__, dfx[cluster], dfy[cluster], 
                        cluster = cluster, pic_name = 'Trajectories_optSigma', 
                        saving_plot = True)
        
        print('Parameters estimated:')
        print(new_params[cluster].x, opt_sigma[cluster].x)
        
    return new_params, opt_sigma
            
def plot_simulated_trajectories(dfx : list, dfy : list, 
                                new_params : list, opt_sigma : list, 
                                n_clusters = 4) -> None:
    
    for cluster in range(n_clusters):
        plot_multiple_trajectories(dfx[cluster], dfy[cluster], cluster = cluster, 
                               new_params = new_params[cluster], opt_Sigma = opt_sigma[cluster], 
                               pic_name = 'Simulated and experimental data', saving_plot = True)
        
        
        
        
def main(loading = True, n_clusters = 4) -> None: 
    if loading: 
        print('Loading processed data...')
        dfx = [[] for _ in range(n_clusters)]
        dfy = [[] for _ in range(n_clusters)]
        for cluster in range(n_clusters): 
            dfx[cluster] = load_processed_data(folder_path='processed_data', file_name='cluster{}_dfx.csv'.format(cluster))
            dfy[cluster] = load_processed_data(folder_path='processed_data', file_name='cluster{}_dfy.csv'.format(cluster))
        results = load_processed_data(folder_path='processed_data', file_name='results.csv')
    else: 
        print('Loading and processing data...')
        results, dfx, dfy = processing_data()
        
    print('Data loaded and processed :)')
    
    print('Fitting paramaters for the optimized trajectory...')   
    new_params, opt_sigma = fitParamaters(results, dfx, dfy)   
    print('Parameters fitted:')
    print('gamma, epsilon, alpha, sigma= ')
    for cluster in range(n_clusters):
        print(new_params[cluster].x, opt_sigma[cluster].x)
        
    print('Plotting simulated trajectories along with experimental data...')
    for cluster in range(n_clusters):
        plot_multiple_trajectories(dfx[cluster], dfy[cluster], cluster = cluster, 
                               new_params = new_params[cluster], opt_Sigma = opt_sigma[cluster], 
                               pic_name = 'Simulated and experimental data', saving_plot = True)
    
    
          
if __name__ == '__main__':
    main()