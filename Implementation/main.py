#########################################
##          GENERAL IMPORTS            ##
#########################################

import random
import pandas as pd
import numpy as np
from typing import Tuple

# Load the custom functions from .py file  
from utils_data import processing_data, load_processed_data
from utils_model import fitParamaters, plot_multiple_trajectories



##############################################
## GENERAL PROCESSING AND FITTING FUNCTIONS ##
##############################################



def plot_simulated_trajectories(dfx : list, dfy : list, 
                                new_params : list, opt_sigma : list, 
                                n_clusters = 4) -> None:
    
    for cluster in range(n_clusters):
        plot_multiple_trajectories(dfx[cluster], dfy[cluster], cluster = cluster, 
                               new_params = new_params[cluster], opt_Sigma = opt_sigma[cluster], 
                               pic_name = 'Simulated and experimental data', saving_plot = True)
        
        
        
        
def main(processing = False, n_clusters = 4) -> None: 
    if processing: 
        print('Loading and processing data...')
        results, dfx, dfy = processing_data()
        
    else: 
        print('Loading processed data...')
        dfx = [[] for _ in range(n_clusters)]
        dfy = [[] for _ in range(n_clusters)]
        for cluster in range(n_clusters): 
            dfx[cluster] = load_processed_data(folder_path='processed_data', file_name='cluster{}_dfx.csv'.format(cluster))
            dfy[cluster] = load_processed_data(folder_path='processed_data', file_name='cluster{}_dfy.csv'.format(cluster))
        results = load_processed_data(folder_path='processed_data', file_name='results.csv')
        
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