#########################################
##          GENERAL IMPORTS            ##
#########################################

import random
import pandas as pd
import numpy as np
from typing import Tuple

# Load the custom functions from .py file  
from utils_data_multiple import load_multiple_data, plot_multiple_data, point_to_segment, cleaning_clustering_multiple_data
from utils_data_multiple import saving_processed_mult_data, load_processed_mult_data, multiple_linear_transf
from utils_model_multiple import fitParamaters_mult, saving_new_params

##############################################
## GENERAL PROCESSING AND FITTING FUNCTIONS ##
##############################################

def processing_data(n_clusters = 4, path = 'dataTrajectories') -> Tuple[pd.DataFrame, list, list]:
    
    data_dict = load_multiple_data(first_subj = 25, last_subj = 37, 
                       file_names = path )
    plot_multiple_data(data_dict)
    
    cluster_points = [
        (10.5, 4.85),   # pt2
        (-11, -5),       # pt3  
        (10.5, 0),      # pt1    
        (-10.5, 1.25)  # pt0
    ]

    segments = point_to_segment(cluster_points, n_clusters)
    
    cleaned_data_dict, idxrule_dict = cleaning_clustering_multiple_data(data_dict, 
                                      segments, 
                                      first_subj = 25, last_subj = 37,
                                      save_dir = 'subject_plots')
    
    saving_processed_mult_data(cleaned_data_dict = cleaned_data_dict, idxrule_dict = idxrule_dict,
                               folder_name = 'clustered_multiple_data')
    
    scaled_data_dict, velocity_dict, results_dict = multiple_linear_transf(cleaned_data_dict, idxrule_dict, 
                            segments, 
                            first_subj = 25, last_subj = 37,
                            n_clusters = 4,
                            saving = True, 
                            save_dir = 'subject_plots_2')
    
    saving_processed_mult_data(cleaned_data_dict = scaled_data_dict, idxrule_dict = results_dict, 
                               folder_name = 'scaled_multiple_data')
    saving_processed_mult_data(cleaned_data_dict = velocity_dict, 
                               folder_name = 'velocity')
    
    return scaled_data_dict, results_dict, idxrule_dict, segments 

def main(processing = False, n_clusters = 4): 
    
    if processing: 
        print('Loading and processing data...')
        scaled_data_dict, results_dict, idxrule_dict, segments = processing_data()
        
    else:   
        print('Loading processed data...')
        scaled_data_dict, results_dict = load_processed_mult_data(folder_name='scaled_multiple_data') 
        cleaned_data_dict, idxrule_dict = load_processed_mult_data(folder_name='clustered_multiple_data')
        
        cluster_points = [
            (10.5, 4.85),   # pt2
            (-11, -5),       # pt3  
            (10.5, 0),      # pt1    
            (-10.5, 1.25)  # pt0
        ]
        segments = point_to_segment(cluster_points, n_clusters)
        
        
    print('Data loaded and processed :)')
    print('Fitting paramaters for the optimized trajectory...')   
    
    new_params, opt_sigma = fitParamaters_mult(scaled_data_dict,
                  idxrule_dict, 
                  results_dict,
                  segments, 
                  first_subj = 25, last_subj = 30,
                  n_clusters = 4, folder_name = 'fitted_trajectories_', 
                  saving = True)   
    
    saving_new_params(new_params, folder_name= 'fitted_params')
    saving_new_params(opt_sigma, folder_name= 'fitted_params_sigma')
      

if __name__ == '__main__':
    main(loading = False)