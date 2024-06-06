#########################################
##          GENERAL IMPORTS            ##
#########################################

import random
import pandas as pd
import numpy as np
from typing import Tuple

# Load the custom functions from .py file  
from utils_data_multiple import load_multiple_data, plot_multiple_data, point_to_segment, cleaning_clustering_multiple_data



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
    
    return data_dict

def main(loading = True, n_clusters = 4): 
    if loading: 
        print('Loading processed data...')
        
    else: 
        print('Loading and processing data...')
        data_dict = processing_data()

if __name__ == '__main__':
    main(loading = False)