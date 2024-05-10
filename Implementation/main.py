from utils import load_data, data_clustering, plot_data, cleaning_data, linear_transf
from utils import experimental_velocity, plot_velocity, saving_processed_data, load_processed_data
import random
import pandas as pd
from typing import Tuple

def processing_data() -> Tuple[pd.DataFrame, list, list]:
    
    dfx, dfy = load_data()
    
    random.seed(10)
    n_clusters = 4
    
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
    for cluster in range(4):
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
        
def main(loading = True, n_clusters = 4) -> None: 
    if loading: 
        dfx = [[] for _ in range(n_clusters)]
        dfy = [[] for _ in range(n_clusters)]
        for cluster in range(n_clusters): 
            dfx[cluster] = load_processed_data(folder_path='processed_data', file_name='cluster{}_dfx.csv'.format(cluster))
            dfy[cluster] = load_processed_data(folder_path='processed_data', file_name='cluster{}_dfy.csv'.format(cluster))
        results = load_processed_data(folder_path='processed_data', file_name='results.csv')
    else: 
        results, rotated_dfx, rotated_dfy = processing_data() 
        
    print('Data loaded and processed :)')
          
if __name__ == '__main__':
    main()