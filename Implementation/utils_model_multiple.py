#########################################
##          GENERAL IMPORTS            ##
#########################################

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
from scipy.stats import ks_2samp 
import os
import warnings
from typing import Tuple, List, Dict
import json
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize_scalar
from scipy import stats
from scipy.stats import norm, shapiro


#########################################
##          CUSTOM IMPORTS             ##
#########################################

from utils_data_multiple import get_cluster_data, get_cluster_idxrule 
from utils_model import ComputeFunctional, ComputeVel 

#########################################
##        NUMERICAL SIMULATION         ##
#########################################


def numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                        sigma = 0.5, gamma = 0.5, epsilon = 0.5, alpha = 0.5,
                        u_0 = (0,0), l_0 = (0,0,0,0), 
                        i_max = 1000, dt = 1./500,
                        Autoregr = True, 
                        Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1): 
    
    ## Set the initial conditions: 
    p1, p2, v1, v2 = x_0
    u1, u2 = u_0
    l1, l2, l3, l4 = l_0
    
    ## Initialize the loop
    i = 0
    t = 0.
    Wt = 0
    l = p_T - p1
    
    ## Initialize the trajectory vector
    p1_ = []
    p2_ = []
    v1_ = []
    v2_ = []
    
    ## Initialize the utility vector 
    u1_ = []
    u2_ = []
     
    ## Model time evolution definition following the system's dynamics
    while (i < i_max and p1 < p_T):
        
        t = i * dt
        if ( Autoregr == True ):
            if np.linalg.norm(sigma) > 0:
                W_increment=np.random.normal(0.,np.sqrt(np.abs(sigma)),1)[0]
            else: 
                W_increment=np.random.normal(0.,np.sqrt(dt),1)[0]
            Wt = Wt + W_increment
        else: 
            if np.linalg.norm(sigma) > 0:
                Wt = np.random.normal(0.,np.sqrt(np.abs(sigma)),1)[0] 
                #Wt = np.random.normal(0.,np.sqrt(dt*sigma),1)[0] 
            else: 
                Wt = np.random.normal(0.,np.sqrt(dt),1)[0] 
            #Wt = np.random.normal(0.,np.sqrt(dt),1)[0] 
           
        ## System's dynamics:
        
        # Control vector
        u1 = - (l * l3) / ( 2 * Wt * l * l2 * sigma * np.exp(t/gamma) - alpha * epsilon + epsilon ) * np.exp(t/gamma)
        u2 = - (l * l4) / (alpha * epsilon) * np.exp(t/gamma)
        # Trajectory evolution differential equations 
        p1 = p1 + dt * v1
        p2 = p2 + dt * (v2 + Wt * sigma * u1**2)
        v1 = v1 + dt * u1
        v2 = v2 + dt * u2 
        # Lagrange multipliers
        l1 = l1
        if (np.linalg.norm(p1 - p_T) < dt * v1):
            l2 = l2 - dt * (2 * p2 * np.exp(- t / gamma)) 
        l3 = l3 - dt * l1
        l4 = l4 - dt * l2 
        # Controller over the via point (circular trajectory)
        if(Arc):
            p1_.append((1+p2)*np.cos(angle*(p1/p_T-1)+angle0))
            p2_.append((1+p2)*np.sin(angle*(p1/p_T-1)+angle0))
        else:
            p1_.append(p1)
            p2_.append(p2)
        v1_.append(v1)
        v2_.append(v2)
        u1_.append(u1)
        u2_.append(u2)
        
        i = i + 1
        
    return np.array(p1_), np.array(p2_), np.array(v1_), np.array(v2_), np.array(u1_), np.array(u2_), t

def arc_length(x, y):
    '''
        Function that calculates the arc length of a curve defined by points (x, y).
    '''
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)

    return arc



#########################################
##         GENERATE TRAJECTORIES       ##
#########################################

def generate_trajectory(params = ( 3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371), 
                        sigma = 0, gamma = 0.5, epsilon = 0.1, alpha = 0.5, timestep=1/500, 
                        plotting = True):
    '''
        Function that computes and plots the trajectory for the initial given parameters 
        (control function and lagrange multipliers) found by optimizing the functional in 
        terms of the parameters. For fixed values of sigma, gamma, epsilon and alpha.
    ''' 
    initial_cond = scipy.optimize.minimize(ComputeFunctional, params, args=(), method=None)
    parameters = initial_cond.x
    x, y, v, w, ux, uy, T= numericalSimulation(x_0 = (0,0,0,0),  p_T = 2.0, 
                        sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters[:2], l_0 = parameters[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = True, 
                        Arc = True, angle=math.pi*12/24, angle0=0, p=(.2,0), r=.1
                        )
    if plotting: 
        plot_trajectory(x,y, showing = True)
    return x, y, T

def generate_trajectory_vel(params = (.5, .5, .5), 
                            parameters = ( 3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371),
                            sigma = 0, timestep=1/500, 
                            plotting = True, T = 1.3, vel = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Function that computes and plots the trajectory for the initial given parameters 
        (sigma, gamma, epsilon and alpha) found by optimizing the velocity in 
        terms of the parameters. 
    ''' 
    
    # Define a partial function to pass additional arguments to ComputeVel
    partial_compute_vel = lambda params: ComputeVel(params, vel=vel, T=T, sigma=sigma)   
    # Use scipy.optimize.minimize with the partial function
    new_params = scipy.optimize.minimize(partial_compute_vel, params, args=(), method=None)
    gamma, epsilon, alpha = new_params.x
    # Call numericalSimulation with the optimized parameters
    x, y, v, w, ux, uy, T= numericalSimulation(x_0 = (0,0,0,0),  p_T = 2.0, 
                        sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters[:2], l_0 = parameters[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = True, 
                        Arc = True, angle=math.pi*12/24, angle0=0, p=(.2,0), r=.1
                        )
    # Plot trajectory if required
    if plotting: 
        plot_trajectory(x, y, showing=True)
        
    return x, y, new_params

def computeSamples(parameters, new_params : np.ndarray = (0,0,0),
                   xT_samples : list = [], n = 50, timestep = 1/500, 
                   parameters2 = (3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371)
                   ) -> np.float64:
    '''
        Kolmogorov Smirnov Test
    '''
    
    sigma = parameters
    gamma, epsilon, alpha = new_params.x
     
    xT2_samples=[]

    for i in range(n):
        x, y, v, w, ux, uy, T2= numericalSimulation(x_0 = (0,0,0,0),  p_T = 2.0, 
                            sigma = sigma*0.012, gamma = gamma, epsilon = epsilon, alpha = alpha,
                            u_0 = parameters2[:2], l_0 = parameters2[2:], 
                            i_max = 1000, dt = timestep,
                            Autoregr = True, 
                            Arc = True, angle=math.pi*12/24, angle0=0, p=(.2,0), r=.1)
        xT2_samples.append(x.flatten()[-1])
            
    return ks_2samp(xT_samples, xT2_samples)[0]

def optimize_Sigma(dfx, dfy, idxrule, new_params, 
                   parameters=(3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371),
                   timestep=1/500, plotting=False):
    xT_samples = []
    dfx.reset_index(drop=True, inplace=True)
    
    for i, row in dfx.iterrows():
        if idxrule[i] > 0:
            xT_samples.append(dfx.loc[i][idxrule[i]-1])
    
    partial_computeSamples = lambda params: computeSamples(params, new_params=new_params, xT_samples=xT_samples)
    
    # Setting bounds and options
    bounds = (0.01, 10)  # Example bounds, adjust as necessary
    tol = 1e-6
    options = {'maxiter': 1000}
    
    warnings.filterwarnings("ignore")
    opt_Sigma = minimize_scalar(partial_computeSamples, bounds=bounds, method='bounded', tol=tol, options=options)
    
    # Check if the optimizer has converged
    #if not opt_Sigma.success:
    #    raise ValueError("Optimization did not converge: ", opt_Sigma.message)
    
    gamma, epsilon, alpha = new_params.x
    sigma = opt_Sigma.x
    
    # Call numericalSimulation with the optimized parameters
    x, y, v, w, ux, uy, T = numericalSimulation(x_0=(0, 0, 0, 0), p_T=2.0, 
                                                sigma=sigma * 0.005, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                u_0=parameters[:2], l_0=parameters[2:], 
                                                i_max=1000, dt=timestep,
                                                Autoregr=True, 
                                                Arc=True, angle=math.pi * 12 / 24, angle0=0, p=(.2, 0), r=.1)
    # Plot trajectory if required
    if plotting: 
        plot_trajectory(x, y, showing=True) 
    
        
    return x, y, opt_Sigma



#########################################
##         PLOTTING TRAJECTORIES       ##
#########################################

def plot_trajectory(x, y, ax=None, showing=True, via=False, plot_title='Simulated Trajectory'):
    '''
        Function that plots the given trajectory (x(t), y(t)). 
    ''' 
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(x, y, color='blue', label=plot_title, alpha=1)
    if via:
        angle = math.pi * 12 / 24
        T_1 = .2
        ax.plot(np.cos(angle * (T_1 - 1)), np.sin(angle * (T_1 - 1)), marker='o', markersize=20)
    
    ax.legend()
    if showing:
        plt.show()

        
def plot_simulation(x : np.ndarray , y : np.ndarray,
                    dfx : pd.DataFrame, dfy : pd.DataFrame,
                    cluster : int, subject : int, 
                    key_: str,
                    pic_name = 'Trajectories', 
                    folder_name = 'pics',
                    saving_plot = False, 
                    show_plot = False): 
    '''
        Function that plots the given trajectory (x(t), y(t)) and 
        the experimental data (dfx, dfy) for the given cluster. 
    '''
    for i in range(len(dfx)):
        plt.plot(dfx.iloc[i], dfy.iloc[i], color='gray', alpha=0.5)
    
    plot_trajectory(x,y, showing = False)
    plt.title(f'Trajectories for subject {subject} in Cluster {cluster}')
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    if saving_plot:
        
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        data_folder = os.path.join(parent_dir, folder_name)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder) 
        
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}_{key_}.png'
        filepath = os.path.join(data_folder, filename)
        plt.savefig(filepath)
    
    if show_plot:
        plt.show()
    else: 
        plt.close() 
    
def plot_multiple_trajectories(dfx: pd.DataFrame, dfy: pd.DataFrame,
                               subject: int, new_params: np.ndarray, opt_Sigma: np.float64,  
                               parameters2=(3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371),
                               n=50, timestep=1/500, pic_name ='Simulated_Trajectories', 
                               ax=None, saving_plot=False, color = 'blue'): 
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ## Plotting experimental data
    for i in range(len(dfx)):
        ax.plot(dfx.iloc[i], dfy.iloc[i], color='gray', alpha=0.5)
        
    ## Plotting numerical simulation
    gamma, epsilon, alpha = new_params.x
    sigma = opt_Sigma.x
    
    for i in range(n):
        x, y, v, w, ux, uy, T = numericalSimulation(x_0=(0,0,0,0), p_T=2.0, 
                                                     sigma=sigma*0.05, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                     u_0=parameters2[:2], l_0=parameters2[2:], 
                                                     i_max=1000, dt=timestep,
                                                     Autoregr=False, 
                                                     Arc=True, angle=math.pi*12/24, angle0=0, p=(.2,0), r=.1)
        ax.plot(x, y, color = color) 
    
    x_, y_, v_, w_, ux_, uy_, T_ = numericalSimulation(x_0=(0,0,0,0), p_T=2.0, 
                                                       sigma=sigma * 0.05, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                       u_0=parameters2[:2], l_0=parameters2[2:], 
                                                       i_max=1000, dt=timestep,
                                                       Autoregr=False, 
                                                       Arc=True, angle=math.pi*12/24, angle0=0, p=(.2,0), r=.1)
    plot_trajectory(x_, y_, ax=ax, showing=False, plot_title='Trajectory with no noise')
    
    ax.set_title('Trajectories for subject {}'.format(subject))
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    if saving_plot:
        # Check if the 'pics' folder exists, if not, create it
        if not os.path.exists('pics'):
            os.makedirs('pics')
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}{subject}.png'
        filepath = os.path.join('pics', filename)
        plt.savefig(filepath)
    
    if ax is None:
        plt.show()
        
    return x_, y_

def generated_multiple_trajectories_plot(scaled_data_dict: dict, params_loaded: dict, 
                            opt_sigma : dict, 
                            first_subj: int = 25, last_subj: int = 37,
                            n_clusters = 4,
                            saving = True, 
                            save_dir: str = 'subject_plots'): 
    

    
    for subject in range(first_subj, last_subj):
        
        print('Simulated and experimental trajectories for subject ', subject )
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))  # 3 rows, 2 columns of subplots
        fig.suptitle(f'Subject {subject} Simulated Trajectories', fontsize=16)
        subplot_index = 0
        pic_name = f'{save_dir}_{subject}'

        colors = ['r', 'g', 'b', 'c']
        
        for motivation in range(1, 4):
            
            for mode in range(1, 3):
                ax = axes[subplot_index // 2, subplot_index % 2]
                
                
                for cluster in range(n_clusters):
                    
                    dfx_, dfy_ = get_cluster_data(scaled_data_dict, subject, motivation, mode, cluster)
                    params = get_cluster_idxrule(params_loaded, subject, motivation, mode, cluster)
                    sigma = get_cluster_idxrule(opt_sigma, subject, motivation, mode, cluster)
                    
                    x_, y_ = plot_multiple_trajectories(dfx_, dfy_, cluster, params, sigma, 
                                               ax=ax, saving_plot=False, pic_name = pic_name,
                                               color = colors[cluster])
                
                subplot_index += 1
                
        fig.tight_layout(rect=[0, 0, 1, 0.96])  
        
        if saving:
            # Check if the 'pics' folder exists, if not, create it
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Save the figure with a specific name based on the cluster
            filename = f'{save_dir}_{subject}.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath)
            #plt.savefig(os.path.join(save_dir, pic_name))
        
        plt.close(fig)
    
    return x_, y_



#########################################
##          SAVING PARAMETERS         ##
#########################################

def saving_new_params(new_params: dict, folder_name: str = 'new_params_data') -> None:
    
    if new_params is None:
        print("No data to save.")
        return

    print('Saving the new parameters...')

    # Get the current directory
    current_dir = os.getcwd()

    # Navigate one step up (to the parent directory)
    parent_dir = os.path.dirname(current_dir)

    # Enter the 'data' folder
    data_folder = os.path.join(parent_dir, folder_name)

    # Check if the folder exists, if not, create it
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for subject, subject_params in new_params.items():
        subject_folder = os.path.join(data_folder, f'subject_{subject}')
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)

        for key, result in subject_params.items():
            # Convert OptimizeResult to a dictionary and save each component
            result_dict = {attr: value.tolist() if isinstance(value, np.ndarray) else value for attr, value in result.items()}
            
            # Define the file path for the JSON
            file_path = os.path.join(subject_folder, f"{key}.json")

            # Save the result dictionary to a JSON file
            with open(file_path, 'w') as f:
                json.dump(result_dict, f)

    print("Parameters JSON files have been saved successfully.")

def load_params(folder_name: str = 'new_params_data') -> dict:
    print('Loading the new parameters...')
    
    # Get the current directory
    current_dir = os.getcwd()

    # Navigate one step up (to the parent directory)
    parent_dir = os.path.dirname(current_dir)

    # Enter the 'data' folder
    data_folder = os.path.join(parent_dir, folder_name)

    # Check if the folder exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"The folder {data_folder} does not exist.")

    new_params = {}

    # Iterate over subjects in the folder
    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if os.path.isdir(subject_path):
            subject_id = int(subject_folder.split('_')[-1])  # Extract subject ID
            new_params[subject_id] = {}
            
            # Iterate over JSON files in the subject folder
            for file_name in os.listdir(subject_path):
                if file_name.endswith('.json'):
                    key = file_name.replace('.json', '')
                    file_path = os.path.join(subject_path, file_name)
                    
                    with open(file_path, 'r') as f:
                        result_dict = json.load(f)
                    
                    # Convert dictionary back to OptimizeResult
                    result = OptimizeResult(**{attr: np.array(value) if isinstance(value, list) else value for attr, value in result_dict.items()})
                    
                    new_params[subject_id][key] = result

    print("New parameters have been loaded successfully.")
    return new_params



#########################################
##         PLOTTING PARAMETERS         ##
#########################################

def dict_to_array(params_dict):
    parameters = []
    ## Recurrent function:   
    def extract_parameters(sub_dict):
        for key in sub_dict:
            value = sub_dict[key]
            if isinstance(value, dict):
                if 'x' in value:
                    parameters.append(value['x'])
                else:
                    extract_parameters(value)
            else:
                try:
                    parameters.append(value.x)
                except AttributeError:
                    parameters.append(value)

    extract_parameters(params_dict)
    
    return np.array(parameters)

def plotting_params(parameters: np.ndarray, barWidth=0.5, 
                    saving_plot = True, folder_name = 'fitted_pics', 
                    pic_name = 'params', 
                    style_label = 'seaborn-whitegrid'):
    
    # Choose the height of the bars
    bars1 = np.nanmean(parameters, axis=0)
 
    # Choose the height of the error bars (bars1)
    yer1 = 2 * np.nanstd(parameters, axis=0)
 
    # The x position of bars
    r1 = np.arange(len(bars1))

    # Configure matplotlib to use LaTeX for rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    plt.style.use(style_label)
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    ax.bar(r1, bars1, width=barWidth, color=bar_colors, edgecolor='black', yerr=yer1, capsize=7)
    
    plt.xticks(r1, [r'$\gamma$', r'$\epsilon$', r'$\alpha$', r'$\sigma$'], fontsize=20)
    plt.ylabel(r'Value', fontsize=14)
    plt.title(r'Parameters Distribution', fontsize=16)
    
    if saving_plot:
        
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        data_folder = os.path.join(parent_dir, folder_name)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder) 
        
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}.png'
        filepath = os.path.join(data_folder, filename)
        plt.savefig(filepath)
    
    plt.show()

def box_plot_params(parameters: np.ndarray, style_label ='seaborn-whitegrid', 
                                     saving_plot = True, folder_name = 'fitted_pics', 
                                     pic_name = 'params_gaussian', 
                                     ):
    num_params = parameters.shape[1]  # Number of parameters (should be 4)
    titles = [r'$\gamma (s^{-1})$', r'$\varepsilon (m^{-1} kg^{-2} s^{4})$', r'$\alpha (kg^{-1})$', r'$\sigma$']
    titles_ = [r'$\gamma $', r'$\varepsilon$', r'$\alpha$', r'$\sigma$']
     
    plt.style.use(style_label)  
    
    fig, axs = plt.subplots(nrows=1, ncols=num_params, figsize=(18, 6))
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i in range(num_params):
        ax = axs[i]
        param_values = parameters[:, i]
        mean = np.mean(param_values)
        std_dev = np.std(param_values)
        
        sns.boxplot(y=param_values, ax=ax, color=bar_colors[i])
        ax.set_title(titles[i])
        ax.set_xlabel(titles_[i])
         
        #ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        #ax.legend(loc = 2, fontsize = 10)

    plt.tight_layout()
    
    if saving_plot:
        
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        data_folder = os.path.join(parent_dir, folder_name)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder) 
        
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}.png'
        filepath = os.path.join(data_folder, filename)
        plt.savefig(filepath)
        
    plt.show()

def plot_gaussian_distributions_theo(parameters: np.ndarray, style_label ='seaborn-whitegrid', 
                                     saving_plot = True,    folder_name = 'fitted_pics', 
                                     pic_name = 'params_gaussian', 
                                     ):
    num_params = parameters.shape[1]  # Number of parameters (should be 4)
    titles = [r'$\gamma (s^{-1})$', r'$\varepsilon (m^{-1} kg^{-2} s^{4})$', r'$\alpha (kg^{-1})$', r'$\sigma$']
    titles_ = [r'$\gamma $', r'$\varepsilon$', r'$\alpha$', r'$\sigma$']
    
    t_tests = []
    sw_tests = []
     
    plt.style.use(style_label)  
    
    fig, axs = plt.subplots(nrows=1, ncols=num_params, figsize=(18, 6))
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(num_params):
        ax = axs[i]
        param_values = parameters[:, i]
        mean = np.mean(param_values)
        std_dev = np.std(param_values)
        
        # Generate x values for the plot
        x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
        # Calculate Gaussian pdf
        y = norm.pdf(x, mean, std_dev)
        
        T_test = stats.ttest_ind(y, param_values, trim=.2)
        t_tests.append(T_test)
        print(titles[i])
        print('T-test results: ', T_test)
        
        # Perform Shapiro-Wilk test for Gaussianity
        shapiro_test = shapiro(param_values)
        sw_tests.append(shapiro_test)
        print(titles[i])
        print('Shapiro-Wilk test results:', shapiro_test)
        
        # Plot the Gaussian distribution
        ax.plot(x, y, color = 'black', label=f'Theoretical Gaussian: $\mu$={mean:.2f}, $std$={std_dev:.2f}')
        ax.hist(param_values, histtype='stepfilled', bins=20, color = bar_colors[i], density=True, alpha=0.7, label='Experimental distribution')
        
        #ax.set_title(f'Parameter {i+1}')
        ax.set_title(titles[i])
        ax.set_xlabel(titles_[i])
         
        #ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(loc = 2, fontsize = 10)

    plt.tight_layout()
    
    if saving_plot:
        
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        data_folder = os.path.join(parent_dir, folder_name)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder) 
        
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}.png'
        filepath = os.path.join(data_folder, filename)
        plt.savefig(filepath)
        
    plt.show()
    return t_tests, sw_tests

def plotting_dict_params(params_dict: dict, opt_sigma: dict,
                         style_label ='seaborn-v0_8-white', 
                         barWidth=0.5, plotting_ = 0,
                         saving_plot = True, folder_name = 'fitted_pics', 
                         pic_name = 'params'): 
    
    params_array = dict_to_array(params_dict)
    sigma_array = dict_to_array(opt_sigma)
    
    sigma_array = sigma_array.reshape(-1, 1)
    combined_params = np.hstack((params_array, sigma_array))
    combined_params[:, 2] *= -1 
    #combined_params[:, 3] *= 0.005 
   
    if plotting_ == 0: 
        box_plot_params(parameters = combined_params, style_label = style_label, 
                                     saving_plot = saving_plot, folder_name = folder_name, 
                                         pic_name = pic_name) 
    elif plotting_ == 1: 
        t_tests, sw_tests = plot_gaussian_distributions_theo(combined_params, style_label = style_label, 
                                         saving_plot = saving_plot, folder_name = folder_name, 
                                         pic_name = pic_name)
        return t_tests, sw_tests
    
    else:    
        plotting_params(combined_params, 
                        style_label = style_label,
                        barWidth=0.5, 
                        saving_plot = saving_plot, folder_name = folder_name, 
                        pic_name = pic_name)
    

#########################################
##          FITTING PARAMETERS         ##
#########################################


def fitParamaters_mult(scaled_data_dict: dict,
                  idxrule_dict: dict, 
                  results_dict: dict,
                  segments:  List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                  first_subj: int = 25, last_subj: int = 37,
                  n_clusters = 4, 
                  folder_name = 'fitted_trajectories',
                  data_folder_name = 'fitted_parameters', 
                  saving = False, 
                  saving_data = True) -> Tuple[list, list]:
    
    new_params = {}
    opt_sigma = {}
    
     
    for subject in range(first_subj, last_subj):
        print('Computing trajectory with optimized velocity for subject ', subject )
        
        for motivation in range(1, 4):
            
            for mode in range(1, 3):
                
                for cluster in range(n_clusters):
                    
                    key_ = f'{subject}_{motivation}{mode}_cluster_{cluster}'
                    # Initialize the dictionaries if not already
                    if subject not in new_params:
                        new_params[subject] = {}
                    if subject not in opt_sigma:
                        opt_sigma[subject] = {}
                        
                    dfx, dfy = get_cluster_data(scaled_data_dict, subject, motivation, mode, cluster)
                    idxrule = get_cluster_idxrule(idxrule_dict, subject, motivation, mode, cluster)
                    results = get_cluster_idxrule(results_dict, subject, motivation, mode, cluster)
                    ## !
                    if(len(dfx) > 0):
                        ## Generate the optimal trajectory by optimizing the Functional in terms of the time T 
                        x, y, T = generate_trajectory(plotting = False)
                        plot_simulation(x, y, dfx, dfy, 
                                    cluster = cluster, subject = subject, 
                                    key_ = key_, 
                                    pic_name = 'Trajectories_optFunctional', 
                                    folder_name = folder_name,
                                    saving_plot = saving)
                        
                        ## Generate the optimal trajectory with the time provided from optimizing the Functional 
                        # by optimizing the velocity in terms of the parameters (alpha, epsilon, gamma)
                        x_, y_, new_params[subject][key_] = generate_trajectory_vel(plotting = False, 
                                                    T = T,
                                                    vel = results[1])
                        plot_simulation(x_, y_, dfx, dfy, 
                                    cluster = cluster,  subject = subject, 
                                    key_ = key_, 
                                    pic_name = 'Trajectories_optVel', 
                                    folder_name = folder_name,
                                    saving_plot = saving)
                        
                        ## Generate the optimal trajectory with the optimum stopping time and parameters
                        # by optimizing the Kolmogorov Sirnov estimate in terms of the sigma

                        idxr = np.array(idxrule)
                        x__, y__, opt_sigma[subject][key_] = optimize_Sigma(dfx, dfy,
                                                            idxrule = idxr, 
                                                            new_params = new_params[subject][key_])
                        plot_simulation(x__, y__, dfx, dfy, 
                                        cluster = cluster,  subject = subject, 
                                        key_ = key_, pic_name = 'Trajectories_optSigma', 
                                        folder_name = folder_name,
                                        saving_plot = saving)
                      
    sigma_folder = f'{data_folder_name}_sigma'
    saving_new_params(new_params, folder_name= data_folder_name)
    saving_new_params(opt_sigma, folder_name= sigma_folder)
                    
    return new_params, opt_sigma