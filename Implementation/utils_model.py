#########################################
##          GENERAL IMPORTS            ##
#########################################

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import ks_2samp 
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from dtaidistance import dtw
import os
from typing import Tuple
import warnings

from utils_data import linear_transf



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
##        FUNCTIONS TO OPTIMIZE        ##
#########################################

def ComputeFunctional(parameters, sigma = 0, gamma = 0.5, epsilon = 0.1, alpha = 0.5, timestep=1/500):
    
    x, y, v, w, ux, uy, T= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                        sigma = 0, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters[:2], l_0 = parameters[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = False, 
                        Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1)
    
    trange=np.linspace(0, len(ux)*timestep, num=len(ux))
    force=((1-alpha)*ux**2+alpha*uy**2)*math.e**(trange/gamma)/(2*arc_length(x,y))
    integral=(force[1:]+force[:-1]).sum()*T/(2*(len(ux)-1))

    J=(1-y[-1]**2)*math.e**(-T/gamma)-epsilon*integral
    
    return -J

def ComputeVel(parameters, vel = 0.1, T = 1.3, sigma = 0, gamma = 0.5, epsilon = 0.1, alpha = 0.5, timestep=1/500):
    
    gamma, epsilon, alpha = parameters
    parameters2 = ( 3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371)
    x, y, v, w, ux, uy, T2= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                        sigma = 0, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters2[:2], l_0 = parameters2[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = False, 
                        Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1)
    
    vel_max = np.max(np.sqrt(np.square(v) + np.square(w)))
    
    return (T2 - T)**2 + (vel_max - vel)**2

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
        x, y, v, w, ux, uy, T2= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                            sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                            u_0 = parameters2[:2], l_0 = parameters2[2:], 
                            i_max = 1000, dt = timestep,
                            Autoregr = False, 
                            Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1)
        xT2_samples.append(x.flatten()[-1])
            
    return ks_2samp(xT_samples, xT2_samples)[0]



#########################################
##        PLOTTING FUNCTIONS           ##
#########################################


def plot_trajectory(x, y, showing = True, via = False, plot_title = 'Simulated Trajectory',  ax=None):
    '''
        Function that plots the given trajectory (x(t), y(t)). 
    ''' 
    if ax is None:
        ax = plt.gca()
    ax.plot(x,y,color='blue', label=plot_title, alpha = 1)
    
    if via:
        angle=math.pi*7/24
        T_1=.2
        ax.plot(np.cos(angle*(T_1-1)),np.sin(angle*(T_1-1)),marker='o',markersize=35)
        
    if showing: 
        plt.show()

def plot_simulation(x : np.ndarray , y : np.ndarray,
                    dfx : pd.DataFrame, dfy : pd.DataFrame,
                    cluster : int, pic_name = 'Trajectories', 
                    saving_plot = False): 
    '''
        Function that plots the given trajectory (x(t), y(t)) and 
        the experimental data (dfx, dfy) for the given cluster. 
    '''
    for i in range(len(dfx)):
        plt.plot(dfx.iloc[i], dfy.iloc[i], color='gray', alpha=0.5)
    
    plot_trajectory(x,y, showing = False)
    plt.title('Trajectories in Cluster {}'.format(cluster))
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    if saving_plot:
        # Check if the 'pics' folder exists, if not, create it
        if not os.path.exists('pics'):
            os.makedirs('pics')
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}{cluster}.png'
        filepath = os.path.join('pics', filename)
        plt.savefig(filepath)
    
    plt.show()
    
def plot_multiple_trajectories(dfx : pd.DataFrame, dfy : pd.DataFrame,
                               n_clusters : int, new_params : np.ndarray, opt_Sigma : np.float64, 
                               results : pd.DataFrame, 
                               parameters2 = ( 3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371),
                               ax=None, subject = 25,
                               n = 50, timestep = 1/500, 
                               pic_name = 'Trajectories', pic_folder = 'project_plots',
                               saving_plot = False,
                               inverse = False): 
    
    if ax is None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    fig.suptitle(f'Subject {subject} Simulated Trajectories', fontsize=16)
    subplot_index = 0
    colors = ['r', 'g', 'b', 'c']
       
     
    for cluster in range(n_clusters):
        
        rectx_list = results.rectx[cluster].replace(']', ' ')
        rectx_list = rectx_list.replace('[', ' ')
        rectx_list = rectx_list.split()
        rectx = np.array(rectx_list , dtype= float)

        recty_list = results.recty[cluster].replace(']', ' ')
        recty_list = recty_list.replace('[', ' ')
        recty_list = recty_list.split()
        recty = np.array(recty_list , dtype= float)
        
        ax = axes[subplot_index // 2, subplot_index % 2]
        #fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ## Plotting experimental data
        for i in range(len(dfx[cluster])):
            if inverse:
                dfx_, dfy_ = linear_transf(dfx[cluster], dfy[cluster], rectx, recty, inverse = True)
                ax.plot(dfx_.iloc[i], dfy_.iloc[i], color='gray', alpha=0.5, label='Experimental Trajectories' if i == 0 else "")
            else:
                ax.plot(dfx[cluster].iloc[i], dfy[cluster].iloc[i], color='gray', alpha=0.5, label='Experimental Trajectories' if i == 0 else "")
            
        ## Plotting numerical simulation
        gamma, epsilon, alpha = new_params[cluster]
        sigma = opt_Sigma[cluster]
        
        for i in range(n):
            x, y, v, w, ux, uy, T= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                            sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                            u_0 = parameters2[:2], l_0 = parameters2[2:], 
                            i_max = 1000, dt = timestep,
                            Autoregr = False, 
                            Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1)
            
            if inverse: 
                x_, y_ = linear_transf(x, y, rectx, recty, inverse = True)
                ax.plot(x_, y_, color = colors[cluster], label='Simulated Trajectories' if i == 0 else "") 
            else:
                ax.plot(x, y, color = colors[cluster], label='Simulated Trajectories' if i == 0 else "") 
     
        
        ax.set_title(f'Trajectories in Cluster {cluster}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.legend()
        
        subplot_index += 1
                
    fig.tight_layout(rect=[0, 0, 1, 0.96])  
        
    if saving_plot:
        # Check if the 'pics' folder exists, if not, create it
        if not os.path.exists(pic_folder):
            os.makedirs(pic_folder)
        # Save the figure with a specific name based on the cluster
        filename = f'{pic_name}.png'
        filepath = os.path.join(pic_folder, filename)
        plt.savefig(filepath)
    
    plt.show() 
    
def plot_multiple_trajectories_combined(dfx: pd.DataFrame, dfy: pd.DataFrame,
                                         n_clusters: int, new_params: np.ndarray, opt_Sigma: np.float64,
                                         results: pd.DataFrame,
                                         parameters2=(3.7, -0.15679707, 0.97252444, 0.54660283, -6.75775885, -0.06253371),
                                         subject=25,
                                         n=50, timestep=1/500,
                                         pic_name='Trajectories_combined', pic_folder='project_plots',
                                         saving_plot=False,
                                         inverse=False, 
                                         metrics=True):

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f'Subject {subject} Simulated Trajectories', fontsize=16)
    colors = ['r', 'g', 'b', 'c']
    
    metrics_results = {}

    for cluster in range(n_clusters):

        rectx_list = results.rectx[cluster].replace(']', ' ')
        rectx_list = rectx_list.replace('[', ' ')
        rectx_list = rectx_list.split()
        rectx = np.array(rectx_list, dtype=float)

        recty_list = results.recty[cluster].replace(']', ' ')
        recty_list = recty_list.replace('[', ' ')
        recty_list = recty_list.split()
        recty = np.array(recty_list, dtype=float)

        # Plotting experimental data
        if inverse:
            dfx_, dfy_ = linear_transf(dfx[cluster], dfy[cluster], rectx, recty, inverse=True)
            for i in range(len(dfx[cluster])):
                ax.plot(dfx_.iloc[i], dfy_.iloc[i], color='gray', alpha=0.5, label='Experimental Trajectories' if i == 0 and cluster == 0 else "")
        else:
            for i in range(len(dfx[cluster])):
                ax.plot(dfx[cluster].iloc[i], dfy[cluster].iloc[i], color='gray', alpha=0.5, label='Experimental Trajectories' if i == 0 else "")

        # Plotting numerical simulation
        gamma, epsilon, alpha = new_params[cluster]
        sigma = opt_Sigma[cluster]

        simulated_x = []
        simulated_y = []

        for i in range(n):
            x, y, v, w, ux, uy, T = numericalSimulation(x_0=(0, 0, 0, 0), p_T=1.0,
                                                        sigma=sigma, gamma=gamma, epsilon=epsilon, alpha=alpha,
                                                        u_0=parameters2[:2], l_0=parameters2[2:],
                                                        i_max=1000, dt=timestep,
                                                        Autoregr=False,
                                                        Arc=True, angle=math.pi*7/24, angle0=0, p=(.2, 0), r=.1)

            if inverse:
                x_, y_ = linear_transf(x, y, rectx, recty, inverse=True)
                simulated_x.append(x_)
                simulated_y.append(y_)
                ax.plot(x_, y_, color=colors[cluster], label=f'Simulated Trajectories Cluster {cluster}' if i == 0 else "")
            else:
                simulated_x.append(x)
                simulated_y.append(y)
                ax.plot(x, y, color=colors[cluster], label=f'Simulated Trajectories Cluster {cluster}' if i == 0 else "")
        
        if metrics:
            if cluster not in metrics_results:
                metrics_results[cluster] = {}
            # Padding trajectories to the same length
            simulated_x = pad_trajectories(simulated_x)
            simulated_y = pad_trajectories(simulated_y)

            dfx_padded = pad_or_truncate_trajectories(dfx[cluster].values, simulated_x.shape[1])
            dfy_padded = pad_or_truncate_trajectories(dfy[cluster].values, simulated_y.shape[1])

            metrics = ['MSE', 'RMSE', 'MAE']
            metrics_res = []
            for metric in metrics:
                mse_avg_x, mse_avg_y = compute_mse_between_averages(dfx_padded, dfy_padded, simulated_x, simulated_y, error = metric)
            
                #metrics_res.append({metric : (mse_avg_x, mse_avg_y)})
                metrics_res.append((mse_avg_x, mse_avg_y))
            
            metrics_results[cluster] = metrics_res
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    fig.tight_layout()

    if saving_plot:
        if not os.path.exists(pic_folder):
            os.makedirs(pic_folder)
        filename = f'{pic_name}.png'
        filepath = os.path.join(pic_folder, filename)
        plt.savefig(filepath)

    plt.show()

    if metrics:
        return pd.DataFrame(metrics_results)

    
def plotting_params(parameters : np.ndarray,
                    barWidth = 0.5):
    # Choose the height of the blue bars
    bars1 = parameters.mean(axis=0)
 
    # Choose the height of the error bars (bars1)
    yer1 = 2*parameters.std(axis=0)
 
    # The x position of bars
    r1 = np.arange(len(bars1))

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    
    # Create blue bars
    ax.bar(r1, bars1, width = barWidth, color = ['blue','red','green','orange'], edgecolor = 'black', yerr=yer1, capsize=7,)
    
    # general layout
    plt.xticks(r1, ['gamma', 'epsilon', 'alpha', 'sigma'])
    plt.ylabel('height')    

    
    
#########################################
##        PARAMATER ESTIMATION         ##
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
    x, y, v, w, ux, uy, T= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                        sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters[:2], l_0 = parameters[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = True, 
                        Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1
                        )
    if plotting: 
        plot_trajectory(x,y, showing = True)
    return x, y, T, parameters

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
     
    # Optimize the partial function
    new_params = scipy.optimize.minimize(partial_compute_vel, params, args=(), method=None)
    
    gamma, epsilon, alpha = new_params.x
    
    # Call numericalSimulation with the optimized parameters
    x, y, v, w, ux, uy, T= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                        sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters[:2], l_0 = parameters[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = True, 
                        Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1
                        )
    
    # Plot trajectory if required
    if plotting: 
        plot_trajectory(x, y, showing=True)
        
    return x, y, new_params

def optimize_Sigma(dfx : pd.DataFrame, dfy : pd.DataFrame, idxrule : np.ndarray, 
                 new_params : np.ndarray,
                 parameters = ( 3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371),
                 timestep=1/500, plotting = False) -> Tuple[np.ndarray, np.ndarray, np.float64]:
    '''
        Function that computes and plots the trajectory for the initial given parameters 
        (optimal gamma, epsilon and alpha, and optimized velocity) 
        found by optimizing the Kolmogorov Smirnov estimate in 
        terms of the sigma. 
    ''' 
    
    xT_samples=[]
    dfx.reset_index(drop=True, inplace=True)
    
    for i, row in dfx.iterrows():
        if (idxrule[i]>0):
            xT_samples.append(dfx.loc[i][idxrule[i]-1])
            
    # Define a partial function to pass additional arguments to ComputeVel
    partial_computeSamples = lambda params: computeSamples(params, new_params = new_params,
                                                        xT_samples = xT_samples) 
    opt_Sigma=scipy.optimize.minimize_scalar(partial_computeSamples, bracket=None, bounds=None, args=(), method='golden', tol=None, options=None)
    gamma, epsilon, alpha = new_params.x
    sigma = opt_Sigma.x
    
    # Call numericalSimulation with the optimized parameters
    x, y, v, w, ux, uy, T= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                        sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters[:2], l_0 = parameters[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = False, 
                        Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1
                        )
    # Plot trajectory if required
    if plotting: 
        plot_trajectory(x, y, showing=True) 
        
    return x, y, opt_Sigma



#########################################
##        METRICS COMPUTATION          ##
#########################################

def pad_trajectories(trajectories):
    max_length = max(len(t) for t in trajectories)
    padded_trajectories = np.array([np.pad(t, (0, max_length - len(t)), 'constant', constant_values=np.nan) for t in trajectories])
    return padded_trajectories

def pad_or_truncate_trajectories(trajectories, target_length):
    return np.array([np.pad(t, (0, max(0, target_length - len(t))), 'constant', constant_values=np.nan)[:target_length] for t in trajectories])

def compute_mse_between_averages(dfx: pd.DataFrame, dfy: pd.DataFrame, x_sim: np.ndarray, y_sim: np.ndarray, 
                                 error = 'MSE'):
    avg_x_exp = dfx.mean(axis=0)
    avg_y_exp = dfy.mean(axis=0)
    
    avg_x_sim = np.nanmean(x_sim, axis=0)
    avg_y_sim = np.nanmean(y_sim, axis=0)

    # Truncate or pad the experimental averages to match the simulated length
    if len(avg_x_exp) > len(avg_x_sim):
        avg_x_exp = avg_x_exp[:len(avg_x_sim)]
        avg_y_exp = avg_y_exp[:len(avg_y_sim)]
    else:
        avg_x_exp = np.pad(avg_x_exp, (0, len(avg_x_sim) - len(avg_x_exp)), 'constant', constant_values=np.nan)
        avg_y_exp = np.pad(avg_y_exp, (0, len(avg_y_sim) - len(avg_y_exp)), 'constant', constant_values=np.nan)
    
    if error == 'MSE':
        mse_x = mean_squared_error(avg_x_exp, avg_x_sim)
        mse_y = mean_squared_error(avg_y_exp, avg_y_sim)
    elif error == 'RMSE': 
        mse_x = np.sqrt(mean_squared_error(avg_x_exp, avg_x_sim))
        mse_y = np.sqrt(mean_squared_error(avg_y_exp, avg_y_sim))
    else:
        mse_x = mean_absolute_error(avg_x_exp, avg_x_sim)
        mse_y = mean_absolute_error(avg_y_exp, avg_y_sim) 
        
    return mse_x, mse_y

def compute_means_single_df(df):
    means_list = []
    for j in range(df.shape[1]):
        x_mean = 0
        y_mean = 0
        for i in range(df.shape[0]): 
            x_mean += df[i][j][0]
            y_mean += df[i][j][1]
        x_mean = x_mean / 4
        y_mean = y_mean / 4
        means_list.append((x_mean, y_mean))
        
    return means_list


#########################################
##          FITTING PARAMETERS         ##
#########################################

def fitParamaters(results : pd.DataFrame, 
                  dfx : list, dfy : list, 
                  n_clusters = 4, 
                  plotting = True, saving = False, 
                  pic_name = 'Trajectories') -> Tuple[list, list]:
    
    new_params = [[] for _ in range(n_clusters)]
    opt_sigma = [[] for _ in range(n_clusters)]
    
    for cluster in range(n_clusters): 
        
        print('Computing trajectory with optimized velocity for cluser: ', cluster)
        
        ## Generate the optimal trajectory by optimizing the Functional in terms of the time T 
        x, y, T, parameters_0 = generate_trajectory(plotting = False)
        if plotting:
            plot_simulation(x, y, dfx[cluster], dfy[cluster], 
                    cluster = cluster, pic_name = f'{pic_name}_optFunctional', 
                    saving_plot = saving)
        
        ## Generate the optimal trajectory with the time provided from optimizing the Functional 
        # by optimizing the velocity in terms of the parameters (alpha, epsilon, gamma)
        x_, y_, new_params[cluster] = generate_trajectory_vel(plotting = False, 
                                 T = T,
                                 vel = results[results['cluster'] == cluster].max_vel.values[0])
        if plotting:
            plot_simulation(x_, y_, dfx[cluster], dfy[cluster], 
                    cluster = cluster, pic_name = f'{pic_name}_optVel', 
                    saving_plot = saving)
        
        ## Generate the optimal trajectory with the optimum stopping time and parameters
        # by optimizing the Kolmogorov Sirnov estimate in terms of the sigma
            # Converting idxrule to array from string
        idxr = results[results['cluster'] == cluster].idxrule.values[0]
        idxrule = np.fromstring(idxr[1: -1], dtype = int, sep = ', ')
        
        x__, y__, opt_sigma[cluster] = optimize_Sigma(dfx[cluster] , dfy[cluster],
                                            idxrule = idxrule, 
                                            new_params = new_params[cluster])
        if plotting: 
            plot_simulation(x__, y__, dfx[cluster], dfy[cluster], 
                        cluster = cluster, pic_name = f'{pic_name}_optSigma', 
                        saving_plot = saving)
        
        print('Parameters estimated:')
        print(new_params[cluster].x, opt_sigma[cluster].x)
        
    return new_params, opt_sigma
            
