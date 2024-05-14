#########################################
##          GENERAL IMPORTS            ##
#########################################

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import ks_2samp 
import os
from typing import Tuple
import warnings



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


def plot_trajectory(x, y, showing = True, via = True, plot_title = 'Simulated Trajectory'):
    '''
        Function that plots the given trajectory (x(t), y(t)). 
    ''' 
    plt.plot(x,y,color='blue', label=plot_title, alpha = 1)
    if via:
        angle=math.pi*7/24
        T_1=.2
        plt.plot(np.cos(angle*(T_1-1)),np.sin(angle*(T_1-1)),marker='o',markersize=35)
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
                               cluster : int, new_params : np.ndarray, opt_Sigma : np.float64,  
                               parameters2 = ( 3.7, -0.15679707,  0.97252444,  0.54660283, -6.75775885, -0.06253371),
                               n = 50, timestep = 1/500, 
                               pic_name = 'Trajectories', 
                               saving_plot = False): 
    
    
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ## Plotting experimental data
    for i in range(len(dfx)):
        plt.plot(dfx.iloc[i], dfy.iloc[i], color='gray', alpha=0.5)
        
    ## Plotting numerical simulation
    gamma, epsilon, alpha = new_params.x
    sigma = opt_Sigma.x
    
    for i in range(n):
        x, y, v, w, ux, uy, T= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                        sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters2[:2], l_0 = parameters2[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = False, 
                        Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1)
        plt.plot(x, y) 
    
    x_, y_, v_, w_, ux_, uy_, T_= numericalSimulation(x_0 = (0,0,0,0),  p_T = 1.0, 
                        sigma = sigma, gamma = gamma, epsilon = epsilon, alpha = alpha,
                        u_0 = parameters2[:2], l_0 = parameters2[2:], 
                        i_max = 1000, dt = timestep,
                        Autoregr = False, 
                        Arc = True, angle=math.pi*7/24, angle0=0, p=(.2,0), r=.1)
    plot_trajectory(x_,y_, showing = False, plot_title= 'Trajectory with no noise')
    
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