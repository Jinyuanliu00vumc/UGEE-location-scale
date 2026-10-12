# LSM_II_Data_Gene_Once_v4.py
# Author: Jinyuan Liu, Guanchao Wang
# Date: 3/15/26
# Description: Simulates random numerical data and calls the UGEE_LSM function.

# run the following two in R console if using reticulate package in R, then run the code in python console
#library(reticulate)
# directly run the script if using python environment IDLE

## rquire python modules in R
#py_require("pandas")
#py_require("matplotlib")
#py_require("seaborn")
#py_require("statsmodels")
#py_require(scipy)
#py_require("itertolls")
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import itertools as itertools
import csv
import json
import os
from statsmodels.formula.api import mixedlm

from itertools import combinations
from numpy.linalg import solve
from scipy.linalg import pinv
from scipy.linalg import inv
from scipy.linalg import block_diag
from scipy.spatial.distance import pdist, squareform
import random
from IPython import get_ipython
from scipy.spatial.distance import pdist, squareform

np.set_printoptions(suppress=True)


####################################
###### 0.2.1 Data Generation #######
## simulate data based on the LSM ##
####################################
# Create multiple n for simulation
#n_values = [300] #, 300, 500]
# Set preferred Number of repetitions
num_repetitions = 150 
results = []

# initialize constants
p_u = 0.5


Beta0 = [1, -2, 1]
Alpha0 = [-1, 2]
Tau0 = [-0.5, 2]


sigma_w = 0.5
t_star = 2
K = 3


# set preferred sample size
n = 100

for i in range(num_repetitions):
    random.seed(1)
    
    wi = np.random.normal(0, sigma_w, n)
    
    ## time-invariant cov u_i
    # Generate random binary values ui ~ Binomial(1, p_u)
    ui = np.random.binomial(1, p_u, n)
    
    ## compute between- and within-subject variances
    ## time-k-invariant between-subject variances
    b_cov_vec = np.column_stack((np.ones(n), ui))
    Sigma_b_i = np.exp(Alpha0 @ b_cov_vec.T)
    Sigma_e_ki = []
    
    
    # Loop through values of k
    # Loop through 1 to K (1-based indexing)
    for k in range(1, K+1):  # Adjust for 1-based indexing
        k_pl = max(k - t_star, 0)  # Equivalent to R's `pmax(k - t_star, 0)`
        k_vec = np.array([1, k, k_pl])  # Create the vector [1, k, k_pl]
    
        # Perform matrix multiplication and exponentiation
        # Tau0 is applied to `k_vec[-1]` (the last two elements of k_vec)
        exp_value = np.exp(np.dot(Tau0, k_vec[1:]) + wi)
    
        # Append the result to Sigma_e_ki
        Sigma_e_ki.append(exp_value)
    #print(Sigma_e_ki)    
    
    #print(Sigma_b_i)
    # np.mean(Sigma_e_ki[0])
    # np.mean(Sigma_e_ki[1])
    # np.mean(Sigma_e_ki[2])
    # 
    # np.exp(0.5*sigma_w**2)
    
    ## generate the random location effect bi ∼ N(0,σ2 )
    b_i = np.random.normal(0, np.sqrt(Sigma_b_i), n)
    ## and error term εik ∼ N(0,σε2 )
    e_ki = []
    for k in range(K):
        e_k = np.random.normal(0, np.sqrt(Sigma_e_ki[k]), n)
        e_ki.append(e_k)
    
    
    y_ki = []
    
    # Loop over k
    for k in range(1, K + 1):  # Python index starts at 0, so use range(1, K+1) to match R
        k_pl = max(k - t_star, 0)  # Compute k_pl (pmax equivalent)
        k_vec = np.array([1, k, k_pl])  # Create k_vec as a NumPy array
        
        # Compute y_ki for this k
        # Beta0 @ k_vec is the dot product of Beta0 and k_vec
        y_k = (Beta0 @ k_vec) + b_i + e_ki[k - 1]  # Use k-1 because Python is 0-indexed
        
        # Append the result for this k to y_ki
        y_ki.append(y_k)
      
    # Function call
    # start_time = time.time()    
    UGEE_LSM = ugee_solve_FRM(
         y_ki,  # List of within-subject outcome
         ui, # Between-subject covariates
         t_star,
         eps=1e-5,  # Tolerance for convergence
         max_iter=449  # Maximum number of iterations 
    )
    #end_time = time.time()
    #print(f"Execution time for one iteration: {end_time - start_time:.2f} seconds")

    #Save the iteration results
    theta_est = UGEE_LSM["theta_est"]  # 3Beta 2Alpha 3Tau
    sigma_est = UGEE_LSM["sigma_est"]
    Sigma_est = UGEE_LSM["Sigma_est"]
    converge = UGEE_LSM["converged"]
    
    
    results.append({
        "theta_est": theta_est,
        "sigma_est": sigma_est,
        "Sigma_est": Sigma_est,
        "converge": converge
    })


#print(f"Completed {num_repetitions} repetitions. First result:")
current_dir = os.getcwd() 
file_path = os.path.join(current_dir, "results300itr.json")

print(f"Saving to: {file_path}")

with open(file_path, 'w') as f:
    json.dump(results, f, indent=4,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# Extract all `theta_est`, `sigma_est`, and `Sigma_est` values across iterations
theta_est_list = [result["theta_est"] for result in results]
sigma_est_list = [result["sigma_est"] for result in results]
Sigma_est_list = [result["Sigma_est"] for result in results]

# Convert lists to numpy arrays for easier computation
theta_est_array = np.array(theta_est_list)  # Shape: (300, len(theta_est))
sigma_est_array = np.array(sigma_est_list)  # Shape: (300, 1)
Sigma_est_array = np.array(Sigma_est_list)  # Shape: (300, len(Sigma_est))

# Calculate means
theta_mean = np.mean(theta_est_array, axis=0)  # Mean of `theta_est` across iterations
sigma_mean = np.mean(sigma_est_array, axis=0)  # Mean of `sigma_est` across iterations
Sigma_mean = np.mean(Sigma_est_array, axis=0)  # Mean of `Sigma_est` across iterations

# Calculate variances
theta_variance = np.var(theta_est_array, axis=0)  # Variance of `theta_est` across iterations
sigma_variance = np.var(sigma_est_array, axis=0)  # Variance of `sigma_est` across iterations
Sigma_variance = np.var(Sigma_est_array, axis=0)  # Variance of `Sigma_est` across iterations

# get alpha_hat
# alpha_hats = theta_est_array[:, 3:5] 
# for i in range(alpha_hats.shape[1]):  # Iterate over the 2 components of alpha_hat
#     plt.figure(figsize=(8, 5))
#     
#     # Extract values for the i-th alpha_hat component
#     component_values = alpha_hats[:, i]
#     
#     # Plot histogram and density plot
#     sns.histplot(component_values, kde=True, bins=30, color="blue", alpha=0.6)
#     plt.title(f"Distribution of alpha_hat[{i}]")
#     plt.xlabel(f"alpha_hat[{i}]")
#     plt.ylabel("Density")
#     plt.grid(True)
#     
#     plt.show()  
    
# Print results
print("Mean of theta_est:", theta_mean)
print("Variance of theta_est:", theta_variance)

print("Mean of sigma_est:", sigma_mean)
print("Variance of sigma_est:", sigma_variance)

print("Mean of Sigma_est:", Sigma_mean)
print("Variance of Sigma_est:", Sigma_variance)




