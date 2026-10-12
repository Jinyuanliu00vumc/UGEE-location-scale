# Dist_UGEE_func_v4_SE.py
# Author: Jinyuan Liu, Guanchao Wang
# Date: 3/15/2026
# Description: Contains a function to estimate Mixed-effect location scale model parameters.
def ugee_solve_FRM(y_ki_all,ui,t_star,eps=1e-5, max_iter=449):
    
    # --------------------
    # Basic dimensions
    # --------------------
    y_ki = y_ki_all
    K = len(y_ki)
    n = len(y_ki[0])
    if not all(len(yk) == n for yk in y_ki):
     raise ValueError("All y_ki[k] must have the same length n.")
    n_comb = n*(n-1)/2
    n_comb = int(n_comb)
    # covert the list to array
    pair_index = np.array(list(combinations(range(n), 2)))

    # --------------------
    # Parameter blocks
    # --------------------
    p_Beta = 3
    
    p_Eta  = 2 + 3                    # (alpha0, alpha1) + (tau0, tau1, tau2)
    p      = p_Beta + p_Eta
    
    # init
    theta0 = [0.01] * (p - 1) + [0.2] # this was -0.01 and chanegd to 0.01
    Beta_est = [1.1,-2.1,1.1]
    Alpha_est = [-1.1, 0.6]
    Tau_est = [0.5,-0.5,2]
    
    # time design rows
    k_vecs = np.array([[1, k, max(0, k - t_star)] for k in range(1, K + 1)])
    # Helper function to build per-pair quantities under current parameters
    def build_pair_blocks_f1(i1, i2, Beta_est, Alpha_est, Tau_est):
    # Random effect variances per subject
        sigma_b_vec = np.exp(Alpha_est[0] + Alpha_est[1] * ui)  # Length n
        
        # h2 matrix (pair-average of sigma_b^2)
        h2_mat = 0.5 * (np.outer(sigma_b_vec, np.ones(n)) + np.outer(np.ones(n), sigma_b_vec))
        
        # f1, h1, Var(f1) for this pair
        f1_list = []
        h1_list = []
        Var_f1 = np.zeros(K)
        
        for k in range(K):
            k_vec = k_vecs[k, :]
            f1_list.append(0.5 * (np.add.outer(y_ki[k], y_ki[k])))
            h1_list.append(np.dot(Beta_est, k_vec))
        
            # Working Var(f1_k) = 0.25 * sigmaD2 + 0.5 * WS_k
            h2_pair = h2_mat[i1, i2]  # = 0.5 * (sig_b1^2 + sig_b2^2)
            WS_k = np.exp(np.dot(Tau_est, k_vec))
            Var_f1[k] = 0.25 * (2 * h2_pair) + 0.5 * WS_k
        
        return {"f1": f1_list, "h1": h1_list, "Var_f1": Var_f1, "h2_mat": h2_mat}
      
      

    def build_pair_blocks_f23(i1, i2, Beta_est, Alpha_est, Tau_est):
        # Random effect variances per subject
        sigma_b_vec = np.exp(Alpha_est[0] + Alpha_est[1] * ui)
        h2_mat = 0.5 * (np.outer(sigma_b_vec, np.ones(n)) + np.outer(np.ones(n), sigma_b_vec))
        dh2a0 = h2_mat
        dh2a1 = 0.5 * (np.outer(ui * sigma_b_vec, np.ones(n)) + np.outer(np.ones(n), ui * sigma_b_vec))
        
        f2_list = [None] * (K - 1)
        f3_list = [None] * K
        h3_list = [None] * K
        Var_f2 = np.zeros(K - 1)
        Var_f3 = np.zeros(K)
        
        # Build residuals from fixed Beta
        e_list = [None] * K
        h1_list = [None] * K
        for k in range(K):
            k_vec = k_vecs[k, :]
            h1_list[k] = np.dot(Beta_est, k_vec).astype(float)
            e_list[k] = y_ki[k] - h1_list[k]
        
        for k in range(K):
            e_k = e_list[k]
            dist_matrix = squareform(pdist(e_k[:, None])) ** 2  # Pairwise squared distances
            f3_list[k] = 0.5 * dist_matrix
            np.fill_diagonal(f3_list[k], 0)
            
            # Means and working variances for f3
            h2_pair = h2_mat[i1, i2]
            k_vec = k_vecs[k, :]
            WS_k = np.exp(np.sum(Tau_est * k_vec))
            h3_list[k] = h2_pair + WS_k
            
            sigmaD2 = 2 * h2_pair
            v_k2 = sigmaD2 + 2 * WS_k
            Var_f3[k] = 0.5 * (v_k2 ** 2)
            
            # f2: only for k = 1..K-1
            if k < K - 1:
                e_k1 = e_k
                e_k2 = e_list[k + 1]
                diff_k1 = np.subtract.outer(e_k1, e_k1)
                diff_k2 = np.subtract.outer(e_k2, e_k2)
                f2_list[k] = 0.5 * diff_k1 * diff_k2
                np.fill_diagonal(f2_list[k], 0)
                
                # Var(f2_k) = 0.25 * (v_k^2 v_{k+1}^2 + sigmaD2^2)
                k1_vec = k_vecs[k + 1, :]
                WS_k1 = np.exp(np.sum(Tau_est * k1_vec))
                v_k1_2 = sigmaD2 + 2 * WS_k1
                Var_f2[k] = 0.25 * (v_k2 * v_k1_2 + sigmaD2 ** 2)
        
        # Derivatives d(h2,h3)/d(eta)
        diff_h23_Xs = np.zeros(((K - 1) + K, p_Eta))
        for k in range(K):
          # test 
            if k < K - 1:
                diff_h23_Xs[k, :] = np.concatenate((np.array([dh2a0[i1, i2], dh2a1[i1, i2]]), np.zeros(3)))
            k_vec = k_vecs[k, :]
            WS_k = np.exp(np.sum(Tau_est * k_vec))
            diff_h23_Xs[(K - 1) + k, :] = np.concatenate((np.array([dh2a0[i1, i2], dh2a1[i1, i2]]), k_vec * WS_k))
        
        return {
            "f2": f2_list,
            "f3": f3_list,
            "h2_mat": h2_mat,
            "h3": h3_list,
            "Var_f2": Var_f2,
            "Var_f3": Var_f3,
            "diff_h23_Xs": diff_h23_Xs
        }
        
        
        
    
    # ============================================================
    # STEP 1: estimate Beta from f1 only (iterative NR/GEE)
    # ============================================================
    # Initialization
    iter1 = 0
    while True:
        U1 = np.zeros((3, 1))  # Initialize U1
        Wr1 = np.zeros((3, 3))  # Initialize Wr1
    
        for i in range(n_comb):
          #i = 1
          #print(i)
          ind = pair_index[i]
          i1, i2 = ind[0], ind[1]
          
          # Call `build_pair_blocks_f1` function (define this function separately)
          pb = build_pair_blocks_f1(i1, i2, Beta_est, Alpha_est, Tau_est)
          
          # Gi1 and Si1 for this pair
          Gi1 = k_vecs  # K x 3 matrix
          Si1 = np.array([pb["f1"][0][i1, i2],
                          pb["f1"][1][i1, i2],
                          pb["f1"][2][i1, i2]]) - np.array(pb["h1"]).flatten()
                          
          # reshape Si1 so it works with matrix multiplication
          Si1 = Si1.reshape((3, 1))               
                          
          Vi1 = np.diag(pb["Var_f1"])  # K x K diagonal matrix
          
          # Update U1 and Wr1
          U1 += Gi1.T @ pinv(Vi1) @ Si1
          Wr1 += Gi1.T @ pinv(Vi1) @ Gi1
  
  
          
          
          
        # Update Beta_new
        Beta_new = Beta_est + pinv(Wr1) @ U1.flatten()
    
        # Check convergence
        if np.max(np.abs(Beta_new - Beta_est)) < eps:
            break
    
        Beta_est = Beta_new
        iter1 += 1
    
        # Check for maximum iterations
        if iter1 > max_iter:
            print("Mean-step (Beta) did not converge.")
            break
          
          
    # ============================================================
    # STEP 2: estimate Eta=(Alpha,Tau) from f2,f3 with Beta fixed
    # ============================================================
    iter2 = 0
    while True:
        U23 = 0  # Initialize U23
        Wr23 = 0  # Initialize Wr23
    
        for i in range(n_comb):
            #i = 0
            ind = pair_index[i]
            i1, i2 = ind[0], ind[1]
            #print(i)
            # Call `build_pair_blocks_f23` function (define this function separately)
            pv = build_pair_blocks_f23(i1, i2, Beta_est, Alpha_est, Tau_est)
            
            # Compute Si2 (K-1) and Si3 (K), subtract means
            Si2 = np.array([pv["f2"][0][i1, i2],
                            pv["f2"][1][i1, i2]]) - np.repeat(pv["h2_mat"][i1, i2], K - 1)
            Si3 = np.array([pv["f3"][0][i1, i2],
                            pv["f3"][1][i1, i2],
                            pv["f3"][2][i1, i2]]) - np.array(pv["h3"]).flatten()
            Si23 = np.concatenate((Si2, Si3))  # Concatenate Si2 and Si3
            
            # Compute Gi23 and Vi23
            Gi23 = pv["diff_h23_Xs"]  # (K-1 + K) x 5 matrix
            Vi23 = np.diag(np.concatenate((pv["Var_f2"], pv["Var_f3"])))  # (K-1 + K) x (K-1 + K) diagonal matrix
            
            # Update U23 and Wr23
            U23 += Gi23.T @ pinv(Vi23) @ Si23
            Wr23 += Gi23.T @ pinv(Vi23) @ Gi23
    
        # Update Theta_til_new
        Theta_til = np.concatenate((Alpha_est, Tau_est)) 
        lambda_value = 1
        Theta_til_new = Theta_til + lambda_value * pinv(Wr23) @ U23.flatten()
    
        # Check convergence
        if np.max(np.abs(Theta_til_new - Theta_til)) < eps:
            break
    
        # Update Alpha_est and Tau_est
        Alpha_est = Theta_til_new[:2]
        Tau_est = Theta_til_new[2:]
        iter2 += 1
    
        # Check for maximum iterations
        if iter2 > max_iter:
            print("Variance-step (Eta) did not converge.")
            break
          
    # ============================================================
    # Sandwich variance (independence working)
    # ============================================================
    # Rebuild Wr1 and Wr23 at final estimates for B
    
    # Initialize Wr1 and Wr23
    Wr1 = 0
    Wr23 = 0
    
    # Loop through combinations
    for i in range(n_comb):  # Assuming `n_comb` is defined
        #i = 0
        ind = pair_index[i]
        i1, i2 = ind[0], ind[1]
    
        pb = build_pair_blocks_f1(i1, i2, Beta_est, Alpha_est, Tau_est)
        Gi1 = k_vecs
        Vi1 = np.diag(pb["Var_f1"])
        Wr1 += Gi1.T @ pinv(Vi1) @ Gi1
    
        pv = build_pair_blocks_f23(i1, i2, Beta_est, Alpha_est, Tau_est)
        Gi23 = pv['diff_h23_Xs']
        Vi23 = np.diag(np.concatenate([pv["Var_f2"], pv["Var_f3"]]))
        Wr23 += Gi23.T @ pinv(Vi23) @ Gi23
    
    Wr = block_diag(Wr1, Wr23)
    B = Wr / n_comb
    
    # Empirical Sigma.U
    Sigma_U = np.zeros((p, p))
    
    for i in range(n):  
        #i = 0
        vi_vec = np.zeros((p,1))
        others = [j for j in range(n) if j != i]
    
        for j in others:
            i1, i2 = min(i, j), max(i, j)
    
            pb = build_pair_blocks_f1(i1, i2, Beta_est, Alpha_est, Tau_est)
            pv = build_pair_blocks_f23(i1, i2, Beta_est, Alpha_est, Tau_est)
    
            Gi1 = k_vecs
            Vi1 = np.diag(pb['Var_f1'])
            Si1 = np.array([pb['f1'][0][i1, i2], pb['f1'][1][i1, i2], pb['f1'][2][i1, i2]]) - np.array(pb['h1'])
    
            Gi23 = pv['diff_h23_Xs']
            Vi23 = np.diag(np.concatenate([pv["Var_f2"], pv["Var_f3"]]))
            Si2 = np.array([pv['f2'][0][i1, i2], pv['f2'][1][i1, i2]]) - np.array([pv['h2_mat'][i1, i2]] * (K - 1))
            Si3 = np.array([pv['f3'][0][i1, i2], pv['f3'][1][i1, i2], pv['f3'][2][i1, i2]]) - np.array(pv['h3'])
    
            Vi123 = block_diag(Vi1, Vi23)
            Gi123 = block_diag(Gi1, Gi23)
            Si123 = np.concatenate([Si1, Si2, Si3])
            
            add = (Gi123.T @ pinv(Vi123) @ Si123) / (n - 1)
            add_new = add.reshape((8,1))
            vi_vec = vi_vec + add_new
            
            
        Sigma_U += (4 * vi_vec @ vi_vec.T) / (n - 1)
    
    Sigma = pinv(B) @ Sigma_U @ pinv(B) / n
    sigma_est = np.diag(Sigma)
    
    return {
        "theta_est": np.concatenate([Beta_est, Alpha_est, Tau_est]),
        "Sigma_est": Sigma,
        "sigma_est": sigma_est,
        "iter_beta": iter1,  # Assuming `iter1` is defined
        "iter_eta": iter2,   # Assuming `iter2` is defined
        "converged": iter1 <= max_iter and iter2 <= max_iter  # Assuming `max_iter` is defined
    }
