import numpy as np
import pandas as pd
import patsy.contrasts as cont

'''
Dummy function
'''

def dummy_function(X):
    # Prepare data:
    if len(X.shape) > 1:
        rows, cols = X.shape
        X = list(X.transpose())
        n_factors = len(X)
    else:
        X = X.transpose()
        n_factors = 1
        
        #Make contrast matrix
    contrast = cont.Sum()
        
    
    #for one factor:
    if n_factors == 1:
        
        #number of levels:
        levels = set(X)
        C = contrast.code_with_intercept(list(levels))
        result = C.matrix[X-1,:]
    
    else:            
        #number of levels for each factor
        levels = [list(set(X_i)) 
                  for X_i in X]

        C = [contrast.code_without_intercept(factor).matrix 
             for factor in levels]
        
        temp = [np.asmatrix(np.ones(rows)).transpose()] + [C[i][X[i]-1,:] 
                for i in range(len(C))]
        
        result = np.concatenate(temp, axis =1)
    return np.asarray(result)
'''
PCA function
'''
def PCA(x, n_components):

    #Covariance matrix of X
    cov_mat = np.cov(x , rowvar = False)
     
    #calculate eigenValues and eigen vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Sort based on eigenvalues
    sorted_index = np.argsort(eigen_values)[::-1][:n_components]
    sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    
    #Keep n_components
    eigenvector_subset = sorted_eigenvectors[:,0:n_components]
    
    Loadings = eigenvector_subset*np.sqrt(sorted_eigenvalues)

    Scores = x@eigenvector_subset
    
    explained_variance = sorted_eigenvalues/np.sum(eigen_values)
    
    return Scores, Loadings, explained_variance

'''
Simulates data with specified noise from k1 and k2
'''
# Simulate data
def simulate_data(design, p, k1, k2, k3, seed):
    # Set seed
    np.random.seed(seed)
    n = design.shape[1]
    
    # Make dummy functions
    dummy = dummy_function(design)
    dummy1 = np.random.permutation(dummy_function(design[:,0]))
    dummy2 = np.random.permutation(dummy_function(design[:,1]))
        
    # Make systematic variation
    P = np.random.randn(dummy.shape[1], p)
    Y_sys = (dummy @ P)/n

    # Make Nuisance factor and white noise
    E_nuisance1 = dummy1 @ np.random.randn(dummy1.shape[1], p)
    E_nuisance2 = dummy2 @ np.random.randn(dummy2.shape[1], p)
    E_wn = np.random.randn(design.shape[0],p)

    # Blend data
    E = (k1*E_nuisance1 + k2*E_nuisance2 + k3*E_wn)/3
    Y = Y_sys + E
    
    # Binomial Data
    PR = 1/(1 + np.exp(-Y))
    Y_bin = np.random.binomial(1, PR)
    
    # Poisson Data
    mu = np.exp(Y)
    Y_pois = np.random.poisson(mu)
    
    # systematic to error ratio
    a = 100*(k1+k2+k3)/3
    
    
    return Y, Y_sys, Y_bin, Y_pois, a

'''
autoscale
'''

def autoscale(x):
    y = (x-x.mean(axis=0))/x.std(axis=0)
    return y