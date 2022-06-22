import numpy as np
import pandas as pd
import patsy.contrasts as cont
from pyDOE2 import fullfact
from ASCA.GLMSCA_Class import GLMSCA
from ASCA.ASCA_Class import ASCA
from ASCA.Simulation_functions import PCA, simulate_data, dummy_function, autoscale
import time


'''
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Test:
'''

# Set parameters
k = 3
n = 100
p = 50
n_iter = 50
design = np.tile(fullfact([k,k]), (n,1)).astype('int')

# Make nuisance, error and DOE based on nuisance and error
nuisance = np.arange(start = 0, stop = .5, step = 0.05)
error = np.arange(start = 0, stop = .4, step = .1)

DOE = fullfact([n_iter, len(nuisance), len(nuisance), len(error)])
DOE[:,1] = nuisance[DOE[:,1].astype('int')]
DOE[:,2] = nuisance[DOE[:,2].astype('int')]
DOE[:,3] = error[DOE[:,3].astype('int')]

# Dataframes for results
col = ['Error', 'ASCA', 'GLMSCA_anscombe', 'GLMSCA_deviance', 'GLMSCA_pearson', 'GLMSCA_response', 'GLMSCA_working', 'GLMSCA_quantile']
# Poisson results
results_poisson_error = pd.DataFrame(np.empty([DOE.shape[0], len(col)]), columns = col)
results_poisson_mu = pd.DataFrame(np.empty([DOE.shape[0], 2]), columns= ['ASCA', 'GLMSCA'])
results_poisson_mu_relative = pd.DataFrame(np.empty([DOE.shape[0], 2]), columns= ['ASCA', 'GLMSCA'])
results_poisson_lambda = pd.DataFrame(np.empty([DOE.shape[0], len(col)]), columns = col)
results_poisson_mu_SS = pd.DataFrame(np.empty([DOE.shape[0], 3]), columns= ['ASCA', 'GLMSCA_mu', 'GLMSCA_eta'])

# Binomial results
results_bin_error = pd.DataFrame(np.empty([DOE.shape[0], len(col)]), columns = col)
results_bin_mu = pd.DataFrame(np.empty([DOE.shape[0], 2]), columns= ['ASCA', 'GLMSCA'])
results_bin_mu_relative = pd.DataFrame(np.empty([DOE.shape[0], 2]), columns= ['ASCA', 'GLMSCA'])
results_bin_lambda = pd.DataFrame(np.empty([DOE.shape[0], len(col)]), columns = col)
results_bin_mu_SS = pd.DataFrame(np.empty([DOE.shape[0], 3]), columns= ['ASCA', 'GLMSCA_mu', 'GLMSCA_eta'])

Fails = pd.Series(np.empty(DOE.shape[0]))
t0 = time.time()
for i in range(DOE.shape[0]):
    print(f'model {i+1} / {DOE.shape[0]}')
    try:
        # Set k1, k2 and k3 based on DOE
        k1 = DOE[i,1]
        k2 = DOE[i,2]
        k3 = DOE[i,3]

        # Simulate data
        Y, Y_sys, Y_bin, Y_pois, a = simulate_data(design, p, k1, k2, k3, i)
        
        # Poisson
        # ASCA model
        m1 = ASCA(design, Y_pois)
        m1.fit()
        
        # Comparisson of error
        ssr = np.sum(np.square(m1._results.residuals))
        results_poisson_error.loc[i,'ASCA'] = ssr
        
        # Comparison of mu
        mu_pois = np.exp(Y_sys)
        results_poisson_mu.loc[i,'ASCA'] = 100*np.sum(np.square(mu_pois - m1._results.mu))/np.sum(np.square(mu_pois))
        results_poisson_mu_SS.loc[i,'ASCA'] = np.sum(np.square(m1._results.mu))
        
        # Comparison of mu relative
        results_poisson_mu_relative.loc[i,'ASCA'] = 100*np.sum(np.square(autoscale(mu_pois) - autoscale(m1._results.mu)))/np.sum(np.square(autoscale(mu_pois)))
        
        # Wilks Lambda
        scores = m1._results.sca_results[0][0].iloc[:,:(k-1)]
        results_poisson_lambda.loc[i,'ASCA'] = MANOVA(scores,design[:,0]).mv_test().summary_frame.iloc[0,0]
    
        
        # GLMSCA model
        m2 = GLMSCA(design, Y_pois)
        m2.Options.dist = 'Poisson'
        m2.fit()
        
        # Comparisson of error
        for residual in ['anscombe', 'deviance', 'pearson', 'response', 'working', 'quantile']:
            res = getattr(m2._results.residuals, residual)
            ssr = np.sum(np.square(res))
            results_poisson_error.loc[i,f'GLMSCA_{residual}'] = ssr
            
        # Comparison of mu
        results_poisson_mu.loc[i,'GLMSCA'] = 100*np.sum(np.square(mu_pois - m2._results.mu))/np.sum(np.square(mu_pois))
        results_poisson_mu_SS.loc[i,'GLMSCA_mu'] = np.sum(np.square(m2._results.mu))
        results_poisson_mu_SS.loc[i,'GLMSCA_eta'] = np.sum(np.square(m2._results.eta))
        
        # Comparison of mu relative
        results_poisson_mu_relative.loc[i,'GLMSCA'] = 100*np.sum(np.square(autoscale(mu_pois) - autoscale(m2._results.mu)))/np.sum(np.square(autoscale(mu_pois)))
        
        # Wilks Lambda
        for residual in ['anscombe', 'deviance', 'pearson', 'response', 'working', 'quantile']:    
            scores = getattr(m2._results.sca_results, residual)[0][0].iloc[:,:(k-1)]
            results_poisson_lambda.loc[i, f'GLMSCA_{residual}'] = MANOVA(scores, design[:,0]).mv_test().summary_frame.iloc[0,0]

        
        
        # Binomial
        # ASCA model
        m1 = ASCA(design, Y_bin)
        m1.fit()
        
        # Comparisson of error
        ssr = np.sum(np.square(m1._results.residuals))
        results_bin_error.loc[i,'ASCA'] = ssr
        
        # Comparison of mu
        mu_bin = 1/(1 + np.exp(-Y_sys))
        results_bin_mu.loc[i,'ASCA'] = 100*np.sum(np.square(mu_bin - m1._results.mu))/np.sum(np.square(mu_bin))
        results_bin_mu_SS.loc[i,'ASCA'] = np.sum(np.square(m1._results.mu))
        
        # Comparison of mu relative
        results_bin_mu_relative.loc[i,'ASCA'] = 100*np.sum(np.square(autoscale(mu_bin) - autoscale(m1._results.mu)))/np.sum(np.square(autoscale(mu_bin)))
        
        # Wilks Lambda
        scores = m1._results.sca_results[0][0].iloc[:,:(k-1)]
        results_bin_lambda.loc[i,'ASCA'] = MANOVA(scores,design[:,0]).mv_test().summary_frame.iloc[0,0]
    
    
        # GLMSCA model
        m2 = GLMSCA(design, Y_bin)
        m2.Options.dist = 'Binomial'
        m2.fit()
        
        # Comparisson of error
        for residual in ['anscombe', 'deviance', 'pearson', 'response', 'working', 'quantile']:
            res = getattr(m2._results.residuals, residual)
            ssr = np.sum(np.square(res))
            results_bin_error.loc[i,f'GLMSCA_{residual}'] = ssr
            
        # Comparison of mu
        results_bin_mu.loc[i,'GLMSCA'] = 100*np.sum(np.square(mu_bin - m2._results.mu))/np.sum(np.square(mu_bin))
        results_bin_mu_SS.loc[i,'GLMSCA_mu'] = np.sum(np.square(m2._results.mu))
        results_bin_mu_SS.loc[i,'GLMSCA_eta'] = np.sum(np.square(m2._results.eta))
        
        # Comparison of mu relative
        results_bin_mu_relative.loc[i,'GLMSCA'] = 100*np.sum(np.square(autoscale(mu_bin) - autoscale(m2._results.mu)))/np.sum(np.square(autoscale(mu_bin)))
        
        # Wilks Lambda for each residual from GLMSCA scores
        for residual in ['anscombe', 'deviance', 'pearson', 'response', 'working', 'quantile']:    
            scores = getattr(m2._results.sca_results, residual)[0][0].iloc[:,:(k-1)]
            results_bin_lambda.loc[i, f'GLMSCA_{residual}'] = MANOVA(scores, design[:,0]).mv_test().summary_frame.iloc[0,0]

        
        Fails[i] = False
    except:
        Fails[i] = True
        print('Something failed')

t1 = time.time()
total_time = (t1-t0)/60

# Save        
results_poisson_error.to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/results_poisson_error_{n_iter}.csv', index = False)
results_poisson_mu.to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/results_poisson_mu_{n_iter}.csv', index = False)
results_poisson_mu_relative.to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/results_poisson_mu_relative_{n_iter}.csv', index = False)
results_poisson_lambda.to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/results_poisson_lambda_{n_iter}.csv', index = False)

results_bin_error.to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/results_bin_error_{n_iter}.csv', index = False)
results_bin_mu.to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/results_bin_mu_{n_iter}.csv', index = False)
results_bin_mu_relative.to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/results_bin_mu_relative_{n_iter}.csv', index = False)
results_bin_lambda.to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/results_bin_lambda_{n_iter}.csv', index = False)

pd.DataFrame(DOE[:,1:], columns = ['Nuisance1', 'Nuisance2', 'e']).to_csv(f'/Users/mac/OneDrive - University of Copenhagen/KU/Chr. Hansen/Python/ASCA/Results/DOE_{n_iter}.csv', index = False)