#!/usr/bin/env python

#Imports 
import cosmicfish as cf 
import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

#Workspace setup
projectdir = cf.correct_path("/n/home02/ndeporzio/projects/cosmicfish/")
datastore = os.path.join(projectdir, "datastore") 
classpath = os.path.join(projectdir, "class")

#Analysis setup
derivative_step = 0.005 
mu_integral_step = 0.01 

#Experiment setup
z_table = np.arange(0.65, 1.85, 0.1)
dNdz = np.array([309., 2269., 1923., 2094., 1441., 1353., 1337., 523., 466., 329., 126., 0., 0.])
skycover = 14000. 

#Fiducial cosmology  
fiducial = {
        "A_s" : 2.2321e-9, 
        "n_s" : 0.96659,
        "omega_b" : 0.02226,
        "omega_cdm" : 0.11271,
        "tau_reio" : 0.059888,
        "h" : 0.70148,
        "T_cmb" : 2.726, # Units [K]
        "N_ncdm" : 1., 
        "T_ncdm" : 1.5/2.726, # Units [T_cmb]  Are Julian's matrices at 1.64 K? 
        "m_ncdm" : 0.1, # Units [eV]
        "b0" : 1.0, 
        "alphak2" : 1.0,
        "sigma_fog_0" : 250000, #Units [m s^-2]
        "N_eff" : 3.046, 
        "relic_fix" : "m_ncdm" # Fix T_ncdm or m_ncdm 
        }

#Fisher parameter ordering
fishorder = [
    'omega_b',                                    
    'omega_cdm',                                  
    'n_s',                                        
    'A_s',                                        
    'tau_reio',                                   
    'h',                                          
    #'M_ncdm',                                    
    'T_ncdm',                                 
    'sigma_fog',                                   
    'b0',                                         
    'alpha_k2']

convergencetest = cf.convergence(
    classpath, # Path to CLASS installation
    datastore, # Path to directory holding CLASS output data
    'relic', # 'relic' or 'neutrino' forecasting scheme 
    fiducial, # The fiducial cosmology 
    z_table, # Redshift steps in observation
    dNdz, # Redshift noise in observation
    fisher_order=fishorder
    fcoverage_deg=14000, # Sky coverage in observation
    RSD=True, # Use RSD correction to Pm
    FOG=True, # Use FOG correction to Pm
    AP=True, # Use AP correction to PM
    COV=True, #Use AP Change of Variables correction to PM
    mu_step=mu_integral_step,
    varyfactors=[0.003, 0.005, 0.007], # Relative factors used to compute convergence
    saveplots=True,
    showplots=False,
    savepath="/n/home02/ndeporzio/projects/cosmicfish/results",
    plotparams=[
        'omega_b',                                    
        'omega_cdm',                                  
        'n_s',                                        
        'A_s',                                        
        'tau_reio',                                   
        'h',                                          
        'omega_ncdm',                                    
        'T_ncdm',                                 
        'sigmafog',                                   
        'b0',                                         
        'alpha_k2']
    )
convergencetest.gen_all_plots()

masses = np.geomspace(0.1, 10.0, 21) 
fiducialset = [dict(fiducial, **{'m_ncdm': masses[midx]}) for midx, mval in enumerate(masses)]

forecastset = [cf.forecast(
    classpath, 
    datastore, 
    'relic', 
    fidval, 
    z_table, 
    dNdz, 
    fcoverage_deg=skycover, 
    dstep=derivative_step,
    RSD=True,
    FOG=True,
    AP=True,
    COV=True) for fididx, fidval in enumerate(fiducialset)]

for fidx, fcst in enumerate(forecastset): 
    fcst.gen_pm()
    fcst.gen_fisher(
        fisher_order=[
            'omega_b',                                    
            'omega_cdm',                                  
            'n_s',                                        
            'A_s',                                        
            'tau_reio',                                   
            'h',                                          
            'T_ncdm',                            
            'sigma_fog',                                   
            'b0',                                         
            'alpha_k2'],
        mu_step=mu_integral_step, 
        skipgen=False)
    
    print("Relic Forecast ", fidx, " complete...")

inpath = os.path.join(cf.priors_directory(), "CMBS4_Fisher_Relic_")
for fidx, fval in enumerate(forecastset):
    outpath = os.path.join(projectdir, "results", str(fidx+1))
    fval.load_cmb_fisher(
        fisher_order=[
            'omega_b',                                    
            'omega_cdm',                                  
            'n_s',                                        
            'A_s',                                        
            'tau_reio',                                   
            'h',                                                                             
            'T_ncdm'],
            fisherpath=inpath+str(fidx+1)+".dat")
    print("CMB Fisher information loaded to forecast " + str(fidx) + "...")
    fval.export_matrices(outpath)

errors_Tncdm_cmb = [np.sqrt(fval.numpy_cmb_covariance[6,6]) for fidx, fval in enumerate(forecastset)]    
errors_Tncdm_lss = [np.sqrt(fval.numpy_lss_covariance[6,6]) for fidx, fval in enumerate(forecastset)]
errors_Tncdm_full = [np.sqrt(fval.numpy_full_covariance[6,6]) for fidx, fval in enumerate(forecastset)]

plt.figure(figsize=(15,7.5))
plt.semilogx(masses, errors_Tncdm_full, label="CMB+LSS")
plt.semilogx(masses, errors_Tncdm_lss, label="LSS")
plt.semilogx(masses, errors_Tncdm_cmb, label="CMB")
plt.title("Relic Temperature Uncertainty")
plt.xlabel("Relic Mass [eV]")
plt.ylabel(r"$\sigma_T ~[K]$")
plt.legend()
plt.savefig(os.path.join(projectdir, "results", "results1.png"))

plt.figure(figsize=(15,7.5))                                                    
plt.semilogx(masses, errors_Tncdm_full, label="CMB+LSS")                        
plt.semilogx(masses, errors_Tncdm_cmb, label="CMB")                             
plt.title("Relic Temperature Uncertainty")                                      
plt.xlabel("Relic Mass [eV]")                                                   
plt.ylabel(r"$\sigma_T ~[K]$")                                                  
plt.legend()                                                                    
plt.savefig(os.path.join(projectdir, "results", "results2.png"))
