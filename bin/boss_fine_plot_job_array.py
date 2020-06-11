import os                                                                       
import shutil                                                                   
import numpy as np                                                              
import pandas as pd                                                             
import seaborn as sns                                                           
import cosmicfish as cf                                                         
import matplotlib.pyplot as plt                                                 
import dill    

# Instruct pyplot to use seaborn                                                
sns.set()                                                                       
                                                                                
# Set project, data, CLASS directories                                          
projectdir = os.environ['STORAGE_DIR']                                          
datastore = os.environ['DATASTORE_DIR']                                         
classpath = os.environ['CLASS_DIR']  
fidx = int(os.environ['FORECAST_INDEX'])


# Generate output paths                                                                
fp_resultsdir = projectdir                                                               
cf.makedirectory(fp_resultsdir) 


# Specify resolution of numerical integrals                                     
derivative_step = 0.008 # How much to vary parameter to calculate numerical derivative
g_derivative_step = 0.1
mu_integral_step = 0.05 # For calculating numerical integral wrt mu between -1 and 1 

# Linda Fiducial Cosmology                                                      
fp_fid = {                                                                    
        "A_s" : 2.2321e-9,                                                      
        "n_s" : 0.967,                                                          
        "omega_b" : 0.02226,                                                    
        "omega_cdm" : 0.1127,                                                   
        "tau_reio" : 0.0598,                                                    
        "h" : 0.701,                                                            
        "T_cmb" : 2.726, # Units [K]                                            
        "N_ncdm" : 4.,                                                          
        "deg_ncdm" : 1.0,                                                       
        "T_ncdm" : (0.79/2.726), # Units [T_cmb].                               
        "m_ncdm" : 0.01, # Units [eV]                                           
        "b0" : 1.0, 
        "beta0" : 1.7, 
        "beta1" : 1.0,                                                             
        "alphak2" : 1.0,                                                        
        "sigma_fog_0" : 250000, #Units [m s^-2]                                 
        "N_eff" : 0.0064, #We allow relativistic neutrinos in addition to our DM relic
        "relic_vary" : "N_ncdm",  # Fix T_ncdm or m_ncdm                        
        "m_nu" : 0.02                                                           
        }  

# BOSS values
z_table = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
dNdz = np.array([8., 50., 125., 222., 332., 447., 208., 30.])
skycover = 10000. # Sky coverage of BOSS in degrees^2

# Run Fisher Forecast                                                                                               
full_masses = np.geomspace(0.01, 10., 21)                                                    
full_temps = np.array([0.79, 0.91, 0.94, 1.08])    

mass_index=(fidx % 21)
temp_index=(fidx // 21)

masses = np.array([full_masses[mass_index]])
temps = np.array([full_temps[temp_index]])

omegacdm_set = np.array([                                                       
    fp_fid['omega_cdm']                                                       
    - ((masses/cf.NEUTRINO_SCALE_FACTOR)* np.power(tval / 1.95, 3.))                                            
    for tidx, tval in enumerate(temps)])  

fp_fiducialset = [[                                                           
    dict(fp_fid, **{                                                          
        'm_ncdm' : masses[midx],                                                
        'omega_cdm' : omegacdm_set[tidx, midx],                                 
        'T_ncdm' : temps[tidx]/2.726})                                          
    for midx, mval in enumerate(masses)]                                        
    for tidx, tval in enumerate(temps)] 

fp_forecastset = [[cf.forecast(                                               
    classpath,                                                                  
    datastore,                                                                  
    '2relic',                                                                   
    fidval,                                                                     
    z_table,                                                                    
    "EUCLID",                                                                     
    dNdz,                                                                       
    fcoverage_deg=skycover,                                                     
    dstep=derivative_step,
    gstep=g_derivative_step,                                                       
    RSD=True,                                                                   
    FOG=True,                                                                   
    AP=True,                                                                    
    COV=True)                                                                   
    for fididx, fidval in enumerate(fidrowvals)]                                
    for fidrowidx, fidrowvals in enumerate(fp_fiducialset)] 

#dill.load_session('')                                                          
for frowidx, frowval in enumerate(fp_forecastset):                            
    for fidx, fcst in enumerate(frowval):                                       
        if type(fcst.fisher)==type(None):                                       
            fcst.gen_pm()                                                       
            fcst.gen_fisher(                                                    
                fisher_order=[                                                  
                    'omega_b',                                                  
                    'omega_cdm',                                                
                    'n_s',                                                      
                    'A_s',                                                      
                    'tau_reio',                                                 
                    'h',                                                        
                    'N_ncdm',                                                   
                    'M_ncdm',                                                   
                    'sigma_fog',                                                
                    'beta0',
                    'beta1',                                                       
                    'alpha_k2'],                                                
                mu_step=mu_integral_step,                                       
                skipgen=False)                                                  
            print("Relic Forecast ", fidx, " complete...")                      
            dill.dump_session(os.path.join(fp_resultsdir, 'fp_'+str(temp_index)+'_'+str(mass_index)+'.db'))
        else:                                                                   
            print('Fisher matrix already generated!')  
