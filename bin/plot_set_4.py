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
projectdir = cf.correct_path("/Users/nicholasdeporzio/Desktop/cfworkspace/")
datastore = cf.correct_path("/Users/nicholasdeporzio/Desktop/cfworkspace/data.nosync4/")
classpath = os.path.join(projectdir, "class")

# Specify resolution of numerical integrals
derivative_step = 0.008 # How much to vary parameter to calculate numerical derivative
mu_integral_step = 0.05 # For calculating numerical integral wrt mu between -1 and 1 


# Generate output paths  
ps4_resultsdir = os.path.join(projectdir, 'results', 'ps4')
ps4_convergencedir = os.path.join(ps4_resultsdir, 'convergence')
cf.makedirectory(ps4_resultsdir)
cf.makedirectory(ps4_convergencedir) 

# Linda Fiducial Cosmology 
ps4_fid = {
        "A_s" : 2.2321e-9, 
        "n_s" : 0.967,
        "omega_b" : 0.02226,
        "omega_cdm" : 0.1127,
        "tau_reio" : 0.0598,
        "h" : 0.701,
        "T_cmb" : 2.726, # Units [K]
        "N_ncdm" : 1., 
        "T_ncdm" : (1.5072/2.726), # Units [T_cmb]. 
        "m_ncdm" : 0.0328, # Units [eV]
        "b0" : 1.0, 
        "alphak2" : 1.0,
        "sigma_fog_0" : 250000, #Units [m s^-2]
        "N_eff" : 3.046, #We allow relativistic neutrinos in addition to our DM relic
        "relic_fix" : "m_ncdm" # Fix T_ncdm or m_ncdm 
        }

# DESI Parameters
z_table = z_table = np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85])
dNdz = np.array([309., 2269., 1923., 2094., 1441., 1353., 1337., 523., 466., 329., 126., 0., 0.])
skycover = 14000. # Sky coverage of survey in degrees^2

# Run Fisher Forecast
masses = np.append(np.array([0.001, 0.01]), np.geomspace(0.1, 22.0, 25))

omegacdm_set = ps4_fid['omega_cdm'] - ((masses/cf.NEUTRINO_SCALE_FACTOR)*np.power(ps4_fid['T_ncdm']*2.726 / 1.95, 3.))                                     
ps4_fiducialset = [dict(ps4_fid, **{'m_ncdm' : masses[midx], 'omega_cdm' : omegacdm_set[midx]}) 
               for midx, mval in enumerate(masses)]
ps4_forecastset = [cf.forecast(
    classpath, 
    datastore, 
    'relic', 
    fidval, 
    z_table, 
    "DESI",
    dNdz, 
    fcoverage_deg=skycover, 
    dstep=derivative_step,
    RSD=True,
    FOG=True,
    AP=True,
    COV=True) for fididx, fidval in enumerate(ps4_fiducialset)]
for fidx, fcst in enumerate(ps4_forecastset): 
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

dill.dump_session(os.path.join(ps4_resultsdir, 'ps4.db'))

# Save results 
inpath = "/Users/nicholasdeporzio/Documents/Academic/Research/Projects/cosmicfish/cosmicfish/priors/New_Relic_CMB_Fisher_Matrices/FisherCMBS4_bin"
#inpath = "/Users/nicholasdeporzio/Documents/Academic/Research/Projects/cosmicfish/cosmicfish/priors/New_Relic_CMB_Fisher_Matrices/FisherPlanck_bin"
head = 'omega_b \t omega_cdm \t n_s \t A_s \t tau_reio \t h \t T_ncdm \t sigma_fog \t beta0 \t beta1 \t alpha_k2'
for fidx, fval in enumerate(ps4_forecastset):
    fval.load_cmb_fisher(
        fisher_order=[
            'omega_b',                                    
            'omega_cdm',                                  
            'n_s',                                        
            'A_s',                                        
            'tau_reio',                                   
            'h',                                                                             
            'T_ncdm'],
        fisherpath=inpath+str(fidx+1)+"_T1.5K.txt", 
        colnames=[
            'omega_b',                                    
            'omega_cdm',                                  
            'n_s',                                        
            'A_s',                                        
            'tau_reio',                                   
            'h',                                                                             
            'T_ncdm[gamma]'])
    print("CMB Fisher information loaded to forecast " + str(fidx) + "...")
    outdir=os.path.join(ps4_resultsdir, 'M'+str(fidx+1))
    cf.makedirectory(outdir)
    fval.export_matrices(outdir)
    np.savetxt(outdir + '_Full_Fisher.dat', fval.numpy_full_fisher, delimiter='\t', header=head)
