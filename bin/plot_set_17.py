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

# Specify resolution of numerical integrals
derivative_step = 0.008 # How much to vary parameter to calculate numerical derivative
mu_integral_step = 0.05 # For calculating numerical integral wrt mu between -1 and 1 

# Generate output paths  
#ps17_resultsdir = os.path.join(projectdir, 'results', 'ps17')
ps17_resultsdir = projectdir
#ps17_convergencedir = os.path.join(ps17_resultsdir, 'convergence')
cf.makedirectory(ps17_resultsdir)
#cf.makedirectory(ps17_convergencedir) 

# Linda Fiducial Cosmology 
ps17_fid = {
        "A_s" : 2.2321e-9, 
        "n_s" : 0.967,
        "omega_b" : 0.02226,
        "omega_cdm" : 0.1127,
        "tau_reio" : 0.0598,
        "h" : 0.701,
        "T_cmb" : 2.726, # Units [K]
        "N_ncdm" : 1., 
        "deg_ncdm" : 1,
        "T_ncdm" : (1.5072/2.726), # Units [T_cmb]. 
        "m_ncdm" : 0.0328, # Units [eV]
        "b0" : 1.0, 
        "alphak2" : 1.0,
        "sigma_fog_0" : 250000, #Units [m s^-2]
        "N_eff" : 3.046, #We allow relativistic neutrinos in addition to our DM relic
        "relic_vary" : "N_ncdm" # Fix T_ncdm or m_ncdm 
        }

# DESI Parameters
z_table = np.array([0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85])
dNdz = np.array([309., 2269., 1923., 2094., 1441., 1353., 1337., 523., 466., 329., 126., 0., 0.])
skycover = 14000. # Sky coverage of survey in degrees^2

# Run Fisher Forecast
masses = np.geomspace(0.01, 10., 10)
temps = np.linspace(0.79, 1.5, 12)

omegacdm_set = np.array([ps17_fid['omega_cdm'] - ps17_fid["N_ncdm"]*((masses/cf.NEUTRINO_SCALE_FACTOR)*np.power(tval / 1.95, 3.)) for tidx, tval in enumerate(temps)])                                  
ps17_fiducialset = [[dict(ps17_fid, **{'m_ncdm' : masses[midx], 'omega_cdm' : omegacdm_set[tidx, midx], 'T_ncdm' : temps[tidx]/2.726}) 
               for midx, mval in enumerate(masses)] for tidx, tval in enumerate(temps)] 
ps17_forecastset = [[cf.forecast(
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
    COV=True) for fididx, fidval in enumerate(fidrowvals)] for fidrowidx, fidrowvals in enumerate(ps17_fiducialset)]
for frowidx, frowval in enumerate(ps17_forecastset): 
    for fidx, fcst in enumerate(frowval): 
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
                'sigma_fog',                                   
                'b0',                                         
                'alpha_k2'],
            mu_step=mu_integral_step, 
            skipgen=False)
        print("Relic Forecast ", fidx, " complete...")
        dill.dump_session(os.path.join(ps17_resultsdir, str(frowidx)+'_'+str(fidx)+'_ps17.db'))



data = np.array([[np.sqrt(
            np.linalg.inv(
                np.delete(np.delete(fval.fisher, 4, axis=0), 4, axis=1))[5,5])
                               for fidx, fval in enumerate(frowval)] 
                               for frowidx, frowval in enumerate(ps17_forecastset)]) 

flatdata = data.flatten()
flatnsigmas = 1./flatdata
logflatdata = np.log10(flatdata)
logflatnsigmas = np.log10(flatnsigmas)


plt.figure(figsize=(15,7.5))
for tidx, tval in enumerate(temps): 
    plt.semilogx(masses, data[tidx, :], label=r"$\sigma_{g_\chi}$, DESI, " + str(tval), linestyle='solid')
plt.xlabel(r"$M_\chi$ [eV]", fontsize=24)
plt.ylabel(r"$\sigma_{g_\chi}$", fontsize=24)
plt.legend(fontsize=18, loc='lower left')
plt.tick_params(axis='x', which='minttor')
plt.grid(True, which='minor')
plt.savefig(ps17_resultsdir + "/plotA17a.png")

plt.figure(figsize=(15,15))    
# Create long format
people = np.repeat(np.flip(temps), 10)
feature = list(masses) * 12
df=pd.DataFrame({'Mass [eV]': feature, 'Temperature [K]': people, 'value': flatdata })
# plot it
df_wide=df.pivot_table( index='Temperature [K]', columns='Mass [eV]', values='value' )
p2=sns.heatmap(df_wide, 
               yticklabels=(np.flip(temps)),
               #xticklabels=(np.flip(masses)),
               cbar_kws={'label': r'$\sigma_{g_\chi}$'}, 
               #vmin=0.2, 
               #vmax=1.5,
               annot=True
              )
plt.savefig(ps17_resultsdir + "/plotA17b.png")
