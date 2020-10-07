import os
import shutil
import dill

plot_type = 'Fine' #'Grid' or 'Fine' 

# NOTE! You must run this python script using the same version of 
# python that was used to generate the .db Dill files!

########################################
## User modify above here
########################################

prepath='/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/'+plot_type+'Plot_'

if plot_type == 'Fine': 
    fcsts = 84
    mass_index = 21
    prefix = 'fp'

elif plot_type == 'Grid': 
    fcsts = 120
    mass_index = 10
    prefix = 'gp'

for idx in range(fcsts): 
    temp_index=(idx // mass_index)
    mass_index=(idx % mass_index)
    
    if (temp_index<100000): 
        dill.load_session(prepath+str(idx)+'/'+prefix+'_'+str(temp_index)+'_'+str(mass_index)+'.db')

        varnames = ['A_s', 'n_s', 'omega_b', 'omega_cdm', 'tau_reio', 'h', 'N_ncdm', 'M_chi', 'M_ncdm']
        for frowidx, frowval in enumerate(eval(prefix+'_forecastset')): 
            for fidx, fval  in enumerate(frowval): 
                mass = eval(prefix+"_fiducialset[frowidx][fidx]['m_ncdm']")
                temp = eval(prefix+"_fiducialset[frowidx][fidx]['T_ncdm']*2.726") 
        
                omega_b_high = fval.omega_b_high[0].datadirectory
                omega_b_low = fval.omega_b_low[0].datadirectory
                omega_cdm_high = fval.omega_cdm_high[0].datadirectory
                omega_cdm_low = fval.omega_cdm_low[0].datadirectory
                h_high = fval.h_high[0].datadirectory
                h_low = fval.h_low[0].datadirectory
                tau_reio_high = fval.tau_reio_high[0].datadirectory
                tau_reio_low = fval.tau_reio_low[0].datadirectory
                N_ncdm_high = fval.N_ncdm_high[0].datadirectory
                N_ncdm_low = fval.N_ncdm_low[0].datadirectory
                M_chi_high = fval.M_chi_high[0].datadirectory
                M_chi_low = fval.M_chi_low[0].datadirectory
                M_ncdm_high = fval.M_ncdm_high[0].datadirectory
                M_ncdm_low = fval.M_ncdm_low[0].datadirectory
                A_s_high = fval.A_s_high[0].datadirectory
                A_s_low = fval.A_s_low[0].datadirectory
                n_s_high = fval.n_s_high[0].datadirectory
                n_s_low = fval.n_s_low[0].datadirectory
                for pidx, pval in enumerate(varnames): 
                    path = ('/n/dvorkin_lab/ndeporzio/'+plot_type+'_Cls/T_ncdm_'+f'{temp:.2f}'+'/m_ncdm_' + f'{mass:.3f}' + '/' + pval + '/')
                    print('From: ' + eval((pval+'_high')))
                    print('To: ' +path+'high')
                    shutil.copytree(eval((pval+'_high')), (path+'high'))
                    shutil.copytree(eval((pval+'_low')), (path+'low'))
    else:  
        pass

