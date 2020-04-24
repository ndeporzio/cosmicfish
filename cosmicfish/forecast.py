import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import quad 

import cosmicfish as cf 

class forecast: 
    '''Forecasts Fisher/Covariance matrices from fid cosmo.'''
    
    def __init__(self, 
                 classdir, 
                 datastore,
                 forecast_type,  
                 fiducialcosmo, 
                 z_steps,
                 lss_survey_name,  
                 dNdz,
                 fsky=None, 
                 fcoverage_deg=None,
                 dstep=0.03,
                 RSD=True,  
                 FOG=True, 
                 AP=True, 
                 COV=True): 
        self.classdir = classdir                                                
        self.datastore = datastore
        self.forecast = forecast_type
        self.fid = fiducialcosmo
        self.z_steps = z_steps
        self.lss_survey_name = lss_survey_name
        self.dNdz = dNdz
        self.dstep = dstep
        self.use_rsd = RSD
        self.use_fog = FOG     
        self.use_ap = AP
        self.use_cov = COV

        self.A_s_fid = self.fid['A_s']
        self.n_s_fid = self.fid['n_s']
        self.omega_b_fid = self.fid['omega_b']
        self.omega_cdm_fid = self.fid['omega_cdm']
        self.h_fid = self.fid['h']
        self.tau_reio_fid = self.fid['tau_reio']
        self.m_ncdm_fid = self.fid['m_ncdm'] # Unit [eV]   
        self.N_eff_fid = self.fid['N_eff'] 
        self.relic_vary = self.fid['relic_vary']   
        self.T_cmb_fid = self.fid['T_cmb']
    
        self.fisher=None

        if self.lss_survey_name=='DESI': 
            self.lss_survey_params = ['alphak2', 'b0', 'delta_L'] 
        elif self.lss_survey_name=='EUCLID':
            self.lss_survey_params = ['alphak2', 'beta0', 'beta1', 'delta_L'] 
        elif self.lss_survey_name=='BOSS': 
            self.lss_survey_params = []

        if 'b0' in self.fid:
            self.b0_fid = self.fid['b0']

        if self.use_rsd==True:
            for pidx, pval in enumerate(self.lss_survey_params): 
                if pval not in self.fid: 
                    if pval=='delta_L':
                        self.delta_L_fid = cf.RSD_DELTA_L_NUMERATOR_FACTOR
                        self.fid['delta_L'] = self.delta_L_fid
                    else: 
                        print("ERROR: missing parameter "
                            + pval
                            + " from fiducial"
                            + " cosmology!") 
 
        if self.use_fog==True:
            if 'sigma_fog_0' in self.fid:                                       
                self.sigma_fog_0_fid = self.fid['sigma_fog_0']                  
            else:                                                               
                print("Error: Must specify fiducial sigma_fog_0" +              
                    " to compute RSD.") 

        self.n_densities = np.zeros(len(self.dNdz))

        if self.forecast=="relic":
            self.N_ncdm_fid = self.fid['N_ncdm']
            self.M_ncdm_fid = self.m_ncdm_fid 
            self.T_ncdm_fid = self.fid['T_ncdm'] * self.T_cmb_fid
            self.omega_ncdm_fid = cf.omega_ncdm(self.T_ncdm_fid, 
                                                self.m_ncdm_fid,
                                                self.N_ncdm_fid,        
                                                "relic")
        elif self.forecast=="neutrino":
            self.T_ncdm_fid = cf.RELIC_TEMP_SCALE
            self.M_ncdm_fid = 3. * self.m_ncdm_fid # Unit [eV] 
            self.omega_ncdm_fid = cf.omega_ncdm(None, 
                                                self.m_ncdm_fid,
                                                3., 
                                                "neutrino")
        self.kp = cf.k_pivot() # Units [Mpc^-1]

        self.fsky, self.fcoverage_deg = cf.set_sky_cover(fsky, fcoverage_deg)
            
        self.psterms = [] 

        # Generate tables
        self.k_table = [0] * len(self.z_steps)
        for zidx, zval in enumerate(self.z_steps):
            V = self.V(zidx) #Redundant to cf.gen_V 
            self.k_table[zidx] = cf.gen_k_table(
                volume=V, 
                z=self.z_steps[zidx],
                h=self.h_fid, 
                n_s=self.n_s_fid,
                k_steps=cf.DEFAULT_K_TABLE_STEPS)           

    def gen_pm(self):  
        if 'pm' not in self.psterms: 
            self.psterms.append('pm') 

        # Generate spectra at each z for fid cosmo
        self.spectra_mid = [cf.spectrum(
                                cf.generate_data(
                                    dict(self.fid,      
                                         **{'z_pk' : zval}),
                                    self.classdir,       
                                    self.datastore)[0:-20],
                                fsky=self.fsky,
                                k_table=self.k_table[zidx],
                                forecast=self.forecast) 
                            for zidx, zval in enumerate(self.z_steps)]

        self.Pm = [self.spectra_mid[zidx].ps_table 
                   for zidx, zval in enumerate(self.z_steps)]

        # Generate variation spectra at each z   
 
        self.omega_b_high, self.omega_b_low = self.generate_spectra('omega_b')
        self.omega_cdm_high, self.omega_cdm_low = self.generate_spectra(
                                                      'omega_cdm')
        self.h_high, self.h_low = self.generate_spectra('h')
        self.tau_reio_high, self.tau_reio_low = self.generate_spectra(
                                                    'tau_reio')
        if self.forecast=="neutrino":
            # We give 'm_ncdm' as parameter name to 'generate_spectra', but 
            # 'generate_data' makes 3 copies of this for neutrinos. So really,
            # we are varying M_ncdm
            self.M_ncdm_high, self.M_ncdm_low = self.generate_spectra('m_ncdm')
        elif self.forecast=="relic": 
            if self.relic_vary=="m_ncdm": 
                self.M_ncdm_high, self.M_ncdm_low = self.generate_spectra(
                    'm_ncdm')
            if self.relic_vary=="T_ncdm":                                        
                self.T_ncdm_high, self.T_ncdm_low = self.generate_spectra(      
                    'T_ncdm', 
                    manual_high = ((1. + self.dstep)
                                    *(self.T_ncdm_fid / self.T_cmb_fid)), 
                    manual_low = ((1. - self.dstep)                              
                                    *(self.T_ncdm_fid / self.T_cmb_fid)), 
                    )
            if self.relic_vary=="N_ncdm": 
                self.N_ncdm_high, self.N_ncdm_low = self.generate_spectra(      
                    'deg_ncdm') 

        # Calculate centered derivatives about fiducial cosmo at each z 
        if cf.ANALYTIC_A_S==False: 
            self.A_s_high, self.A_s_low = self.generate_spectra('A_s') 
            self.dPdA_s, self.dlogPdA_s = cf.dPs_array(self.A_s_low, 
                self.A_s_high, self.fid['A_s']*self.dstep) 
        else: 
            self.dlogPdA_s = [[cf.dlogPdAs(self.A_s_fid)  
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)] # Analytic form

        if cf.ANALYTIC_N_S==False: 
            self.n_s_high, self.n_s_low = self.generate_spectra('n_s')
            self.dPdn_s, self.dlogPdn_s = cf.dPs_array(self.n_s_low, 
                self.n_s_high, self.fid['n_s']*self.dstep) 
        else:  
            self.dlogPdn_s = [[cf.dlogPdns(kval, self.kp) 
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)] # Analytic form

        self.dPdomega_b, self.dlogPdomega_b = cf.dPs_array(
            self.omega_b_low, 
            self.omega_b_high, 
            self.omega_b_fid * self.dstep) 
        self.dPdomega_cdm, self.dlogPdomega_cdm = cf.dPs_array(
            self.omega_cdm_low, 
            self.omega_cdm_high, 
            self.omega_cdm_fid * self.dstep)
        self.dPdh, self.dlogPdh = cf.dPs_array(
            self.h_low, 
            self.h_high, 
            self.h_fid * self.dstep)
        self.dPdtau_reio, self.dlogPdtau_reio = cf.dPs_array(
            self.tau_reio_low, 
            self.tau_reio_high, 
            self.tau_reio_fid * self.dstep)

        if self.forecast=="neutrino": 
            self.dPdM_ncdm, self.dlogPdM_ncdm = cf.dPs_array(
                self.M_ncdm_low, 
                self.M_ncdm_high, 
                3. * self.m_ncdm_fid * self.dstep)
            self.dPdomega_ncdm = [
                (np.array(self.dPdM_ncdm[zidx]) 
                    * cf.dM_ncdm_domega_ncdm(self.T_ncdm_fid))
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogPdomega_ncdm = [
                (np.array(self.dlogPdM_ncdm[zidx]) 
                    * cf.dM_ncdm_domega_ncdm(self.T_ncdm_fid))
                for zidx, zval in enumerate(self.z_steps)]
        elif self.forecast=="relic":
            if self.relic_vary=="m_ncdm": 
                self.dPdM_ncdm, self.dlogPdM_ncdm = cf.dPs_array(
                    self.M_ncdm_low, 
                    self.M_ncdm_high, 
                    self.M_ncdm_fid*self.dstep)
                self.dPdomega_ncdm = [
                    (np.array(self.dPdM_ncdm[zidx]) 
                        * cf.dM_ncdm_domega_ncdm(self.T_ncdm_fid))
                    for zidx, zval in enumerate(self.z_steps)]
                self.dlogPdomega_ncdm = [
                    (np.array(self.dlogPdM_ncdm[zidx]) 
                        * cf.dM_ncdm_domega_ncdm(self.T_ncdm_fid))
                    for zidx, zval in enumerate(self.z_steps)]
                #print("T_ncdm fixed. dlogPmdMncdm: ", self.dlogPdM_ncdm)
                #print("T_ncdm fixed. dMncdmdomegancdm: ", cf.dM_ncdm_domega_ncdm(self.T_ncdm_fid))
                #print("T_ncdm fixed. dlogPmdomegancdm: ", self.dlogPdomega_ncdm)  
            elif self.relic_vary=="T_ncdm": 
                self.dPdT_ncdm, self.dlogPdT_ncdm = cf.dPs_array(               
                    self.T_ncdm_low,                                            
                    self.T_ncdm_high,                                           
                    self.T_ncdm_fid*self.dstep)                                 
                self.dPdomega_ncdm = [                                          
                    (np.array(self.dPdT_ncdm[zidx])                              
                        * cf.dT_ncdm_domega_ncdm(
                            self.T_ncdm_fid, 
                            self.M_ncdm_fid))                   
                    for zidx, zval in enumerate(self.z_steps)]                  
                self.dlogPdomega_ncdm = [                                       
                    (np.array(self.dlogPdT_ncdm[zidx])                           
                        * cf.dT_ncdm_domega_ncdm(
                            self.T_ncdm_fid, 
                            self.M_ncdm_fid))                   
                    for zidx, zval in enumerate(self.z_steps)]
            elif self.relic_vary=="N_ncdm": 
                self.dPdN_ncdm, self.dlogPdN_ncdm = cf.dPs_array(
                    self.N_ncdm_low, 
                    self.N_ncdm_high,
                    self.N_ncdm_fid*self.dstep)
                self.dPdomega_ncdm = [
                    np.array(self.dPdN_ncdm[zidx])
                    * np.array(1./self.omega_ncdm_fid)
                    for zidx, zval in enumerate(self.z_steps)]
                self.dlogPdomega_ncdm = [
                    np.array(self.dlogPdN_ncdm[zidx])                              
                    * np.array(1./self.omega_ncdm_fid)                          
                    for zidx, zval in enumerate(self.z_steps)]  
 
    def gen_rsd(self, mu): 
        '''Given mu, creates len(z_steps) array. Each elem is len(k_table).'''

        if self.use_rsd==True:
            if 'rsd' not in self.psterms:                                           
                self.psterms.append('rsd')                                          
                                                                                    
            if self.forecast=="neutrino":                                           
                relic = False                                                       
            elif self.forecast=="relic":                                            
                relic = True 

            fiducial = {'omega_b' : self.omega_b_fid, 
                        'omega_cdm' : self.omega_cdm_fid, 
                        'omega_ncdm' : self.omega_ncdm_fid,
                        'h' : self.h_fid, 
                        'mu' : mu,
                        'relic' : relic,
                        'T_ncdm' : self.T_ncdm_fid,
                        'lss_survey_name' : self.lss_survey_name,
                        'delta_L' : self.delta_L_fid,
                        'b0' : self.b0_fid}
            for pidx, pval in enumerate(self.lss_survey_params):
                fiducial[pval]=self.fid[pval] 
    
            self.RSD = [[cf.rsd(**dict(fiducial, **{'z' : zval, 'k' : kval,
                'D' : self.spectra_mid[zidx].D})) 
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]   

            self.RSD_norelicstep = [[
                cf.rsd(**dict(fiducial, **{'z' : zval, 'k' : kval,     
                'D' : self.spectra_mid[zidx].D, "step" : False}))                               
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]

            # FOR DEBUGGING. DELETE LATER.  
            self.RSD_specialb = [[                                           
                cf.rsd(**dict(fiducial, **{'z' : zval, 'k' : kval,              
                'D' : self.spectra_mid[zidx].D, "b0" : 1.001}))               
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.RSD_specialb_high = [[                                              
                cf.rsd(**dict(fiducial, **{'z' : zval, 'k' : kval,              
                'D' : self.spectra_mid[zidx].D, "b0" : 1.0005}))                 
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.RSD_specialb_low = [[                                              
                cf.rsd(**dict(fiducial, **{'z' : zval, 'k' : kval,              
                'D' : self.spectra_mid[zidx].D, "b0" : 0.9995}))                 
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)] 
    
            self.dlogRSDdomega_b = [[cf.derivative(
                cf.log_rsd, 
                'omega_b', 
                self.dstep,
                **dict(fiducial, **{'z' : zval, 'k' :  kval,
                    'D' : self.spectra_mid[zidx].D}))
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            self.dlogRSDdomega_cdm = [[cf.derivative(
                cf.log_rsd, 
                'omega_cdm', 
                self.dstep,  
                **dict(fiducial, **{'z' : zval, 'k' :  kval,
                    'D' : self.spectra_mid[zidx].D})) 
                for kidx, kval in enumerate(self.k_table[zidx])]                                           
                for zidx, zval in enumerate(self.z_steps)]
    
            self.dlogRSDdomega_ncdm = [[cf.derivative(                                  
               cf.log_rsd,                                                         
                'omega_ncdm',                                                        
                self.dstep,                                                         
                **dict(fiducial, **{'z' : zval, 'k' :  kval,
                    'D' : self.spectra_mid[zidx].D})) 
                for kidx, kval in enumerate(self.k_table[zidx])]                                           
                for zidx, zval in enumerate(self.z_steps)]
    
            if self.relic_vary=="T_ncdm": 
                self.dlogRSDdT_ncdm = (np.array(self.dlogRSDdomega_ncdm)            
                    * cf.domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.M_ncdm_fid))    
            elif self.relic_vary=="m_ncdm": 
                self.dlogRSDdM_ncdm = (np.array(self.dlogRSDdomega_ncdm) 
                    * cf.domega_ncdm_dM_ncdm(self.T_ncdm_fid))
            elif self.relic_vary=="N_ncdm": 
                self.dlogRSDdN_ncdm = (np.array(self.dlogRSDdomega_ncdm)
                    * cf.omega_ncdm(
                        self.T_ncdm_fid, 
                        self.m_ncdm_fid,        
                        self.N_ncdm_fid, 
                        'relic'))
    
            self.dlogRSDdh = [[cf.derivative(                                 
                cf.log_rsd,                                                         
                'h',                                                       
                self.dstep,                                                         
                **dict(fiducial, **{'z' : zval, 'k' :  kval,
                    'D' : self.spectra_mid[zidx].D})) 
                for kidx, kval in enumerate(self.k_table[zidx])]                                           
                for zidx, zval in enumerate(self.z_steps)]   
   
            if 'b0' in self.lss_survey_params:  
                self.dlogRSDdb0 = [[cf.derivative(                                       
                    cf.log_rsd,                                                         
                    'b0',                                                                
                    self.dstep,                                                         
                    **dict(fiducial, **{'z' : zval, 'k' :  kval,
                        'D' : self.spectra_mid[zidx].D}))                                                  
                    for kidx, kval in enumerate(self.k_table[zidx])]                                           
                    for zidx, zval in enumerate(self.z_steps)]
            else: 
                self.dlogRSDdb0 = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]                                       
                    for zidx, zval in enumerate(self.z_steps)]     

            if 'alphak2' in self.lss_survey_params:     
                self.dlogRSDdalphak2 = [[cf.derivative(                                       
                    cf.log_rsd,                                                         
                    'alphak2',                                                                
                    self.dstep,                                                         
                    **dict(fiducial, **{'z' : zval, 'k' :  kval,
                        'D' : self.spectra_mid[zidx].D}))                                                  
                    for kidx, kval in enumerate(self.k_table[zidx])]                                           
                    for zidx, zval in enumerate(self.z_steps)]
            else:                                                               
                self.dlogRSDdalphak2 = [[0.                                          
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)] 

            if 'beta0' in self.lss_survey_params:                             
                self.dlogRSDdbeta0 = [[cf.derivative(                             
                    cf.log_rsd,                                                           
                    'beta0',                                                      
                    self.dstep,                                                 
                    **dict(fiducial, **{'z' : zval, 'k' :  kval,                                       
                        'D' : self.spectra_mid[zidx].D}))                                                  
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                  
            else:                                                               
                self.dlogRSDdbeta0 = [[0.                                     
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]

            if 'beta1' in self.lss_survey_params:                               
                self.dlogRSDdbeta1 = [[cf.derivative(                           
                    cf.log_rsd,                                                 
                    'beta1',                                                    
                    self.dstep,                                                 
                    **dict(fiducial, **{'z' : zval, 'k' :  kval,                
                        'D' : self.spectra_mid[zidx].D}))                       
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                  
            else:                                                               
                self.dlogRSDdbeta1 = [[0.                                       
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)] 

            self.dlogRSDddeltaL = [[cf.derivative(
                cf.log_rsd,
                'delta_L',
                self.dstep,
                **dict(fiducial, **{'z' : zval, 'k' :  kval,                    
                    'D' : self.spectra_mid[zidx].D, 
                    'delta_L' : self.delta_L_fid}))                           
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]  
        else:
            self.RSD = [[1. 
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]   
            self.RSD_norelicstep = [[1.                                                     
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]  

            self.RSD_specialb = [[1.    #FOR DEBUGGING. DELETE LATER.                                      
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.RSD_specialb_high = [[1.    #FOR DEBUGGING. DELETE LATER.                                      
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)] 
            self.RSD_specialb_low = [[1.    #FOR DEBUGGING. DELETE LATER.                                      
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)] 

            self.dlogRSDdomega_b = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogRSDdomega_cdm = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogRSDdomega_ncdm = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            if self.relic_vary=="T_ncdm": 
                self.dlogRSDdT_ncdm = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]
            elif self.relic_vary=="m_ncdm":
                self.dlogRSDdM_ncdm = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]
            elif self.relic_vary=="N_ncdm":                                     
                self.dlogRSDdN_ncdm = [[0.                                      
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]
            self.dlogRSDdh = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogRSDdb0 = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogRSDdalphak2 = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogRSDdbeta0 = [[0.                                         
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)] 
            self.dlogRSDdbeta1 = [[0.                                         
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)] 
            self.dlogRSDddeltaL = [[0.                                         
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]

    def gen_fog(self, mu):

        if self.use_fog==True:
            if 'fog' not in self.psterms:                                           
                self.psterms.append('fog')

            fiducial = {'omega_b' : self.omega_b_fid,                               
                        'omega_cdm' : self.omega_cdm_fid,                           
                        'omega_ncdm' : self.omega_ncdm_fid,                         
                        'h' : self.h_fid,                                           
                        'mu' : mu,
                        'sigma_fog_0' : self.sigma_fog_0_fid}
                        
            self.FOG = [[cf.fog(**dict(fiducial, **{'z' : zval, 'k' :  kval})) 
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            self.dlogFOGdomega_b = [[cf.derivative(cf.log_fog, 'omega_b', 
                self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))                       
                for kidx, kval in enumerate(self.k_table[zidx])]                                           
                for zidx, zval in enumerate(self.z_steps)]
    
            self.dlogFOGdomega_cdm = [[cf.derivative(cf.log_fog, 'omega_cdm',           
                self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))           
                for kidx, kval in enumerate(self.k_table[zidx])]                                           
                for zidx, zval in enumerate(self.z_steps)]
    
            self.dlogFOGdomega_ncdm = [[cf.derivative(cf.log_fog, 'omega_ncdm',           
                self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))           
                for kidx, kval in enumerate(self.k_table[zidx])]                                           
                for zidx, zval in enumerate(self.z_steps)]
    
            if self.relic_vary=="T_ncdm":                                            
                self.dlogFOGdT_ncdm = (np.array(self.dlogFOGdomega_ncdm)            
                    * cf.domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.M_ncdm_fid))     
            elif self.relic_vary=="m_ncdm":                                                                   
                self.dlogFOGdM_ncdm = (np.array(self.dlogFOGdomega_ncdm)            
                    * cf.domega_ncdm_dM_ncdm(self.T_ncdm_fid))
            elif self.relic_vary=="N_ncdm":
                self.dlogFOGdN_ncdm = (np.array(self.dlogFOGdomega_ncdm)
                    * cf.omega_ncdm(
                        self.T_ncdm_fid,
                        self.m_ncdm_fid,
                        self.N_ncdm_fid,
                        'relic'))   
    
            self.dlogFOGdh = [[cf.derivative(cf.log_fog, 'h', self.dstep,           
                **dict(fiducial, **{'z' : zval, 'k' :  kval}))                      
                for kidx, kval in enumerate(self.k_table[zidx])]                                           
                for zidx, zval in enumerate(self.z_steps)]
    
            self.dlogFOGdsigmafog0 = [[cf.derivative(cf.log_fog, 'sigma_fog_0',     
                self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))           
                for kidx, kval in enumerate(self.k_table[zidx])]                                           
                for zidx, zval in enumerate(self.z_steps)]
        else: 
            self.FOG = [[1.  
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]                      
                                                                                
            self.dlogFOGdomega_b = [[0.      
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]                      
                                                                                
            self.dlogFOGdomega_cdm = [[0. 
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]                      
                                                                                
            self.dlogFOGdomega_ncdm = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]                      
                                                                                
            if self.relic_vary=="T_ncdm":                                        
                self.dlogFOGdT_ncdm = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]
            elif self.relic_vary=="m_ncdm":                                                               
                self.dlogFOGdM_ncdm = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]      
            elif self.relic_vary=="N_ncdm":                                                               
                self.dlogFOGdN_ncdm = [[0.                                      
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)] 
                                                                                
            self.dlogFOGdh = [[0. 
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]                      
                                                                                
            self.dlogFOGdsigmafog0 = [[0. 
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)] 

    def gen_ap(self):

        if self.use_ap==True:
            if 'ap' not in self.psterms:                                           
                self.psterms.append('ap')
    
            fiducial = {'omega_b' : self.omega_b_fid,                               
                        'omega_cdm' : self.omega_cdm_fid,                           
                        'omega_ncdm' : self.omega_ncdm_fid,                         
                        'h' : self.h_fid,
                        'omega_b_fid' :  self.omega_b_fid,
                        'omega_cdm_fid' : self.omega_cdm_fid,
                        'omega_ncdm_fid' : self.omega_ncdm_fid,
                        'h_fid' : self.h_fid}                                           
    
            self.AP = [[cf.ap(**dict(fiducial, **{'z' : zval, 'z_fid' : zval}))
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            self.dlogAPdomega_b = [[cf.derivative(cf.log_ap, 'omega_b', self.dstep,             
                **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))                    
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)] 
    
            self.dlogAPdomega_cdm = [[cf.derivative(cf.log_ap, 'omega_cdm', 
                self.dstep, **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))                    
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            self.dlogAPdomega_ncdm = [[cf.derivative(cf.log_ap, 'omega_ncdm', 
                self.dstep, **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))                    
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            if self.relic_vary=="T_ncdm":                                            
                self.dlogAPdT_ncdm = (np.array(self.dlogAPdomega_ncdm)            
                    * cf.domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.M_ncdm_fid))     
            elif self.relic_vary=="m_ncdm":                                                                   
                self.dlogAPdM_ncdm = (np.array(self.dlogAPdomega_ncdm)            
                    * cf.domega_ncdm_dM_ncdm(self.T_ncdm_fid))   
            elif self.relic_vary=="N_ncdm":                                                                  
                self.dlogAPdN_ncdm = (np.array(self.dlogAPdomega_ncdm)            
                    * cf.omega_ncdm(
                        self.T_ncdm_fid,
                        self.m_ncdm_fid,
                        self.N_ncdm_fid,
                        'relic'))  

            self.dlogAPdh = [[cf.derivative(cf.log_ap, 'h', self.dstep,             
                **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))                   
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
        else: 
            self.AP = [[1.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogAPdomega_b = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogAPdomega_cdm = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogAPdomega_ncdm = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            if self.relic_vary=="T_ncdm":
                self.dlogAPdT_ncdm = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]
            elif self.reliv_vary=="m_ncdm":
                self.dlogAPdM_ncdm = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]
            elif self.reliv_vary=="N_ncdm":                                     
                self.dlogAPdN_ncdm = [[0.                                       
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]  
            self.dlogAPdh = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]

                            
    def gen_cov(self, mu):

        # CAUTION! You've written physics here. By your convention, all physics
        # should be in the equations.py module.  
        if self.use_cov==True: 
            if 'cov' not in self.psterms: 
                self.psterms.append('cov') 
    
            self.COV = [[cf.cov() 
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            fiducial = {
                'omega_b' : self.omega_b_fid, 
                'omega_cdm' : self.omega_cdm_fid, 
                'omega_ncdm' : self.omega_ncdm_fid, 
                'h' : self.h_fid}
    
            dHdomegab = [[cf.derivative(cf.H, 'omega_b', self.dstep,             
                **dict(fiducial, **{'z' : zval}))                    
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)] 
     
            dHdomegacdm = [[cf.derivative(cf.H, 'omega_cdm', self.dstep,                
                **dict(fiducial, **{'z' : zval}))                                    
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            dHdomegancdm = [[cf.derivative(cf.H, 'omega_ncdm', self.dstep,                
                **dict(fiducial, **{'z' : zval}))                                    
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            dHdh = [[cf.derivative(cf.H, 'h', self.dstep,                
                **dict(fiducial, **{'z' : zval}))                                    
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)]
    
            dDadomegab = [[cf.derivative(cf.Da, 'omega_b', self.dstep,                
                **dict(fiducial, **{'z' : zval}))                   
                for kidx, kval in enumerate(self.k_table[zidx])]                    
                for zidx, zval in enumerate(self.z_steps)]                 
                                                                                    
            dDadomegacdm = [[cf.derivative(cf.Da, 'omega_cdm', self.dstep,            
                **dict(fiducial, **{'z' : zval}))         
                for kidx, kval in enumerate(self.k_table[zidx])]                    
                for zidx, zval in enumerate(self.z_steps)]                           
                                                                                    
            dDadomegancdm = [[cf.derivative(cf.Da, 'omega_ncdm', self.dstep,          
                **dict(fiducial, **{'z' : zval}))                   
                for kidx, kval in enumerate(self.k_table[zidx])]                    
                for zidx, zval in enumerate(self.z_steps)]                
                                                                                    
            dDadh = [[cf.derivative(cf.Da, 'h', self.dstep,                           
                **dict(fiducial, **{'z' : zval}))                                    
                for kidx, kval in enumerate(self.k_table[zidx])]                    
                for zidx, zval in enumerate(self.z_steps)]
    
            fiducial = {
                'omega_b' : self.omega_b_fid,                                       
                'omega_cdm' : self.omega_cdm_fid,                                   
                'omega_ncdm' : self.omega_ncdm_fid,                                 
                'h' : self.h_fid, 
                'mu' : mu}
    
            dkdH = [[cf.cov_dkdH(**dict(fiducial, **{'z' : zval, 'k' : kval}))                                    
                for kidx, kval in enumerate(self.k_table[zidx])]                    
                for zidx, zval in enumerate(self.z_steps)]
    
            dkdDa = [[cf.cov_dkdDa(**dict(fiducial, **{'z' : zval, 'k' : kval}))                        
                for kidx, kval in enumerate(self.k_table[zidx])]                    
                for zidx, zval in enumerate(self.z_steps)]
    
            dkdomegab = [[(
                dkdH[zidx][kidx] * dHdomegab[zidx][kidx]
                + dkdDa[zidx][kidx] * dDadomegab[zidx][kidx])
                for kidx, kval in enumerate(self.k_table[zidx])]                    
                for zidx, zval in enumerate(self.z_steps)]
    
            dkdomegacdm = [[(                                                         
                dkdH[zidx][kidx] * dHdomegacdm[zidx][kidx]                            
                + dkdDa[zidx][kidx] * dDadomegacdm[zidx][kidx])                       
                for kidx, kval in enumerate(self.k_table[zidx])]                          
                for zidx, zval in enumerate(self.z_steps)] 
    
            dkdomegancdm = [[(                                                         
                dkdH[zidx][kidx] * dHdomegancdm[zidx][kidx]                            
                + dkdDa[zidx][kidx] * dDadomegancdm[zidx][kidx])                       
                for kidx, kval in enumerate(self.k_table[zidx])]                          
                for zidx, zval in enumerate(self.z_steps)]
    
            dkdh = [[(                                                         
                dkdH[zidx][kidx] * dHdh[zidx][kidx]                            
                + dkdDa[zidx][kidx] * dDadh[zidx][kidx])                       
                for kidx, kval in enumerate(self.k_table[zidx])]                          
                for zidx, zval in enumerate(self.z_steps)]
    
            H_fid = [cf.H(self.omega_b_fid, self.omega_cdm_fid, 
                self.omega_ncdm_fid, self.h_fid, zval) 
                for zidx, zval in enumerate(self.z_steps)]
    
            dmudomegab = [[(
                (mu / kval) * dkdomegab[zidx][kidx] 
                + (mu / H_fid[zidx]) * dHdomegab[zidx][kidx])
                for kidx, kval in enumerate(self.k_table[zidx])]
                for zidx, zval in enumerate(self.z_steps)]
    
            dmudomegacdm = [[(                                                        
                (mu / kval) * dkdomegacdm[zidx][kidx]                                             
                + (mu / H_fid[zidx]) * dHdomegacdm[zidx][kidx])                                   
                for kidx, kval in enumerate(self.k_table[zidx])]                          
                for zidx, zval in enumerate(self.z_steps)]
    
            dmudomegancdm = [[(                                                        
                (mu / kval) * dkdomegancdm[zidx][kidx]                                             
                + (mu / H_fid[zidx]) * dHdomegancdm[zidx][kidx])                                   
                for kidx, kval in enumerate(self.k_table[zidx])]                          
                for zidx, zval in enumerate(self.z_steps)]
    
            dmudh = [[(                                                        
                (mu / kval) * dkdh[zidx][kidx]                                             
                + (mu / H_fid[zidx]) * dHdh[zidx][kidx])                                   
                for kidx, kval in enumerate(self.k_table[zidx])]                          
                for zidx, zval in enumerate(self.z_steps)]
    
            # Fix everything below: very redundant, slow, error prone
            # dlogPdmu
            self.gen_rsd((1. + self.dstep) * mu)
            self.gen_fog((1. + self.dstep) * mu)
            logP_mu_high = np.log(self.Pm) + np.log(self.RSD) + np.log(self.FOG) 
     
            self.gen_rsd((1. - self.dstep) * mu)
            self.gen_fog((1. - self.dstep) * mu)
            logP_mu_low = np.log(self.Pm) + np.log(self.RSD) + np.log(self.FOG)
            
            self.gen_rsd(mu)
            self.gen_fog(mu)
            logP_mid = np.log(self.Pm) + np.log(self.RSD) + np.log(self.FOG)
    
            dlogPdmu = (logP_mu_high - logP_mu_low) / (2. * self.dstep * mu) 
    
            # dlogPdk
            dlogPdk = np.zeros((len(self.z_steps), len(self.k_table[0])))
            
            for zidx, zval in enumerate(self.z_steps): 
                for kidx, kval in enumerate(self.k_table[zidx][1:-1]):
                    # Careful with this derivative definition, uneven spacing
                    dlogPdk[zidx][kidx+1] = ((logP_mid[zidx][kidx+2] 
                                              - logP_mid[zidx][kidx])
                                             / (self.k_table[zidx][kidx+2]
                                                - self.k_table[zidx][kidx]))
    
            # Careful with this approximation - is it appropriate? 
            for zidx, zval in enumerate(self.z_steps): 
                dlogPdk[zidx][0] = dlogPdk[zidx][1]
                dlogPdk[zidx][-1] = dlogPdk[zidx][-2]
    
            self.dlogCOVdomega_b = (dlogPdk * dkdomegab 
                + dlogPdmu * dmudomegab)
            self.dlogCOVdomega_cdm = (dlogPdk * dkdomegacdm 
                + dlogPdmu * dmudomegacdm)
            self.dlogCOVdomega_ncdm = (dlogPdk * dkdomegancdm 
                + dlogPdmu * dmudomegancdm)
    
            if self.relic_vary=="T_ncdm":                                            
                self.dlogCOVdT_ncdm = (np.array(self.dlogCOVdomega_ncdm)            
                    * cf.domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.M_ncdm_fid))     
            elif self.relic_vary=="m_ncdm":                                                                   
                self.dlogCOVdM_ncdm = (np.array(self.dlogCOVdomega_ncdm)            
                    * cf.domega_ncdm_dM_ncdm(self.T_ncdm_fid))   
            elif self.relic_vary=="N_ncdm":                                                                  
                self.dlogCOVdN_ncdm = (np.array(self.dlogCOVdomega_ncdm)            
                    * cf.omega_ncdm(
                        self.T_ncdm_fid,
                        self.m_ncdm_fid,
                        self.N_ncdm_fid,
                        'relic'))  
    
            self.dlogCOVdh = (dlogPdk * dkdh 
                + dlogPdmu * dmudh)
        else: 
            self.COV = [[1.                                               
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogCOVdomega_b = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]    
            self.dlogCOVdomega_cdm = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]    
            self.dlogCOVdomega_ncdm = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]    
            if self.relic_vary=="T_ncdm":                                        
                self.dlogCOVdT_ncdm = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]    
            elif self.relic_vary=="m_ncdm":                                                               
                self.dlogCOVdM_ncdm = [[0.
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]    
            elif self.relic_vary=="N_ncdm":                                                              
                self.dlogCOVdN_ncdm = [[0.                                      
                    for kidx, kval in enumerate(self.k_table[zidx])]                
                    for zidx, zval in enumerate(self.z_steps)]   
            self.dlogCOVdh = [[0.
                for kidx, kval in enumerate(self.k_table[zidx])]                
                for zidx, zval in enumerate(self.z_steps)]    

    #MOST INEFFICIENT PART OF CODE 98% OF EVALUATION TIME SPENT HERE    
    def gen_fisher(self, fisher_order, mu_step=0.05, skipgen=False): #inefficient
        
        self.fisher_order = fisher_order

        if skipgen==False: 
            if 'pm' not in self.psterms:                                        
                self.gen_pm() # 5% time spent here 
    
            mu_vals = np.arange(-1., 1., mu_step)
            self.mu_table = mu_vals
    
            Pg = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            Pg_norelicstep = np.zeros(                                                      
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))

            Pg_specialb = np.zeros(  #FOR DEBUGGING. DELETE LATER.                                        
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            Pg_specialb_high = np.zeros(  #FOR DEBUGGING. DELETE LATER.                                        
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            Pg_specialb_low = np.zeros(  #FOR DEBUGGING. DELETE LATER.                                        
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))

            RSD = np.zeros(                                                          
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            RSD_norelicstep = np.zeros(                                                     
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals))) 

            RSD_specialb = np.zeros(  #FOR DEBUGGING. DELETE LATER.                                        
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            RSD_specialb_high = np.zeros(  #FOR DEBUGGING. DELETE LATER.                                        
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals))) 
            RSD_specialb_low = np.zeros(  #FOR DEBUGGING. DELETE LATER.                                        
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals))) 

            FOG = np.zeros(                                                          
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals))) 
            D_Amp =  np.zeros(                                                     
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            b_Amp =  np.zeros(    #FOR DEBUGGING. DELETE LATER.                                              
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))  
            dlogPdA_s = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdn_s = np.zeros(   
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdomega_b = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdomega_cdm = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdh = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdtau_reio = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdomega_ncdm = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdM_ncdm = np.zeros(                                            
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals))) 
            dlogPdT_ncdm = np.zeros(                                                
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdN_ncdm = np.zeros(                                            
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))  
            dlogPdsigmafog = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdb0 = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
            dlogPdalphak2 = np.zeros(
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))        
            dlogPdbeta0 = np.zeros(                                           
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))  
            dlogPdbeta1 = np.zeros(                                           
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))  
            dlogPddeltaL = np.zeros(                                           
                (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
    
            for muidx, muval in enumerate(mu_vals):
                self.gen_rsd(muval) # 39% time spent here
                self.gen_fog(muval) # 12% time spent here
                self.gen_ap() # 30% time spent here 
                self.gen_cov(muval) # 54% time spent here
    
                for zidx, zval in enumerate(self.z_steps): 
                    for kidx, kval in enumerate(self.k_table[zidx]):
    
                        Pg[zidx][kidx][muidx] = ( 1.
                            * self.spectra_mid[zidx].ps_table[kidx] 
                            * self.RSD[zidx][kidx] 
                            * self.FOG[zidx][kidx]
                            * self.AP[zidx][kidx]
                            * self.COV[zidx][kidx]
                            )
                        Pg_norelicstep[zidx][kidx][muidx] = ( 1.                            
                            * self.spectra_mid[zidx].ps_table[kidx]             
                            * self.RSD_norelicstep[zidx][kidx]                              
                            * self.FOG[zidx][kidx]                              
                            * self.AP[zidx][kidx]                               
                            * self.COV[zidx][kidx]                              
                            )

                        Pg_specialb[zidx][kidx][muidx] = ( 1. #FOR DEBUGGING. DELETE LATER.               
                            * self.spectra_mid[zidx].ps_table[kidx]             
                            * self.RSD_specialb[zidx][kidx]                  
                            * self.FOG[zidx][kidx]                              
                            * self.AP[zidx][kidx]                               
                            * self.COV[zidx][kidx]                              
                            )
                        Pg_specialb_high[zidx][kidx][muidx] = ( 1. #FOR DEBUGGING. DELETE LATER.               
                            * self.spectra_mid[zidx].ps_table[kidx]             
                            * self.RSD_specialb_high[zidx][kidx]                     
                            * self.FOG[zidx][kidx]                              
                            * self.AP[zidx][kidx]                               
                            * self.COV[zidx][kidx]                              
                            )
                        Pg_specialb_low[zidx][kidx][muidx] = ( 1. #FOR DEBUGGING. DELETE LATER.               
                            * self.spectra_mid[zidx].ps_table[kidx]             
                            * self.RSD_specialb_low[zidx][kidx]                     
                            * self.FOG[zidx][kidx]                              
                            * self.AP[zidx][kidx]                               
                            * self.COV[zidx][kidx]                              
                            )

                        RSD[zidx][kidx][muidx] = (self.RSD[zidx][kidx]) 
                        RSD_norelicstep[zidx][kidx][muidx] = (
                            self.RSD_norelicstep[zidx][kidx])   

                        RSD_specialb[zidx][kidx][muidx] = ( #FOR DEBUGGING. DELETE LATER.                  
                            self.RSD_specialb[zidx][kidx])
                        RSD_specialb_high[zidx][kidx][muidx] = ( #FOR DEBUGGING. DELETE LATER.                  
                            self.RSD_specialb_high[zidx][kidx]) 
                        RSD_specialb_low[zidx][kidx][muidx] = ( #FOR DEBUGGING. DELETE LATER.                  
                            self.RSD_specialb_low[zidx][kidx]) 

                        FOG[zidx][kidx][muidx] = (self.FOG[zidx][kidx])
                        D_Amp[zidx][kidx][muidx] = (
                            np.log(Pg[zidx][kidx][muidx])
                            - np.log(Pg_norelicstep[zidx][kidx][muidx]))       


#                        b_Amp[zidx][kidx][muidx] = ((  #FOR DEBUGGING. DELETE LATER.                         
#                            Pg[zidx][kidx][muidx]                               
#                            - Pg_specialb[zidx][kidx][muidx])                
#                            / Pg[zidx][kidx][muidx])
#                        b_Amp[zidx][kidx][muidx] = (
#                            np.log(Pg_specialb[zidx][kidx][muidx]) 
#                            - np.log(Pg[zidx][kidx][muidx]))
                        b_Amp[zidx][kidx][muidx] = ((
                            np.log(Pg_specialb_high[zidx][kidx][muidx])
                            - np.log(Pg_specialb_low[zidx][kidx][muidx])
                            ))

                        dlogPdA_s[zidx][kidx][muidx] = (
                            self.dlogPdA_s[zidx][kidx]
                            )
                        dlogPdn_s[zidx][kidx][muidx] = (
                            self.dlogPdn_s[zidx][kidx]
                            )
                        dlogPdomega_b[zidx][kidx][muidx] = (
                            self.dlogPdomega_b[zidx][kidx]
                            + self.dlogRSDdomega_b[zidx][kidx]
                            + self.dlogFOGdomega_b[zidx][kidx]
                            + self.dlogAPdomega_b[zidx][kidx]
                            + self.dlogCOVdomega_b[zidx][kidx]
                            )
                        dlogPdomega_cdm[zidx][kidx][muidx] = (
                            self.dlogPdomega_cdm[zidx][kidx]
                            + self.dlogRSDdomega_cdm[zidx][kidx]
                            + self.dlogFOGdomega_cdm[zidx][kidx]
                            + self.dlogAPdomega_cdm[zidx][kidx]
                            + self.dlogCOVdomega_cdm[zidx][kidx]
                            )
                        dlogPdh[zidx][kidx][muidx] = (
                            self.dlogPdh[zidx][kidx]
                            + self.dlogRSDdh[zidx][kidx] 
                            + self.dlogFOGdh[zidx][kidx]
                            + self.dlogAPdh[zidx][kidx]
                            + self.dlogCOVdh[zidx][kidx]
                            )
                        dlogPdtau_reio[zidx][kidx][muidx] = (
                            self.dlogPdtau_reio[zidx][kidx]
                            ) 
                        dlogPdomega_ncdm[zidx][kidx][muidx] = (
                            self.dlogPdomega_ncdm[zidx][kidx]
                            + self.dlogRSDdomega_ncdm[zidx][kidx]
                            + self.dlogFOGdomega_ncdm[zidx][kidx]
                            + self.dlogAPdomega_ncdm[zidx][kidx]
                            + self.dlogCOVdomega_ncdm[zidx][kidx]
                            )
    
                        if self.relic_vary=="T_ncdm": 
                            dlogPdT_ncdm[zidx][kidx][muidx] = (                     
                                dlogPdomega_ncdm[zidx][kidx][muidx]                 
                                * cf.domega_ncdm_dT_ncdm(
                                    self.T_ncdm_fid, self.M_ncdm_fid)) 
                        elif self.relic_vary=="m_ncdm": 
                            dlogPdM_ncdm[zidx][kidx][muidx] = (
                                dlogPdomega_ncdm[zidx][kidx][muidx] 
                                * cf.domega_ncdm_dM_ncdm(self.T_ncdm_fid)) 
                        elif self.relic_vary=="N_ncdm":                         
                            dlogPdN_ncdm[zidx][kidx][muidx] = (                 
                                dlogPdomega_ncdm[zidx][kidx][muidx]             
                                * cf.omega_ncdm(
                                    self.T_ncdm_fid,
                                    self.m_ncdm_fid,
                                    self.N_ncdm_fid,
                                    'relic')) 
                        if self.forecast=='neutrino':
                            dlogPdM_ncdm[zidx][kidx][muidx] = (                 
                                dlogPdomega_ncdm[zidx][kidx][muidx]             
                                * (1./cf.NEUTRINO_SCALE_FACTOR))
                        # ^^^Careful, this overwrites the earlier dP_g value. 
    
                        dlogPdsigmafog[zidx][kidx][muidx] = (                       
                            self.dlogFOGdsigmafog0[zidx][kidx]
                            )
    
                        dlogPdb0[zidx][kidx][muidx] = (
                            self.dlogRSDdb0[zidx][kidx]    
                            )
    
                        dlogPdalphak2[zidx][kidx][muidx] = (                          
                            self.dlogRSDdalphak2[zidx][kidx]                                                    
                            )
                        dlogPdbeta0[zidx][kidx][muidx] = (                    
                            self.dlogRSDdbeta0[zidx][kidx]                    
                            )
                        dlogPdbeta1[zidx][kidx][muidx] = (                    
                            self.dlogRSDdbeta1[zidx][kidx]                    
                            )
                        dlogPddeltaL[zidx][kidx][muidx] = (                    
                            self.dlogRSDddeltaL[zidx][kidx]                    
                            ) 
            self.Pg = Pg
            self.Pg_norelicstep = Pg_norelicstep

            self.Pg_specialb = Pg_specialb #FOR DEBUGGING. DELETE LATER.
            self.Pg_specialb_high = Pg_specialb_high #FOR DEBUGGING. DELETE LATER.
            self.Pg_specialb_low = Pg_specialb_low #FOR DEBUGGING. DELETE LATER.

            self.RSD = RSD
            self.RSD_norelicstep = RSD_norelicstep

            self.RSD_specialb = RSD_specialb #FOR DEBUGGING. DELETE LATER.
            self.RSD_specialb_high = RSD_specialb_high #FOR DEBUGGING. DELETE LATER.
            self.RSD_specialb_low = RSD_specialb_low #FOR DEBUGGING. DELETE LATER.

            self.FOG = FOG
            self.D_Amp = D_Amp
            self.b_Amp = b_Amp #FOR DEBUGGING. DELETE LATER. 
    
            self.dlogPmdA_s = np.array(self.dlogPdA_s)
            self.dlogPmdn_s = np.array(self.dlogPdn_s)
            self.dlogPmdomega_b = np.array(self.dlogPdomega_b)
            self.dlogPmdomega_cdm = np.array(self.dlogPdomega_cdm)
            self.dlogPmdh = np.array(self.dlogPdh)
            self.dlogPmdtau_reio = np.array(self.dlogPdtau_reio)
            self.dlogPmdomega_ncdm = np.array(self.dlogPdomega_ncdm)
            if self.relic_vary=="T_ncdm":  
                self.dlogPmdT_ncdm  = np.array(self.dlogPdT_ncdm) 
            elif self.relic_vary=="m_ncdm": 
                self.dlogPmdM_ncdm = np.array(self.dlogPdM_ncdm)
            elif self.relic_vary=="N_ncdm":                                     
                self.dlogPmdN_ncdm = np.array(self.dlogPdN_ncdm)

            if self.forecast=='neutrino': 
                self.dlogPmdM_ncdm = np.array(self.dlogPdM_ncdm)

            self.dlogPgdA_s = np.array(dlogPdA_s)
            self.dlogPgdn_s = np.array(dlogPdn_s)
            self.dlogPgdomega_b  = np.array(dlogPdomega_b)
            self.dlogPgdomega_cdm = np.array(dlogPdomega_cdm)
            self.dlogPgdh = np.array(dlogPdh)
            self.dlogPgdH_0 = np.array(dlogPdh) / 100. 
            self.dlogPgdtau_reio = np.array(dlogPdtau_reio)
            self.dlogPgdomega_ncdm = np.array(dlogPdomega_ncdm)
            self.dlogPgdM_ncdm = np.array(dlogPdM_ncdm)
            self.dlogPgdT_ncdm = np.array(dlogPdT_ncdm)
            self.dlogPgdN_ncdm = np.array(dlogPdN_ncdm)
            self.dlogPgdsigmafog = np.array(dlogPdsigmafog)
            self.dlogPgdb0 = np.array(dlogPdb0)
            self.dlogPgdalphak2 = np.array(dlogPdalphak2) 
            self.dlogPgdbeta0 = np.array(dlogPdbeta0)
            self.dlogPgdbeta1 = np.array(dlogPdbeta1)      
            self.dlogPgddeltaL = np.array(dlogPddeltaL) 

        param_dict = {
            'omega_b' : self.dlogPgdomega_b,
            'omega_cdm' : self.dlogPgdomega_cdm,
            'n_s' : self.dlogPgdn_s,
            'A_s' : self.dlogPgdA_s,
            'tau_reio' : self.dlogPgdtau_reio,
            'h' : self.dlogPgdh,
            'H_0' : self.dlogPgdH_0, 
            'M_ncdm' : self.dlogPgdM_ncdm,
            'omega_ncdm' : self.dlogPgdomega_ncdm,
            'T_ncdm' : self.dlogPgdT_ncdm,
            'N_ncdm' : self.dlogPgdN_ncdm,
            'sigma_fog' : self.dlogPgdsigmafog,
            'b0' : self.dlogPgdb0,
            'alpha_k2' : self.dlogPgdalphak2,
            'beta0' : self.dlogPgdbeta0,
            'beta1' : self.dlogPgdbeta1,
            'D_Amp' : self.D_Amp,
            'b_Amp' : self.b_Amp,
            'delta_L' : self.dlogPgddeltaL}

        paramvec =  [] 

        for parameter in self.fisher_order:
            if parameter=='b0':
                if (self.use_rsd==True) and ('b0' in self.lss_survey_params):
                    paramvec.append(param_dict[parameter])
                else:
                    print("RSD not requested. Can't forecast b0.")
                    self.fisher_order.remove('b0') 
            elif (parameter=='alpha_k2') and ('alphak2' 
                in self.lss_survey_params): 
                if self.use_rsd==True:                      
                    paramvec.append(param_dict[parameter]) 
                else: 
                    print("RSD not requested. Can't forecast alpha_k2.")
                    self.fisher_order.remove('alpha_k2')
            elif (parameter=='beta0') and ('beta0' in self.lss_survey_params):    
                if self.use_rsd==True:                                          
                    paramvec.append(param_dict[parameter])                      
                else:                                                           
                    print("RSD not requested. Can't forecast beta0.")        
                    self.fisher_order.remove('beta0') 
            elif (parameter=='beta1') and ('beta1' in self.lss_survey_params):    
                if self.use_rsd==True:                                          
                    paramvec.append(param_dict[parameter])                      
                else:                                                           
                    print("RSD not requested. Can't forecast beta1.")        
                    self.fisher_order.remove('beta1') 
            elif parameter=='delta_L':                                         
                if self.use_rsd==True:                                          
                    paramvec.append(param_dict[parameter])                      
                else:                                                           
                    print("RSD not requested. Can't forecast delta_L.")        
                    self.fisher_order.remove('delta_L')
            elif parameter=='sigma_fog':
                if self.use_fog==True:
                    paramvec.append(param_dict[parameter])
                else: 
                    print("FOG not requested. Can't forecast sigma_fog.")
                    self.fisher_order.remove('sigma_fog')
            else: 
                paramvec.append(param_dict[parameter])

        fisher = np.zeros((len(paramvec), len(paramvec))) 

        # Highly inefficient set of loops 
        for pidx1, p1 in enumerate(paramvec): 
            for pidx2, p2 in enumerate(paramvec[pidx1:]):
                integrand = np.zeros((len(self.z_steps), len(self.k_table[0])))
                integral = np.zeros(len(self.z_steps))
                for zidx, zval in enumerate(self.z_steps):
                    Volume = self.V(zidx)
                    # print("For z = ", zval, ", V_eff = ", 
                    #       (Volume*self.h_fid)/(1.e9), " [h^{-1} Gpc^{3}]") 
                    for kidx, kval in enumerate(self.k_table[zidx]): # Center intgrl? 
                        integrand[zidx][kidx] = np.sum( #Perform mu integral
                            mu_step 
                            * paramvec[pidx1][zidx][kidx]
                            * paramvec[pidx1+pidx2][zidx][kidx]
                            * np.power(kval, 2.)
                            * (1. / (8.  * np.power(np.pi, 2)))
                            * cf.neff(self.n_densities[zidx], 
                                self.Pg[zidx][kidx]) 
                            * Volume)
                            
                for zidx, zval in enumerate(self.z_steps): 
                    val = 0 
                    for kidx, kval in enumerate(self.k_table[zidx][:-1]): 
                        #Center?
                        #Approximating average in k_bin to integrate over k
                        val += (((integrand[zidx][kidx] 
                                  + integrand[zidx][kidx+1])
                                 / 2.)  
                                * (self.k_table[zidx][kidx+1]
                                    -self.k_table[zidx][kidx]))
                    integral[zidx] = val  
                fisher[pidx1][pidx2+pidx1] = np.sum(integral)
                fisher[pidx2+pidx1][pidx1] = np.float(
                    fisher[pidx1][pidx2+pidx1])
                print("Fisher element (", pidx1, ", ", (pidx2+pidx1),"), (", 
                    (pidx2+pidx1), ", ", pidx1,") calculated...")
#                print(fisher[pidx1][pidx1+pidx2]) 
        self.fisher=fisher

    def reorder_fisher(self, fisher_order): 
        if type(self.fisher)==type(None):
            print("Error: You have to first run gen_fisher(...).")
            return
        else: 
            self.fisher_order = fisher_order    
            param_dict = {                                                          
                'omega_b' : self.dlogPgdomega_b,                                    
                'omega_cdm' : self.dlogPgdomega_cdm,                                
                'n_s' : self.dlogPgdn_s,                                            
                'A_s' : self.dlogPgdA_s,                                            
                'tau_reio' : self.dlogPgdtau_reio,                                  
                'h' : self.dlogPgdh,                                                
                'H_0' : self.dlogPgdH_0,                                            
                'M_ncdm' : self.dlogPgdM_ncdm,                                      
                'omega_ncdm' : self.dlogPgdomega_ncdm,                              
                'T_ncdm' : self.dlogPgdT_ncdm,
                'N_ncdm' : self.dlogPgdN_ncdm,                                      
                'sigma_fog' : self.dlogPgdsigmafog,                                 
                'b0' : self.dlogPgdb0,                                              
                'alpha_k2' : self.dlogPgdalphak2,                                   
                'beta0' : self.dlogPgdbeta0,                                        
                'beta1' : self.dlogPgdbeta1,                                        
                'D_Amp' : self.D_Amp,                                               
                'b_Amp' : self.b_Amp,                                               
                'delta_L' : self.dlogPgddeltaL}                                     
                                                                                
            paramvec =  [] 

            for parameter in self.fisher_order:                                     
                if parameter=='b0':                                                 
                    if (self.use_rsd==True) and ('b0' in self.lss_survey_params):   
                        paramvec.append(param_dict[parameter])                      
                    else:                                                           
                        print("RSD not requested. Can't forecast b0.")              
                        self.fisher_order.remove('b0')                              
                elif (parameter=='alpha_k2') and ('alphak2'                         
                    in self.lss_survey_params):                                     
                    if self.use_rsd==True:                                          
                        paramvec.append(param_dict[parameter])                      
                    else:                                                           
                        print("RSD not requested. Can't forecast alpha_k2.")        
                        self.fisher_order.remove('alpha_k2')                        
                elif (parameter=='beta0') and ('beta0' in self.lss_survey_params):  
                    if self.use_rsd==True:                                          
                        paramvec.append(param_dict[parameter])                      
                    else:                                                           
                        print("RSD not requested. Can't forecast beta0.")           
                        self.fisher_order.remove('beta0')                           
                elif (parameter=='beta1') and ('beta1' in self.lss_survey_params):  
                    if self.use_rsd==True:                                          
                        paramvec.append(param_dict[parameter])                      
                    else:                                                           
                        print("RSD not requested. Can't forecast beta1.")           
                        self.fisher_order.remove('beta1')                           
                elif parameter=='delta_L':                                          
                    if self.use_rsd==True:                                          
                        paramvec.append(param_dict[parameter])                      
                    else:                                                           
                        print("RSD not requested. Can't forecast delta_L.")         
                        self.fisher_order.remove('delta_L')                         
                elif parameter=='sigma_fog':                                        
                    if self.use_fog==True:                                          
                        paramvec.append(param_dict[parameter])                      
                    else:                                                           
                        print("FOG not requested. Can't forecast sigma_fog.")       
                        self.fisher_order.remove('sigma_fog')                       
                else:                                                               
                    paramvec.append(param_dict[parameter])                          
                                                                                    
            fisher = np.zeros((len(paramvec), len(paramvec)))
            mu_step = self.mu_table[1] - self.mu_table[0]

            # Highly inefficient set of loops                                       
            for pidx1, p1 in enumerate(paramvec):                                   
                for pidx2, p2 in enumerate(paramvec[pidx1:]):                       
                    integrand = np.zeros((len(self.z_steps), len(self.k_table[0]))) 
                    integral = np.zeros(len(self.z_steps))                          
                    for zidx, zval in enumerate(self.z_steps):                      
                        Volume = self.V(zidx)                                       
                        # print("For z = ", zval, ", V_eff = ",                     
                        #       (Volume*self.h_fid)/(1.e9), " [h^{-1} Gpc^{3}]")    
                        for kidx, kval in enumerate(self.k_table[zidx]): # Center intgrl? 
                            integrand[zidx][kidx] = np.sum( #Perform mu integral    
                                mu_step                                             
                                * paramvec[pidx1][zidx][kidx]                       
                                * paramvec[pidx1+pidx2][zidx][kidx]                 
                                * np.power(kval, 2.)                                
                                * (1. / (8.  * np.power(np.pi, 2)))                 
                                * cf.neff(self.n_densities[zidx],                   
                                    self.Pg[zidx][kidx])                            
                                * Volume)                                           
                                                                                    
                    for zidx, zval in enumerate(self.z_steps):                      
                        val = 0                                                     
                        for kidx, kval in enumerate(self.k_table[zidx][:-1]):       
                            #Center?                                                
                            #Approximating average in k_bin to integrate over k     
                            val += (((integrand[zidx][kidx]                         
                                      + integrand[zidx][kidx+1])                    
                                     / 2.)                                          
                                    * (self.k_table[zidx][kidx+1]                   
                                        -self.k_table[zidx][kidx]))                 
                        integral[zidx] = val                                        
                    fisher[pidx1][pidx2+pidx1] = np.sum(integral)                   
                    fisher[pidx2+pidx1][pidx1] = np.float(                          
                        fisher[pidx1][pidx2+pidx1])                                 
                    print("Fisher element (", pidx1, ", ", (pidx2+pidx1),"), (",    
                        (pidx2+pidx1), ", ", pidx1,") calculated...")               
    #                print(fisher[pidx1][pidx1+pidx2])                              
            self.fisher=fisher


    def generate_spectra(
        self,
        param,
        manual_high=None,
        manual_low=None):

        if (manual_high==None) and (manual_low==None): 
            step_high = (1. + self.dstep) * self.fid[param]
            step_low = (1. - self.dstep) * self.fid[param]
        else: 
            step_high = manual_high                         
            step_low = manual_low    
                                         
        spectra_high = [cf.spectrum(                                            
                            cf.generate_data(                                   
                                dict(self.fid,                                  
                                     **{'z_pk' : zval,                             
                                     param : step_high}), 
                                self.classdir,                                  
                                self.datastore)[0:-20],                         
                            self.fsky,
                            k_table=self.k_table[zidx],
                            forecast=self.forecast)                                   
                        for zidx, zval in enumerate(self.z_steps)]                                  
        spectra_low = [cf.spectrum(                                             
                            cf.generate_data(                                    
                                dict(self.fid,                                   
                                    **{'z_pk' : zval,                              
                                    param : step_low}),  
                                self.classdir,                                   
                                self.datastore)[0:-20],                          
                            self.fsky,
                            k_table=self.k_table[zidx],
                            forecast=self.forecast)                                        
                       for zidx, zval in enumerate(self.z_steps)]                                   
        return spectra_high, spectra_low 

    def V(self, zidx):                                                          
        delta_z = self.z_steps[-1] - self.z_steps[-2] 
        omega_m = self.omega_b_fid + self.omega_cdm_fid + self.omega_ncdm_fid   
        omega_lambda = np.power(self.h_fid, 2.) - omega_m                       
        zmax = self.z_steps[zidx]+(0.5 *  delta_z)                                              
        zmin = self.z_steps[zidx]-(0.5 *  delta_z)                                                
        zsteps = 100.
        dz = ((zmin+zmax)/2.) / zsteps    
        z_table_max = np.arange(0., zmax, dz)                                   
        z_table_min = np.arange(0., zmin, dz)                                   
        z_integrand_max = ((self.h_fid * dz)                                    
                           /  np.sqrt(omega_m                                   
                                      * np.power(1. + z_table_max, 3.)          
                                      + omega_lambda))                          
        z_integrand_min = ((self.h_fid * dz)                                    
                           /  np.sqrt(omega_m                                   
                                      * np.power(1. + z_table_min, 3.)          
                                      + omega_lambda))                          
        z_integral_max = np.sum(z_integrand_max)                                
        z_integral_min = np.sum(z_integrand_min)                                
        V_max = ((4. * np.pi / 3.)                                              
                   * np.power(cf.C / (1000. * 100. * self.h_fid), 3.)           
                   * np.power(z_integral_max, 3.))                              
        V_min = ((4. * np.pi / 3.)                                              
                   * np.power(cf.C / (1000. * 100. * self.h_fid), 3.)           
                   * np.power(z_integral_min, 3.))                              
        Volume = (V_max - V_min) * self.fsky                                    
        self.n_densities[zidx] = (self.fcoverage_deg                            
                                  * self.dNdz[zidx]                             
                                  * (zmax - zmin) / Volume)                     
        return Volume                                                           
                                                                                
    def print_v_table(self, k_index):                                                    
        for zidx, zval in enumerate(self.z_steps):                              
            V = self.V(zidx)                                                    
            n = self.n_densities[zidx]                                          
            n2 = (1.e9 / np.power(self.h_fid, 3.)) * n     
                     
            print((("For z = {0:.2f} \n\t") +                                   
                   ("V = {1:.6f} [h^-3 Gpc^3] \n\t") +                          
                   ("nbar = {2:.6f} [h^3 Gpc^-3] \n\t") +                       
                   ("nbar = {3:.6e} [Mpc^-3] \n\t") +                           
                   ("nbar/deg^2 = {4:.6e} [h^3 Gpc^-3 deg^-2] \n\t") +          
                   ("nbar/deg^2 = {5:.6e} [Mpc^-3 deg^-2] \n\t") +              
                   ("D = {6:.6f} \n\t") +                                       
                   ("b_ELG = {7:.6f} \n\t") +
                   ("P_m(k={8:.3f}) = {9:.2f} \n\t")).format(                            
                       zval,                                                    
                       (V*np.power(self.h_fid, 3.)) / (1.e9),                   
                       n2,                                                      
                       n,                                                       
                       n2 / self.fcoverage_deg,                                 
                       n / self.fcoverage_deg,                                  
                       self.spectra_mid[zidx].D,                    
                       cf.DB_ELG / self.spectra_mid[zidx].D,
                       self.k_table[zidx][k_index],
                       self.Pm[zidx][k_index]))            
        return
                
    def print_P_table(self, k_index, mu_index):
        print("P_g with following corrections: " + str(self.psterms)) 
        for zidx, zval in enumerate(self.z_steps): 
            print((
                ("For z = {0:.2f},\t") 
                + (" Pg({1:.2f}, {2:.2f}) = {3:.2f}\n")
                ).format(
                    zval,
                    self.k_table[zidx][k_index], 
                    self.mu_table[mu_index], 
                    self.Pg[zidx][k_index][mu_index]))


    def load_cmb_fisher(self, fisher_order, fisherpath, colnames=None): 

        valid_params = [
            'omega_b', 
            'omega_cdm', 
            'n_s', 
            'A_s', 
            'tau_reio', 
            'H_0', 
            'M_ncdm',
            'm_ncdm',
            'N_ncdm', 
            'h', 
            'T_ncdm',
            'T_ncdm[gamma]', 
            'omega_ncdm']

        for parameter in fisher_order: 
            if parameter not in valid_params: 
                print("Invalid parameter specified for CMB Fisher!")
                print("No information loaded...")  
                return

        fmat = pd.read_csv(fisherpath, sep='\t', header=0)

        for pidx, pval in enumerate(fisher_order): 
            if pval not in fmat.columns: 
                if (pval=='T_ncdm') and ('T_ncdm[gamma]' in fmat.columns): 
                    # Change of variable T[CMB] -> T[K] 
                    print('Converting T[CMB] --> T[K]') 
                    index =  fmat.columns.get_loc('T_ncdm[gamma]') 
                    fmat.iloc[:, index] *= 1./self.T_cmb_fid 
                    fmat.iloc[index, :] *= 1./self.T_cmb_fid
                    fmat = fmat.rename(index=str, 
                        columns={"T_ncdm[gamma]": "T_ncdm"}) 
                elif (pval=='T_ncdm[gamma]') and ('T_ncdm' in fmat.columns):      
                    # Change of variable T[K] -> T[CMB]                         
                    print('Converting T[K] --> T[CMB]')                         
                    index =  fmat.columns.get_loc('T_ncdm')              
                    fmat.iloc[:, index] *= self.T_cmb_fid                    
                    fmat.iloc[index, :] *= self.T_cmb_fid
                    fmat = fmat.rename(index=str,                               
                        columns={"T_ncdm": "T_ncdm[gamma]"}) 
                elif (pval=='h') and ('H_0' in fmat.columns): 
                    # Change of variables H0->h
                    print('Converting H_0 --> h...') 
                    index = fmat.columns.get_loc('H_0') 
                    fmat.iloc[:,index] *= 100. 
                    fmat.iloc[index,:] *= 100.  
                    fmat = fmat.rename(index=str, columns={"H_0": "h"})
                elif (pval=='H_0')  and ('h' in fmat.columns):
                    # Change of variables h->H_0  
                    print('Converting h --> H_0...') 
                    index = fmat.columns.get_loc('h')                         
                    fmat.iloc[:,index] *= 1./100.      
                    fmat.iloc[index,:] *= 1./100.  
                    fmat = fmat.rename(index=str, columns={"h": "H_0"})

                elif self.forecast=='neutrino': 
                    if (pval=='omega_ncdm') and ('M_ncdm' in fmat.columns): 
                        # Change of variables M->omega_ncdm
                        print('Converting M_ncdm --> omega_ncdm...') 
                        index = fmat.columns.get_loc('M_ncdm')
                        dM_ncdm_domega_ncdm = cf.dM_ncdm_domega_ncdm(
                            cf.RELIC_TEMP_SCALE)
                        print('dM_ncdm/domega_ncdm = ', dM_ncdm_domega_ncdm) 
                        fmat.iloc[:,index] *= dM_ncdm_domega_ncdm
                        fmat.iloc[index,:] *= dM_ncdm_domega_ncdm
                        fmat = fmat.rename(
                            index=str, 
                            columns={"M_ncdm": "omega_ncdm"}) 
                    elif (pval=='M_ncdm') and ('omega_ncdm' in fmat.columns): 
                        # Change of variables omega_ncdm --> M_ncdm 
                        print('Converting omega_ncdm --> M_ncdm...')  
                        index = fmat.columns.get_loc('omega_ncdm')                  
                        dM_ncdm_domega_ncdm = cf.dM_ncdm_domega_ncdm(           
                            cf.RELIC_TEMP_SCALE)                                
                        fmat.iloc[:,index] *= 1./dM_ncdm_domega_ncdm               
                        fmat.iloc[index,:] *= 1./dM_ncdm_domega_ncdm               
                        fmat = fmat.rename(                                     
                            index=str,                                          
                            columns={"omega_ncdm": "M_ncdm"})
                    if (pval=='omega_ncdm') and ('m_ncdm' in fmat.columns):     
                        # Change of variables m->omega_ncdm                     
                        print('Converting m_ncdm --> omega_ncdm...')            
                        index = fmat.columns.get_loc('m_ncdm')                  
                        dM_ncdm_domega_ncdm = cf.dM_ncdm_domega_ncdm(           
                            cf.RELIC_TEMP_SCALE)
                        dm_dM = 3.
                        fmat.iloc[:,index] *= dM_ncdm_domega_ncdm * dm_dM               
                        fmat.iloc[index,:] *= dM_ncdm_domega_ncdm * dm_dM               
                        fmat = fmat.rename(                                     
                            index=str,                                          
                            columns={"m_ncdm": "omega_ncdm"})                   
                    elif (pval=='m_ncdm') and ('omega_ncdm' in fmat.columns):   
                        # Change of variables omega_ncdm --> m_ncdm             
                        print('Converting omega_ncdm --> m_ncdm...')            
                        index = fmat.columns.get_loc('omega_ncdm')              
                        dM_ncdm_domega_ncdm = cf.dM_ncdm_domega_ncdm(           
                            cf.RELIC_TEMP_SCALE) 
                        dm_dM = 3.   
                        fmat.iloc[:,index] *= 1./(dM_ncdm_domega_ncdm * dm_dM)            
                        fmat.iloc[index,:] *= 1./(dM_ncdm_domega_ncdm * dm_dM)            
                        fmat = fmat.rename(                                     
                            index=str,                                          
                            columns={"omega_ncdm": "m_ncdm"})

                elif self.forecast=='relic': 
                    if (pval=='omega_ncdm') and ('T_ncdm' in fmat.columns):
                        # Change T_ncdm --> omega_ncdm
                        print('Converting T_ncdm --> omega_ncdm...')
                        index = fmat.columns.get_loc('T_ncdm') 
                        dT_ncdm_domega_ncdm = cf.dT_ncdm_domega_ncdm(                      
                            self.T_ncdm_fid, self.M_ncdm_fid)
                        fmat.iloc[:,index] *= dT_ncdm_domega_ncdm
                        fmat.iloc[index,:] *= dT_ncdm_domega_ncdm
                        fmat = fmat.rename(
                            index=str, columns={"T_ncdm": "omega_ncdm"})
                    elif (pval=='T_ncdm') and (pval=='omega_ncdm'): 
                        # Change omega_ncdm --> T_ncdm
                        print('Converting omega_ncdm --> T_ncdm...')
                        index = fmat.columns.get_loc('omega_ncdm')                  
                        dT_ncdm_domega_ncdm = cf.dT_ncdm_domega_ncdm(           
                            self.T_ncdm_fid, self.M_ncdm_fid)                   
                        fmat.iloc[:,index] *= 1./dT_ncdm_domega_ncdm               
                        fmat.iloc[index,:] *= 1./dT_ncdm_domega_ncdm               
                        fmat = fmat.rename(                                     
                            index=str, columns={"omega_ncdm": "T_ncdm"})

                    if (pval=='omega_ncdm') and ('T_ncdm[gamma]' in fmat.columns):     
                        # Change T_ncdm[CMB] --> omega_ncdm                          
                        print('Converting T_ncdm[CMB] --> omega_ncdm...')            
                        index = fmat.columns.get_loc('T_ncdm[gamma]')       
                        fmat.iloc[:, index] *= 1./self.T_cmb_fid                    
                        fmat.iloc[index, :] *= 1./self.T_cmb_fid           
                        dT_ncdm_domega_ncdm = cf.dT_ncdm_domega_ncdm(           
                            self.T_ncdm_fid, self.M_ncdm_fid)                   
                        fmat.iloc[:,index] *= dT_ncdm_domega_ncdm               
                        fmat.iloc[index,:] *= dT_ncdm_domega_ncdm               
                        fmat = fmat.rename(                                     
                            index=str, columns={"T_ncdm[gamma]": "omega_ncdm"})        
                    elif (pval=='T_ncdm[gamma]') and (pval=='omega_ncdm'):             
                        # Change omega_ncdm --> T_ncdm[CMB]                          
                        print('Converting omega_ncdm --> T_ncdm[CMB]...')            
                        index = fmat.columns.get_loc('omega_ncdm')              
                        dT_ncdm_domega_ncdm = cf.dT_ncdm_domega_ncdm(           
                            self.T_ncdm_fid, self.M_ncdm_fid)                   
                        fmat.iloc[:,index] *= 1./dT_ncdm_domega_ncdm            
                        fmat.iloc[index,:] *= 1./dT_ncdm_domega_ncdm
                        fmat.iloc[:, index] *= self.T_cmb_fid                       
                        fmat.iloc[index, :] *= self.T_cmb_fid             
                        fmat = fmat.rename(                                     
                            index=str, columns={"omega_ncdm": "T_ncdm[gamma]"}) 

                    elif (pval=='omega_ncdm') and ('M_ncdm' in fmat.columns):     
                        # Change M_ncdm --> omega_ncdm              
                        print('Converting M_ncdm --> omega_ncdm...')             
                        index = fmat.columns.get_loc('M_ncdm')                  
                        dM_ncdm_domega_ncdm = cf.dM_ncdm_domega_ncdm(           
                            self.T_ncdm_fid)  
                        print('T_ncdm = ', self.T_ncdm_fid)
                        print('dM_ncdm/domega_ncdm = ', dM_ncdm_domega_ncdm)                 
                        fmat.iloc[:,index] *= dM_ncdm_domega_ncdm               
                        fmat.iloc[index,:] *= dM_ncdm_domega_ncdm               
                        fmat = fmat.rename(                                     
                            index=str, columns={"M_ncdm": "omega_ncdm"})        
                    elif (pval=='M_ncdm') and ('omega_ncdm' in fmat.columns):     
                        # Change omega_ncdm --> M_ncdm               
                        print('Converting omega_ncdm --> M_ncdm...')            
                        index = fmat.columns.get_loc('omega_ncdm')                  
                        dM_ncdm_domega_ncdm = cf.dM_ncdm_domega_ncdm(           
                            self.T_ncdm_fid)                                    
                        fmat.iloc[:,index] *= 1./dM_ncdm_domega_ncdm               
                        fmat.iloc[index,:] *= 1./dM_ncdm_domega_ncdm               
                        fmat = fmat.rename(                                     
                            index=str, columns={"omega_ncdm": "M_ncdm"})

                    elif (pval=='omega_ncdm') and ('m_ncdm' in fmat.columns):   
                        # Change m_ncdm --> omega_ncdm                          
                        print('Converting m_ncdm --> omega_ncdm...')            
                        index = fmat.columns.get_loc('m_ncdm')                  
                        dM_ncdm_domega_ncdm = cf.dM_ncdm_domega_ncdm(           
                            self.T_ncdm_fid)           
                        dm_dM = 1.                         
                        fmat.iloc[:,index] *= dM_ncdm_domega_ncdm * dm_dM               
                        fmat.iloc[index,:] *= dM_ncdm_domega_ncdm * dm_dM              
                        fmat = fmat.rename(                                     
                            index=str, columns={"m_ncdm": "omega_ncdm"})        
                    elif (pval=='m_ncdm') and ('omega_ncdm' in fmat.columns):   
                        # Change omega_ncdm --> m_ncdm                          
                        print('Converting omega_ncdm --> m_ncdm...')            
                        index = fmat.columns.get_loc('omega_ncdm')              
                        dM_ncdm_domega_ncdm = cf.dM_ncdm_domega_ncdm(           
                            self.T_ncdm_fid)              
                        dm_dM = 1.                      
                        fmat.iloc[:,index] *= 1./(dM_ncdm_domega_ncdm*dm_dM)           
                        fmat.iloc[index,:] *= 1./(dM_ncdm_domega_ncdm*dm_dM)            
                        fmat = fmat.rename(                                     
                            index=str, columns={"omega_ncdm": "m_ncdm"}) 
                    elif (pval=='N_ncdm') and ('T_ncdm[gamma]' in fmat.columns): 
                        print('Converting T_ncdm[gamma] --> N_ncdm...')
                        index = fmat.columns.get_loc('T_ncdm[gamma]') 
                        dT_ncdm_dN_ncdm = cf.dT_ncdm_dN_ncdm(
                            self.m_ncdm_fid, 
                            self.omega_ncdm_fid,
                            self.N_ncdm_fid) 
                        dTcmb_dT = 1./cf.dT_dTcmb(self.T_cmb_fid) 
                        dTcmb_dN_ncdm = dTcmb_dT * dT_ncdm_dN_ncdm 
                        fmat.iloc[:,index] *= dTcmb_dN_ncdm    
                        fmat.iloc[index,:] *= dTcmb_dN_ncdm
                        fmat = fmat.rename(                                     
                            index=str, columns={"T_ncdm[gamma]": "N_ncdm"})  
                    

        for pidx, pval in enumerate(fmat.columns): 
            if pval != self.fisher_order[pidx]:
                print(fmat.columns)
                print(self.fisher_order)  
                print("Parameters in input file don't match requested \
                    ordering and parameter re-ordering hasn't yet been \
                    implemented for prior matrix loading. Please manually \
                    re-order rows/columns in your data file and then try \
                    again.") 
                return    

        self.numpy_cmb_fisher =  np.array(fmat)                             
        self.pandas_cmb_fisher = fmat

        self.numpy_cmb_covariance = np.linalg.inv(self.pandas_cmb_fisher)   
        self.pandas_cmb_covariance = pd.DataFrame(
            np.linalg.inv(self.pandas_cmb_fisher),                                        
            columns=self.pandas_cmb_fisher.columns)

        print("The following CMB Fisher matrix was loaded: ")
        print(self.pandas_cmb_fisher) 

    def export_matrices(self, path): 
        if self.fisher==[]:  
            print("No LSS Fisher matrix has been generated. Please execute \
                    the `forecast.gen_fisher()` function.`") 
        else:  
            lssfisher = pd.DataFrame(np.array(self.fisher), 
                columns=self.fisher_order)

#            if self.use_fog==True: 
#                fogindex = lssfisher.columns.get_loc('sigma_fog')               
#                lssfisher.iloc[:, fogindex] *= 1e3 #To correct units on sigma_fog
#                lssfisher.iloc[fogindex ,:] *= 1e3 #To correct units on sigam_fog

            self.pandas_lss_fisher = lssfisher                              
            self.numpy_lss_fisher = np.array(lssfisher)

            self.pandas_lss_covariance = pd.DataFrame(np.linalg.inv(        
                lssfisher), columns=self.fisher_order)                                  
            self.numpy_lss_covariance = np.array(np.linalg.inv(lssfisher))

            #if self.forecast=="neutrino":
                #lssfisher.iloc[:,6] *= 3. # total to single neutrino mass
                #lssfisher.iloc[6,:] *= 3. # total to single neutrino mass
                #lssfisher = lssfisher.rename(index=str, columns={
                #    "M_ncdm": "m_ncdm"})
            
            self.pandas_lss_covariance.to_csv(                          
                os.path.join(path, "inv_lssfisher.mat"),                          
                sep="\t",                                               
                index=False,                                            
                header=self.fisher_order)

            if hasattr(self, "pandas_cmb_fisher"): 

                size = len(self.fisher_order)
                fullfisher = np.zeros((size, size))

                tau_idx = self.fisher_order.index('tau_reio')     
                self.pandas_lss_fisher.iloc[:, tau_idx] = 0. 
                self.pandas_lss_fisher.iloc[tau_idx, :] = 0. 
                self.numpy_lss_fisher[:, tau_idx] = 0. 
                self.numpy_lss_fisher[tau_idx, :] = 0. 
        
                for pidx1, pval1 in enumerate(self.fisher_order): 
                    for pidx2, pval2 in enumerate(self.fisher_order):
                        if ((pval1 in self.pandas_cmb_fisher.columns) and 
                            (pval2 in self.pandas_cmb_fisher.columns)):
                            fullfisher[pidx1, pidx2] = (
                                self.numpy_lss_fisher[pidx1, pidx2]
                                + self.numpy_cmb_fisher[pidx1, pidx2])
                        else: 
                            fullfisher[pidx1, pidx2] = (
                                self.numpy_lss_fisher[pidx1, pidx2])   
                self.pandas_full_fisher = pd.DataFrame(
                    np.array(fullfisher), columns=self.fisher_order) 
                self.numpy_full_fisher = fullfisher

                self.pandas_full_covariance = pd.DataFrame(np.linalg.inv(   
                    self.pandas_full_fisher),  
                    columns = self.fisher_order)                                
                self.numpy_full_covariance = np.linalg.inv(                 
                    self.numpy_full_fisher)
   
                self.pandas_full_covariance.to_csv(                         
                    os.path.join(path, "inv_fullfisher.mat"),                         
                    sep="\t",                                               
                    index=False,                                            
                    header=self.fisher_order)                                        
                self.pandas_cmb_covariance.to_csv(                          
                    os.path.join(path, "inv_cmbfisher.mat"),                          
                    sep="\t",                                               
                    index=False,                                            
                    header=self.pandas_cmb_fisher.columns)
