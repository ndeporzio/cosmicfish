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
        self.M_ncdm_fid = 3. * self.fid['m_ncdm'] # Unit [eV] i

        self.sigma_fog_0_fid = self.fid['sigma_fog_0']
        self.b1L_fid = self.fid['b1L']
        self.alphak2_fid = self.fid['alphak2'] 

        self.n_densities = np.zeros(len(dNdz))
        self.pandas_cmb_fisher = None

        if self.forecast=="relic": 
            self.T_ncdm_fid = self.fid['T_ncdm'] # Units [T_cmb]
            self.omega_ncdm_fid = cf.omega_ncdm(self.T_ncdm_fid, 
                                                self.m_ncdm_fid, 
                                                "relic")
        elif self.forecast=="neutrino": 
            self.omega_ncdm_fid = cf.omega_ncdm(None, 
                                                self.m_ncdm_fid, 
                                                "neutrino")
        self.kp = cf.KP_PREFACTOR * self.h_fid # Units [Mpc^-1]

        self.fsky, self.fcoverage_deg = cf.set_sky_cover(fsky,  fcoverage_deg)
            
        self.psterms = [] 

        # Generate tables
        self.k_table = [0] * len(self.z_steps)
        for i in range(len(self.z_steps)):
            V = self.V(i)  
            self.k_table[i] = cf.gen_k_table(
                volume=V, 
                z=self.z_steps[i],
                h=self.h_fid, 
                n_s=self.n_s_fid,
                k_steps=100)           


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
                                k_table=self.k_table[zidx]) 
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
            # Carefule with naming - you're varying m_ncdm but 
            # calling it M_ncdm  
            self.M_ncdm_high, self.M_ncdm_low = self.generate_spectra('m_ncdm')
        elif self.forecast=="relic": 
            self.T_ncdm_high, self.T_ncdm_low = self.generate_spectra('T_ncdm')

        # Calculate centered derivatives about fiducial cosmo at each z 

        if cf.ANALYTIC_A_S==False: 
            self.A_s_high, self.A_s_low = self.generate_spectra('A_s') 
            self.dPdA_s, self.dlogPdA_s = cf.dPs_array(self.A_s_low, 
                self.A_s_high, self.fid['A_s']*self.dstep) 
        else: 
            self.dlogPdA_s = [[(1. / self.A_s_fid) 
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)] # Analytic form

        if cf.ANALYTIC_N_S==False: 
            self.n_s_high, self.n_s_low = self.generate_spectra('n_s')
            self.dPdn_s, self.dlogPdn_s = cf.dPs_array(self.n_s_low, 
                self.n_s_high, self.fid['n_s']*self.dstep) 
        else:  
            self.dlogPdn_s = [[np.log( kval / self.kp ) 
                for kidx, kval in enumerate(self.k_table[zidx])] 
                for zidx, zval in enumerate(self.z_steps)] # Analytic form

        self.dPdomega_b, self.dlogPdomega_b = cf.dPs_array(
            self.omega_b_low, 
            self.omega_b_high, 
            self.fid['omega_b'] * self.dstep) 
        self.dPdomega_cdm, self.dlogPdomega_cdm = cf.dPs_array(
            self.omega_cdm_low, 
            self.omega_cdm_high, 
            self.fid['omega_cdm']*self.dstep)
        self.dPdh, self.dlogPdh = cf.dPs_array(
            self.h_low, 
            self.h_high, 
            self.fid['h']*self.dstep)
        self.dPdtau_reio, self.dlogPdtau_reio = cf.dPs_array(
            self.tau_reio_low, 
            self.tau_reio_high, 
            self.fid['tau_reio']*self.dstep)

        if self.forecast=="neutrino": 
            self.dPdM_ncdm, self.dlogPdM_ncdm = cf.dPs_array(
                self.M_ncdm_low, 
                self.M_ncdm_high, 
                self.m_ncdm_fid * self.dstep)
            self.dPdM_ncdm *= (1./3.) 
            self.dlogPdM_ncdm *= (1./3.)    
            # ^^^CAUTION. Carefule of factor of 3. .  
            self.dPdomega_ncdm = [
                np.array(self.dPdM_ncdm[zidx]) * cf.NEUTRINO_SCALE_FACTOR 
                for zidx, zval in enumerate(self.z_steps)]
            self.dlogPdomega_ncdm = [
                np.array(self.dlogPdM_ncdm[zidx]) * cf.NEUTRINO_SCALE_FACTOR 
                for zidx, zval in enumerate(self.z_steps)]
        elif self.forecast=="relic": 
            self.dPdT_ncdm, self.dlogPdT_ncdm = cf.dPs_array(
                self.T_ncdm_low, 
                self.T_ncdm_high, 
                self.fid['T_ncdm']*self.dstep)
            self.dPdomega_ncdm = [
                np.array(self.dPdT_ncdm[zidx]) 
                / domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.m_ncdm_fid)
                for zidx in enumerate(self.z_steps)]
            self.dlogPdomega_ncdm = [
                np.array(self.dlogPdT_ncdm[zidx]) 
                / domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.m_ncdm_fid)
                for zidx, zval in enumerate(self.z_steps)]
 
    def gen_rsd(self, mu): 
        '''Given mu, creates len(z_steps) array. Each elem is len(k_table).'''
        if 'rsd' not in self.psterms: 
            self.psterms.append('rsd') 

        fiducial = {'omega_b' : self.omega_b_fid, 
                    'omega_cdm' : self.omega_cdm_fid, 
                    'omega_ncdm' : self.omega_ncdm_fid,
                    'h' : self.h_fid, 
                    'b1L' : self.b1L_fid,
                    'alphak2' : self.alphak2_fid,
                    'mu' : mu}

        self.RSD = [[cf.rsd(**dict(fiducial, **{'z' : zval, 'k' : kval})) 
            for kidx, kval in enumerate(self.k_table[zidx])] 
            for zidx, zval in enumerate(self.z_steps)]   

        self.dlogRSDdomega_b = [[cf.derivative(
            cf.log_rsd, 
            'omega_b', 
            self.dstep,
            **dict(fiducial, **{'z' : zval, 'k' :  kval}))
            for kidx, kval in enumerate(self.k_table[zidx])] 
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogRSDdomega_cdm = [[cf.derivative(
            cf.log_rsd, 
            'omega_cdm', 
            self.dstep,  
            **dict(fiducial, **{'z' : zval, 'k' :  kval})) 
            for kidx, kval in enumerate(self.k_table[zidx])]                                           
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogRSDdomega_ncdm = [[cf.derivative(                                  
            cf.log_rsd,                                                         
            'omega_ncdm',                                                        
            self.dstep,                                                         
            **dict(fiducial, **{'z' : zval, 'k' :  kval})) 
            for kidx, kval in enumerate(self.k_table[zidx])]                                           
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogRSDdM_ncdm = (np.array(self.dlogRSDdomega_ncdm) 
            / cf.NEUTRINO_SCALE_FACTOR)

        self.dlogRSDdh = [[cf.derivative(                                 
            cf.log_rsd,                                                         
            'h',                                                       
            self.dstep,                                                         
            **dict(fiducial, **{'z' : zval, 'k' :  kval})) 
            for kidx, kval in enumerate(self.k_table[zidx])]                                           
            for zidx, zval in enumerate(self.z_steps)]   

        self.dlogRSDdb1L = [[cf.derivative(                                       
            cf.log_rsd,                                                         
            'b1L',                                                                
            self.dstep,                                                         
            **dict(fiducial, **{'z' : zval, 'k' :  kval}))                                                  
            for kidx, kval in enumerate(self.k_table[zidx])]                                           
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogRSDdalphak2 = [[cf.derivative(                                       
            cf.log_rsd,                                                         
            'alphak2',                                                                
            self.dstep,                                                         
            **dict(fiducial, **{'z' : zval, 'k' :  kval}))                                                  
            for kidx, kval in enumerate(self.k_table[zidx])]                                           
            for zidx, zval in enumerate(self.z_steps)]  

    def gen_fog(self, mu):

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

        self.dlogFOGdM_ncdm = (np.array(self.dlogFOGdomega_ncdm) 
            / cf.NEUTRINO_SCALE_FACTOR) 

        self.dlogFOGdh = [[cf.derivative(cf.log_fog, 'h', self.dstep,           
            **dict(fiducial, **{'z' : zval, 'k' :  kval}))                      
            for kidx, kval in enumerate(self.k_table[zidx])]                                           
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogFOGdsigmafog0 = [[cf.derivative(cf.log_fog, 'sigma_fog_0',     
            self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))           
            for kidx, kval in enumerate(self.k_table[zidx])]                                           
            for zidx, zval in enumerate(self.z_steps)]

    def gen_ap(self):

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

        self.dlogAPdM_ncdm = (np.array(self.dlogAPdomega_ncdm) 
            / cf.NEUTRINO_SCALE_FACTOR) 

        self.dlogAPdh = [[cf.derivative(cf.log_ap, 'h', self.dstep,             
            **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))                   
            for kidx, kval in enumerate(self.k_table[zidx])] 
            for zidx, zval in enumerate(self.z_steps)]
                            
    def gen_cov(self, mu):

        # CAUTION! You've written physics here. By your convention, all physics
        # should be in the equations.py module.  

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
        self.dlogCOVdM_ncdm = (np.array(self.dlogCOVdomega_ncdm)                
            / cf.NEUTRINO_SCALE_FACTOR) 
        self.dlogCOVdh = (dlogPdk * dkdh 
            + dlogPdmu * dmudh)
    
    def gen_fisher(self, mu_step=0.05): # Really messy and inefficient

        if 'pm' not in self.psterms:                                        
            self.gen_pm() 

        mu_vals = np.arange(-1., 1., mu_step)
        Pg = np.zeros(
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
        dlogPdsigmafog = np.zeros(
            (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
        dlogPdb1L = np.zeros(
            (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))
        dlogPdalphak2 = np.zeros(
            (len(self.z_steps), len(self.k_table[0]), len(mu_vals)))        

        for muidx, muval in enumerate(mu_vals):
            if self.use_rsd==True:  
                self.gen_rsd(muval)
            else:  
                self.RSD = [[1. 
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
                self.dlogRSDdM_ncdm = [[0.                     
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]              
                self.dlogRSDdh = [[0.                          
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]         
                self.dlogRSDdb1L = [[0.                        
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]           
                self.dlogRSDdalphak2 = [[0.                    
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]               

            if self.use_fog==True:  
                self.gen_fog(muval)
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
                self.dlogFOGdM_ncdm = [[0.                     
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                  
                self.dlogFOGdh = [[0.                          
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                  
                self.dlogFOGdsigmafog0 = [[0.                  
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                    

            if self.use_ap==True: 
                self.gen_ap()
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
                self.dlogAPdM_ncdm = [[0.                      
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                 
                self.dlogAPdh = [[0.                           
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                 

            if self.use_cov==True: 
                self.gen_cov(muval)
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
                self.dlogCOVdM_ncdm = [[0.                     
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                   
                self.dlogCOVdh = [[0.                          
                    for kidx, kval in enumerate(self.k_table[zidx])]            
                    for zidx, zval in enumerate(self.z_steps)]                   

            for zidx, zval in enumerate(self.z_steps): 
                for kidx, kval in enumerate(self.k_table[zidx]):

                    Pg[zidx][kidx][muidx] = ( 1.
                        * self.spectra_mid[zidx].ps_table[kidx] 
                        * self.RSD[zidx][kidx] 
                        * self.FOG[zidx][kidx]
                        * self.AP[zidx][kidx]
                        * self.COV[zidx][kidx] #Just equals 1
                        )
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

                    dlogPdM_ncdm[zidx][kidx][muidx] = (
                        dlogPdomega_ncdm[zidx][kidx][muidx] 
                        / cf.NEUTRINO_SCALE_FACTOR) 
                    # ^^^Careful, this overwrites the earlier dP_g value. 

                    dlogPdsigmafog[zidx][kidx][muidx] = (                       
                        self.dlogFOGdsigmafog0[zidx][kidx]
                        )

                    dlogPdb1L[zidx][kidx][muidx] = (
                        self.dlogRSDdb1L[zidx][kidx]    
                        )

                    dlogPdalphak2[zidx][kidx][muidx] = (                          
                        self.dlogRSDdalphak2[zidx][kidx]                                                    
                        )
        self.Pg = Pg
        self.dlogPdA_s = dlogPdA_s
        self.dlogPdn_s = dlogPdn_s
        self.dlogPdomega_b  = dlogPdomega_b
        self.dlogPdomega_cdm = dlogPdomega_cdm
        self.dlogPdh = dlogPdh
        self.dlogPdtau_reio = dlogPdtau_reio
        self.dlogPdomega_ncdm = dlogPdomega_ncdm
        self.dlogPdM_ncdm = dlogPdM_ncdm
        self.dlogPdsigmafog = dlogPdsigmafog
        self.dlogPdb1L = dlogPdb1L
        self.dlogPdalphak2 = dlogPdalphak2       

        if self.forecast=="neutrino": 
            paramvec = [dlogPdomega_b, 
                        dlogPdomega_cdm,  
                        dlogPdn_s, 
                        dlogPdA_s,
                        dlogPdtau_reio, 
                        dlogPdh, 
                        dlogPdM_ncdm,
                        dlogPdsigmafog,
                        dlogPdb1L,
                        dlogPdalphak2
                        ]

        elif self.forecast=="relic": 
            paramvec = [dlogPdomega_b, 
                        dlogPdomega_cdm,  
                        dlogPdn_s, 
                        dlogPdA_s,
                        dlogPdtau_reio, 
                        dlogPdh, 
                        dlogPdomega_ncdm, 
                        dlogPdsigmafog,
                        dlogPdb1L, 
                        dlogPdalphak2
                        ] 

        fisher = np.zeros((len(paramvec), len(paramvec))) 

        # Highly inefficient set of loops 
        for pidx1, p1 in enumerate(paramvec): 
            for pidx2, p2 in enumerate(paramvec):
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
                            * paramvec[pidx2][zidx][kidx]
                            * np.power(kval, 2.)
                            * (1. / (8.  * np.power(np.pi, 2)))
                            * np.power((self.n_densities[zidx] 
                                        * self.Pg[zidx][kidx])
                                       / (self.n_densities[zidx] 
                                          * self.Pg[zidx][kidx] + 1.), 
                                       2.) 
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
                fisher[pidx1][pidx2] = np.sum(integral)
                print("Fisher element (", pidx1, ", ", pidx2,") calculated...") 
        self.fisher=fisher

    def generate_spectra(self, param):                                          
        spectra_high = [cf.spectrum(                                            
                            cf.generate_data(                                   
                                dict(self.fid,                                  
                                     **{'z_pk' : zval,                             
                                     param : (1.+self.dstep)*self.fid[param]}), 
                                self.classdir,                                  
                                self.datastore)[0:-20],                         
                                self.fsky,
                                k_table=self.k_table[zidx])                                   
                        for zidx, zval in enumerate(self.z_steps)]                                  
        spectra_low = [cf.spectrum(                                             
                            cf.generate_data(                                    
                                dict(self.fid,                                   
                                    **{'z_pk' : zval,                              
                                    param : (1.-self.dstep)*self.fid[param]}),  
                                self.classdir,                                   
                                self.datastore)[0:-20],                          
                            self.fsky,
                            k_table=self.k_table[zidx])                                        
                       for zidx, zval in enumerate(self.z_steps)]                                   
        return spectra_high, spectra_low 

    def V(self, zidx):                                                          
        delta_z = self.z_steps[-1] - self.z_steps[-2] 
        omega_m = self.omega_b_fid + self.omega_cdm_fid + self.omega_ncdm_fid   
        omega_lambda = np.power(self.h_fid, 2.) - omega_m                       
        zmax = self.z_steps[zidx]+(0.5 *  delta_z)                                              
        zmin = self.z_steps[zidx]-(0.5 *  delta_z)                                                
        zsteps = 100.                                                           
        dz = zmin / zsteps                                                      
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
            print((("For z = {0:.2f},\t") + 
                   (" P(0.2h, 0) = {1:.2f}\n")).format(zval, 
                        self.Pg[zidx][k_index][mu_index]))


    def load_cmb_fisher(self, fisherpath=("~/Desktop/CMBS4_Fisher.dat")): 
        # Parameter ordering: omega_b, omega_cdm, n_s, A_s, tau_reio, H_0, 
        # M_ncdm 

        fmat = pd.read_csv(fisherpath, sep='\t', header=0)
        fmat.iloc[:,5] *= 100. # Change of variables H0->h
        fmat.iloc[5,:] *= 100. # Change of variables H0->h
        fmat.iloc[:,6] *= 3. # Change of variables M->m
        fmat.iloc[6,:] *= 3. # Change of variables M->m
        fmat = fmat.rename(index=str, columns={"H_0": "h", "M_ncdm": "m_ncdm"})

        self.numpy_cmb_fisher =  np.array(fmat)
        self.pandas_cmb_fisher = fmat

        self.numpy_cmb_covariance = np.linalg.inv(self.pandas_cmb_fisher)
        self.pandas_cmb_covariance = pd.DataFrame(np.linalg.inv(                 
            self.pandas_cmb_fisher), columns=self.pandas_cmb_fisher.columns) 

        print("The following CMB Fisher matrix was loaded: ")
        print(self.pandas_cmb_fisher) 

    def export_matrices(self): 
        if self.fisher==[]:  
            print("No LSS Fisher matrix has been generated. Please execute \
                    the `forecast.gen_fisher()` function.`") 
        else:  
            lssfisher = pd.DataFrame(self.fisher,  columns=[
                'omega_b', 
                'omega_cdm', 
                'n_s', 
                'A_s', 
                'tau_reio', 
                'h', 
                'M_ncdm', 
                'sigma_fog', 
                'bLbar', 
                'alpha_k2'])
            lssfisher.iloc[:,7] *= 1e3 #To correct units on sigma_fog
            lssfisher.iloc[7,:] *= 1e3 #To correct units on sigam_fog

            lssfisher.iloc[:,6] *= 3. # total to single neutrino mass
            lssfisher.iloc[6,:] *= 3. # total to single neutrino mass
            lssfisher = lssfisher.rename(index=str, columns={
                "M_ncdm": "m_ncdm"})
            
            self.pandas_lss_fisher = lssfisher
            self.numpy_lss_fisher = np.array(lssfisher)

            nonzeroidx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            nonzeroparams = [              
                'omega_b',                                                      
                'omega_cdm',                                                    
                'n_s',                                                          
                'A_s',                                                          
                'tau_reio',                                                     
                'h',                                                            
                'm_ncdm',                                                       
                'sigma_fog',                                                    
                'bLbar',                                                        
                'alpha_k2']

            if self.use_rsd==False: 
                nonzeroparams.remove('bLbar')
                nonzeroparams.remove('alpha_k2') 
                nonzeroidx.remove(8)
                nonzeroidx.remove(9)

            if self.use_fog==False:
                nonzeroparams.remove('sigma_fog')
                nonzeroidx.remove(7)
    
            self.pandas_lss_covariance = pd.DataFrame(np.linalg.inv(
                lssfisher.iloc[nonzeroidx, nonzeroidx]), 
                columns=nonzeroparams)
            self.numpy_lss_covariance = np.array(np.linalg.inv(
                lssfisher.iloc[nonzeroidx, nonzeroidx]))

        if self.pandas_cmb_fisher is not None: 
            fullfisher = np.zeros((10, 10))
            for i in np.arange(10): 
                for j in np.arange(10):
                    if (i<7) and (j<7): 
                        fullfisher[i, j] = (self.numpy_lss_fisher[i, j] 
                            + self.numpy_cmb_fisher[i, j]) 
                    else:  
                        fullfisher[i, j] = self.numpy_lss_fisher[i, j]

        self.pandas_full_fisher = pd.DataFrame(fullfisher, columns=[              
                'omega_b',                                                      
                'omega_cdm',                                                    
                'n_s',                                                          
                'A_s',                                                          
                'tau_reio',                                                     
                'h',                                                            
                'M_ncdm',                                                       
                'sigma_fog',                                                    
                'bLbar',                                                        
                'alpha_k2'])
        self.numpy_full_fisher = fullfisher   
        
        self.pandas_full_covariance = pd.DataFrame(np.linalg.inv(
            self.pandas_full_fisher.iloc[nonzeroidx, nonzeroidx]), 
            columns = nonzeroparams)
        self.numpy_full_covariance = np.linalg.inv(
            self.numpy_full_fisher[np.ix_(nonzeroidx, nonzeroidx)])

        print("Pandas Fisher Matrices: ")
        print(self.pandas_cmb_fisher)                                            
        print(self.pandas_lss_fisher)                                            
        print(self.pandas_full_fisher)  
                    
        outnames=[
                '# omega_b',                                                    
                'omega_cdm',                                                    
                'n_s',                                                          
                'A_s',                                                          
                'tau_reio',                                                     
                'h',                                                            
                'm_0']
        if self.use_fog==True:
            outnames.append('sigma_fog')
        if self.use_rsd==True:
            outnames.append('bLbar')
            outnames.append('alpha_k2') 

        self.pandas_full_covariance.to_csv(
            "~/Desktop/inv_fullfisher.mat", 
            sep="\t", 
            index=False,
            header=outnames) 
        self.pandas_cmb_covariance.to_csv(                                     
            "~/Desktop/inv_cmbfisher.mat",                                     
            sep="\t",                                                           
            index=False,                                                        
            header=['# omega_b', 'omega_cdm', 'n_s', 'A_s', 'tau_reio', 'h', 
                'm_0'])
        self.pandas_lss_covariance.to_csv(                                     
            "~/Desktop/inv_lssfisher.mat",                                     
            sep="\t",                                                           
            index=False,                                                        
            header=outnames)

# Take care of singular  matrix  when you   turn  of RSD, FOG

 












