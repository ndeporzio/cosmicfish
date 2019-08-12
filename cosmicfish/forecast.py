import numpy as np
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
                 dstep=0.03): 
        self.classdir = classdir                                                
        self.datastore = datastore
        self.forecast = forecast_type
        self.fid = fiducialcosmo
        self.z_steps = z_steps
        self.dNdz = dNdz
        self.dstep = dstep

        self.A_s_fid = self.fid['A_s']
        self.n_s_fid = self.fid['n_s']
        self.omega_b_fid = self.fid['omega_b']
        self.omega_cdm_fid = self.fid['omega_cdm']
        self.h_fid = self.fid['h']
        self.tau_reio_fid = self.fid['tau_reio']
        self.m_ncdm_fid = self.fid['m_ncdm'] # Unit [eV]                   
        self.M_ncdm_fid = 3. * self.fid['m_ncdm'] # Unit [eV] i

        self.sigma_fog_0_fid = cf.FID_SIGMA_FOG_0
        self.b1L_fid = cf.FID_B1L
        self.alphak2_fid = cf.FID_ALPHAK2 

        self.n_densities = np.zeros(len(dNdz))

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

        if (fsky is None) and (fcoverage_deg is not None):
            self.fcoverage_deg = fcoverage_deg
            self.fsky = fcoverage_deg / cf.FULL_SKY_DEGREES
            # ^^^ http://www.badastronomy.com/bitesize/bigsky.html
        elif (fsky is not None) and (fcoverage_deg is None):
            self.fcoverage_deg =  cf.FULL_SKY_DEGREES * fsky
            self.fsky = fsky
        elif (fsky is not None) and (fcoverage_deg is not None):
            print("Both f_sky and sky coverage specified,"
                  + " using value for f_sky.")
            self.fcoverage_deg = cf.FULL_SKY_DEGREES * fsky
            self.fsky = fsky
        else:
            print("Assuming full sky survey.")
            self.fsky = 1.
            self.fcoverage_deg = cf.FULL_SKY_DEGREES

        self.psterms = []  


    def gen_pg(self):  
        if 'pg' not in self.psterms: 
            self.psterms.append('pg') 

        analytic_A_s = True
        analytic_n_s = True

        # Generate spectra at each z for fid cosmo
        self.spectra_mid = [cf.spectrum(
                                cf.generate_data(
                                    dict(self.fid,      
                                         **{'z_pk' : j}),
                                    self.classdir,       
                                    self.datastore)[0:-20],
                                self.z_steps,
                                fsky=self.fsky) 
                            for j in self.z_steps]
        self.k_table = self.spectra_mid[0].k_table 
        # ^^^ All spectra should have same k_table

        self.Pg = [self.spectra_mid[zidx].ps_table 
                   for zidx, zval in enumerate(self.z_steps)]

        # Generate variation spectra at each z    
        self.A_s_high, self.A_s_low = self.generate_spectra('A_s') 
        self.n_s_high, self.n_s_low = self.generate_spectra('n_s')
        self.omega_b_high, self.omega_b_low = self.generate_spectra('omega_b')
        self.omega_cdm_high, self.omega_cdm_low = self.generate_spectra(
                                                      'omega_cdm')
        self.h_high, self.h_low = self.generate_spectra('h')
        self.tau_reio_high, self.tau_reio_low = self.generate_spectra(
                                                    'tau_reio')
        if self.forecast=="neutrino": 
            self.M_ncdm_high, self.M_ncdm_low = self.generate_spectra('m_ncdm')
        elif self.forecast=="relic": 
            self.T_ncdm_high, self.T_ncdm_low = self.generate_spectra('T_ncdm')

        # Calculate centered derivatives about fiducial cosmo at each z 

        if analytic_A_s==False: 
            self.dPdA_s, self.dlogPdA_s = cf.dPs_array(self.A_s_low, 
                self.A_s_high, self.fid['A_s']*self.dstep) 
        else: 
            self.dlogPdA_s = [[(1. / self.A_s_fid) 
                for k in self.k_table] for z in self.z_steps] # Analytic form

        if analytic_n_s==False: 
            self.dPdn_s, self.dlogPdn_s = cf.dPs_array(self.n_s_low, 
                self.n_s_high, self.fid['n_s']*self.dstep) 
        else:  
            self.dlogPdn_s = [[np.log( k / self.kp ) 
                for k in self.k_table] for z in self.z_steps] # Analytic form

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
                self.M_ncdm_fid * self.dstep)   
                # ^^^CAUTION. Check factor of 3 in denominator.  
            self.dPdomega_ncdm = [np.array(self.dPdM_ncdm[k]) 
                * cf.NEUTRINO_SCALE_FACTOR for k in range(len(self.z_steps))]
            self.dlogPdomega_ncdm = [np.array(self.dlogPdM_ncdm[k]) 
                * cf.NEUTRINO_SCALE_FACTOR for k in range(len(self.z_steps))]
        elif self.forecast=="relic": 
            self.dPdT_ncdm, self.dlogPdT_ncdm = cf.dPs_array(
                self.T_ncdm_low, 
                self.T_ncdm_high, 
                self.fid['T_ncdm']*self.dstep)
            self.dPdomega_ncdm = [np.array(self.dPdT_ncdm[k]) 
                / domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.m_ncdm_fid)
                for k in range(len(self.z_steps))]
            self.dlogPdomega_ncdm = [np.array(self.dlogPdT_ncdm[k]) 
                / domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.m_ncdm_fid)
                for k in range(len(self.z_steps))]
 
    def gen_rsd(self, mu): 
        '''Given mu, creates len(z_steps) array. Each elem is len(k_table).'''
        if 'rsd' not in self.psterms: 
            self.psterms.append('rsd') 

        fiducial = {'omega_b' : self.omega_b_fid, 
                    'omega_cdm' : self.omega_cdm_fid, 
                    'omega_ncdm' : self.omega_ncdm_fid,
                    'h' : self.h_fid, 
                    'b1L' : cf.FID_B1L,
                    'alphak2' : cf.FID_ALPHAK2,
                    'mu' : mu}

        kfs_table = cf.kfs(self.omega_ncdm_fid, self.h_fid, 
            np.array(self.z_steps))

        self.RSD = [[cf.rsd(**dict(fiducial, **{'k_fs' : kfs_table[zidx], 
            'z' : zval, 'k' : kval})) 
            for kval in self.k_table] 
            for zidx, zval in enumerate(self.z_steps)]   

        self.dlogRSDdomega_b = [[cf.derivative(
            cf.log_rsd, 
            'omega_b', 
            self.dstep,
            **dict(fiducial, **{'k_fs' : kfs_table[zidx], 'z' : zval, 
                'k' :  kval}))
            for kval in self.k_table] 
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogRSDdomega_cdm = [[cf.derivative(
            cf.log_rsd, 
            'omega_cdm', 
            self.dstep,  
            **dict(fiducial, **{'k_fs' : kfs_table[zidx], 'z' : zval, 
                'k' :  kval})) 
            for kval in self.k_table]                                           
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogRSDdomega_ncdm = [[cf.derivative(                                  
            cf.log_rsd,                                                         
            'omega_ncdm',                                                        
            self.dstep,                                                         
            **dict(fiducial, **{'k_fs' : kfs_table[zidx], 'z' : zval, 
                'k' :  kval})) 
            for kval in self.k_table]                                           
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogRSDdM_ncdm = (np.array(self.dlogRSDdomega_ncdm) 
            / cf.NEUTRINO_SCALE_FACTOR)

        self.dlogRSDdh = [[cf.derivative(                                 
            cf.log_rsd,                                                         
            'h',                                                       
            self.dstep,                                                         
            **dict(fiducial, **{'k_fs' : kfs_table[zidx], 'z' : zval, 
                'k' :  kval})) 
            for kval in self.k_table]                                           
            for zidx, zval in enumerate(self.z_steps)]   

        self.dlogRSDdb1L = [[cf.derivative(                                       
            cf.log_rsd,                                                         
            'b1L',                                                                
            self.dstep,                                                         
            **dict(fiducial, **{'k_fs' : kfs_table[zidx], 'z' : zval,           
                'k' :  kval}))                                                  
            for kval in self.k_table]                                           
            for zidx, zval in enumerate(self.z_steps)]

        self.dlogRSDdalphak2 = [[cf.derivative(                                       
            cf.log_rsd,                                                         
            'alphak2',                                                                
            self.dstep,                                                         
            **dict(fiducial, **{'k_fs' : kfs_table[zidx], 'z' : zval,           
                'k' :  kval}))                                                  
            for kval in self.k_table]                                           
            for zidx, zval in enumerate(self.z_steps)]  

    def gen_fog(self, mu):

        if 'fog' not in self.psterms:                                           
            self.psterms.append('fog')

        fiducial = {'omega_b' : self.omega_b_fid,                               
                    'omega_cdm' : self.omega_cdm_fid,                           
                    'omega_ncdm' : self.omega_ncdm_fid,                         
                    'h' : self.h_fid,                                           
                    'mu' : mu,
                    'sigma_fog_0' : cf.FID_SIGMA_FOG_0}
            

        self.FOG = [[cf.fog(**dict(fiducial, **{'z' : zval, 'k' :  kval})) 
            for kval in self.k_table] for zval in self.z_steps]


        self.dlogFOGdh = [[cf.derivative(cf.log_fog, 'h', self.dstep, 
            **dict(fiducial, **{'z' : zval, 'k' :  kval})) 
            for kval in self.k_table] 
            for zval in self.z_steps]

        self.dlogFOGdomega_b = [[cf.derivative(cf.log_fog, 'omega_b', 
            self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))                       
            for kval in self.k_table]                                           
            for zval in self.z_steps]

        self.dlogFOGdomega_cdm = [[cf.derivative(cf.log_fog, 'omega_cdm',           
            self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))           
            for kval in self.k_table]                                           
            for zval in self.z_steps]

        self.dlogFOGdomega_ncdm = [[cf.derivative(cf.log_fog, 'omega_ncdm',           
            self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))           
            for kval in self.k_table]                                           
            for zval in self.z_steps]

        self.dlogFOGdM_ncdm = (np.array(self.dlogFOGdomega_ncdm) 
            / cf.NEUTRINO_SCALE_FACTOR) 

        self.dlogFOGdsigmafog0 = [[cf.derivative(cf.log_fog, 'sigma_fog_0',     
            self.dstep, **dict(fiducial, **{'z' : zval, 'k' :  kval}))           
            for kval in self.k_table]                                           
            for zval in self.z_steps]

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
            for kval in self.k_table] for zval in self.z_steps]

        self.dlogAPdomega_b = [[cf.derivative(cf.log_ap, 'omega_b', self.dstep,             
            **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))                    
            for kval in self.k_table] for zval in self.z_steps] 

        self.dlogAPdomega_cdm = [[cf.derivative(cf.log_ap, 'omega_cdm', 
            self.dstep, **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))                    
            for kval in self.k_table] for zval in self.z_steps]

        self.dlogAPdomega_ncdm = [[cf.derivative(cf.log_ap, 'omega_ncdm', 
            self.dstep, **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))                    
            for kval in self.k_table] for zval in self.z_steps]

        self.dlogAPdh = [[cf.derivative(cf.log_ap, 'h', self.dstep, 
            **dict(fiducial, **{'z' : zval, 'z_fid' : zval}))
            for kval in self.k_table] for zval in self.z_steps]

        self.dlogAPdM_ncdm = (np.array(self.dlogAPdomega_ncdm) 
            / cf.NEUTRINO_SCALE_FACTOR) 
                            
    def gen_cov(self, mu):

        # CAUTION! You've written physics here. By your convention, all physics
        # should be in the equations.py module.  

        if 'cov' not in self.psterms: 
            self.psterms.append('cov') 

        self.COV = [[cf.cov() 
            for kval in self.k_table] for zval in self.z_steps]

        fiducial = {
            'omega_b' : self.omega_b_fid, 
            'omega_cdm' : self.omega_cdm_fid, 
            'omega_ncdm' : self.omega_ncdm_fid, 
            'h' : self.h_fid}

        dHdomegab = [[cf.derivative(cf.H, 'omega_b', self.dstep,             
            **dict(fiducial, **{'z' : zval}))                    
            for kval in self.k_table] for zval in self.z_steps] 
 
        dHdomegacdm = [[cf.derivative(cf.H, 'omega_cdm', self.dstep,                
            **dict(fiducial, **{'z' : zval}))                                    
            for kval in self.k_table] for zval in self.z_steps]

        dHdomegancdm = [[cf.derivative(cf.H, 'omega_ncdm', self.dstep,                
            **dict(fiducial, **{'z' : zval}))                                    
            for kval in self.k_table] for zval in self.z_steps]

        dHdh = [[cf.derivative(cf.H, 'h', self.dstep,                
            **dict(fiducial, **{'z' : zval}))                                    
            for kval in self.k_table] for zval in self.z_steps]

        dDadomegab = [[cf.derivative(cf.Da, 'omega_b', self.dstep,                
            **dict(fiducial, **{'z' : zval}))                                    
            for kval in self.k_table] for zval in self.z_steps]                 
                                                                                
        dDadomegacdm = [[cf.derivative(cf.Da, 'omega_cdm', self.dstep,            
            **dict(fiducial, **{'z' : zval}))                                    
            for kval in self.k_table] for zval in self.z_steps]                 
                                                                                
        dDadomegancdm = [[cf.derivative(cf.Da, 'omega_ncdm', self.dstep,          
            **dict(fiducial, **{'z' : zval}))                                   
            for kval in self.k_table] for zval in self.z_steps]                 
                                                                                
        dDadh = [[cf.derivative(cf.Da, 'h', self.dstep,                           
            **dict(fiducial, **{'z' : zval}))                                    
            for kval in self.k_table] for zval in self.z_steps]  

        fiducial = {
            'omega_b' : self.omega_b_fid,                                       
            'omega_cdm' : self.omega_cdm_fid,                                   
            'omega_ncdm' : self.omega_ncdm_fid,                                 
            'h' : self.h_fid, 
            'mu' : mu}

        dkdH = [[cf.cov_dkdH(**dict(fiducial, **{'z' : zval, 'k' : kval}))                                    
            for kval in self.k_table] for zval in self.z_steps] 

        dkdDa = [[cf.cov_dkdDa(**dict(fiducial, **{'z' : zval, 'k' : kval}))                        
            for kval in self.k_table] for zval in self.z_steps]

        dkdomegab = [[(
            dkdH[zidx][kidx] * dHdomegab[zidx][kidx]
            + dkdDa[zidx][kidx] * dDadomegab[zidx][kidx])
            for kidx, kval in enumerate(self.k_table)] 
            for zidx, zval in enumerate(self.z_steps)]

        dkdomegacdm = [[(                                                         
            dkdH[zidx][kidx] * dHdomegacdm[zidx][kidx]                            
            + dkdDa[zidx][kidx] * dDadomegacdm[zidx][kidx])                       
            for kidx, kval in enumerate(self.k_table)]                          
            for zidx, zval in enumerate(self.z_steps)] 

        dkdomegancdm = [[(                                                         
            dkdH[zidx][kidx] * dHdomegancdm[zidx][kidx]                            
            + dkdDa[zidx][kidx] * dDadomegancdm[zidx][kidx])                       
            for kidx, kval in enumerate(self.k_table)]                          
            for zidx, zval in enumerate(self.z_steps)]

        dkdh = [[(                                                         
            dkdH[zidx][kidx] * dHdh[zidx][kidx]                            
            + dkdDa[zidx][kidx] * dDadh[zidx][kidx])                       
            for kidx, kval in enumerate(self.k_table)]                          
            for zidx, zval in enumerate(self.z_steps)]

        H_fid = [cf.H(self.omega_b_fid, self.omega_cdm_fid, 
            self.omega_ncdm_fid, self.h_fid, zval) for zval in self.z_steps]

        dmudomegab = [[(
            (mu / kval) * dkdomegab[zidx][kidx] 
            + (mu / H_fid[zidx]) * dHdomegab[zidx][kidx])
            for kidx, kval in enumerate(self.k_table)]
            for zidx, zval in enumerate(self.z_steps)]

        dmudomegacdm = [[(                                                        
            (mu / kval) * dkdomegacdm[zidx][kidx]                                             
            + (mu / H_fid[zidx]) * dHdomegacdm[zidx][kidx])                                   
            for kidx, kval in enumerate(self.k_table)]                          
            for zidx, zval in enumerate(self.z_steps)]

        dmudomegancdm = [[(                                                        
            (mu / kval) * dkdomegancdm[zidx][kidx]                                             
            + (mu / H_fid[zidx]) * dHdomegancdm[zidx][kidx])                                   
            for kidx, kval in enumerate(self.k_table)]                          
            for zidx, zval in enumerate(self.z_steps)]

        dmudh = [[(                                                        
            (mu / kval) * dkdh[zidx][kidx]                                             
            + (mu / H_fid[zidx]) * dHdh[zidx][kidx])                                   
            for kidx, kval in enumerate(self.k_table)]                          
            for zidx, zval in enumerate(self.z_steps)]

        # Fix everything below: very redundant, slow, error prone
        # dlogPdmu
        self.gen_rsd((1. + self.dstep) * mu)
        self.gen_fog((1. + self.dstep) * mu)
        logP_mu_high = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG) 
 
        self.gen_rsd((1. - self.dstep) * mu)
        self.gen_fog((1. - self.dstep) * mu)
        logP_mu_low = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG)
        
        self.gen_rsd(mu)
        self.gen_fog(mu)
        logP_mid = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG)

        dlogPdmu = (logP_mu_high - logP_mu_low) / (2. * self.dstep * mu) 

        # dlogPdk
        dlogPdk = np.zeros((len(self.z_steps), len(self.k_table)))
        
        for zidx, zval in enumerate(self.z_steps): 
            for kidx, kval in enumerate(self.k_table[1:-1]):
                # Careful with this derivative definition, uneven spacing
                dlogPdk[zidx][kidx+1] = ((logP_mid[zidx][kidx+2] 
                                          - logP_mid[zidx][kidx])
                                         / (self.k_table[kidx+2]
                                            - self.k_table[kidx]))

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
        self.dlogCOVdh = (dlogPdk * dkdh 
            + dlogPdmu * dmudh)

        self.dlogCOVdM_ncdm = (np.array(self.dlogCOVdomega_ncdm) 
            / cf.NEUTRINO_SCALE_FACTOR)
    
    def gen_fisher(self, mu_step): # Really messy and inefficient

        mu_vals = np.arange(-1., 1., mu_step)
        Pm = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdA_s = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdn_s = np.zeros(   
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdomega_b = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdomega_cdm = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdh = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdtau_reio = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdomega_ncdm = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdsigmafog = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdb1L = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdalphak2 = np.zeros(
            (len(self.z_steps), len(self.k_table), len(mu_vals)))        

        for muidx, muval in enumerate(mu_vals): 
            self.gen_rsd(muval)
            self.gen_fog(muval)
            self.gen_ap()
            self.gen_cov(muval)

            for zidx, zval in enumerate(self.z_steps): 

                for kidx, kval in enumerate(self.k_table):
                    Pm[zidx][kidx][muidx] = ( 1.
                        * self.spectra_mid[zidx].ps_table[kidx] 
                        * self.RSD[zidx][kidx] 
                        * self.FOG[zidx][kidx]
                        * self.AP[zidx][kidx]
                        #* self.COV[zidx][kidx]<---STILL WRONG 
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
                    dlogPdM_ncdm = dlogPdomega_ncdm / cf.NEUTRINO_SCALE_FACTOR 
                    # ^^^Careful, this overwrites the earlier dP_g value. 
                    # we do this to reflect corrections in the contours
                    # for M_ncdm

                    dlogPdsigmafog[zidx][kidx][muidx] = (                       
                        self.dlogFOGdsigmafog0[zidx][kidx]
                        )

                    dlogPdb1L[zidx][kidx][muidx] = (
                        self.dlogRSDdb1L[zidx][kidx]    
                        )

                    dlogPdalphak2[zidx][kidx][muidx] = (                          
                        self.dlogRSDdalphak2[zidx][kidx]                                                    
                        )
        self.Pm = Pm 

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
                integrand = np.zeros((len(self.z_steps), len(self.k_table)))
                integral = np.zeros(len(self.z_steps))
                for zidx, zval in enumerate(self.z_steps):
                    Volume = self.V(zidx)
                    # print("For z = ", zval, ", V_eff = ", 
                    #       (Volume*self.h_fid)/(1.e9), " [h^{-1} Gpc^{3}]") 
                    for kidx, kval in enumerate(self.k_table): # Center intgrl? 
                        integrand[zidx][kidx] = np.sum(
                            mu_step *2. *np.pi
                            * paramvec[pidx1][zidx][kidx]
                            * paramvec[pidx2][zidx][kidx]
                            * np.power(self.k_table[kidx]/(2.*np.pi), 3.)
                            * np.power((self.n_densities[zidx] 
                                        * self.Pg[zidx][kidx])
                                       / (self.n_densities[zidx] 
                                          * self.Pg[zidx][kidx] + 1.), 
                                       2.) 
                            * Volume
                            * (1. / self.k_table[kidx]))
                            
                for zidx, zval in enumerate(self.z_steps): 
                    val = 0 
                    for kidx, kval in enumerate(self.k_table[:-1]): # Center? 
                        val += (((integrand[zidx][kidx] 
                                  + integrand[zidx][kidx+1])
                                 / 2.)  
                                * (self.k_table[kidx+1]-self.k_table[kidx]))
                    integral[zidx] = val  
                fisher[pidx1][pidx2] = np.sum(integral)
                print("Fisher element (", pidx1, ", ", pidx2,") calculated...") 
        self.fisher=fisher

    def generate_spectra(self, param):                                          
        spectra_high = [cf.spectrum(                                            
                            cf.generate_data(                                   
                                dict(self.fid,                                  
                                     **{'z_pk' : j,                             
                                     param : (1.+self.dstep)*self.fid[param]}), 
                                self.classdir,                                  
                                self.datastore)[0:-20],                         
                                self.z_steps,
                                self.fsky)                                   
                        for j in self.z_steps]                                  
        spectra_low = [cf.spectrum(                                             
                           cf.generate_data(                                    
                               dict(self.fid,                                   
                                    **{'z_pk' : j,                              
                                    param : (1.-self.dstep)*self.fid[param]}),  
                               self.classdir,                                   
                               self.datastore)[0:-20],                          
                           self.z_steps,
                           self.fsky)                                        
                       for j in self.z_steps]                                   
        return spectra_high, spectra_low 

    def V(self, zidx):                                                          
        if zidx==(len(self.z_steps)-1):                                         
            zidx -=1                                                            
        omega_m = self.omega_b_fid + self.omega_cdm_fid + self.omega_ncdm_fid   
        omega_lambda = np.power(self.h_fid, 2.) - omega_m                       
        zmax = self.z_steps[zidx+1]                                             
        zmin = self.z_steps[zidx]                                               
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
                                                                                
    def print_v_table(self):                                                    
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
                   ("b_ELG = {7:.6f} \n\t")).format(                            
                       zval,                                                    
                       (V*np.power(self.h_fid, 3.)) / (1.e9),                   
                       n2,                                                      
                       n,                                                       
                       n2 / self.fcoverage_deg,                                 
                       n / self.fcoverage_deg,                                  
                       self.spectra_mid[zidx].D_table[zidx],                    
                       cf.DB_ELG / self.spectra_mid[zidx].D_table[zidx]))            
        return
                
    def print_P_table(self): 
        for zidx, zval in enumerate(self.z_steps): 
            print((("For z = {0:.2f},\t") + 
                   (" P(0.2h, 0) = {1:.2f}\n")).format(zval, 
                                                        self.Pm[zidx][69][20]))

