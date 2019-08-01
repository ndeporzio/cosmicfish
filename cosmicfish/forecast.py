import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import quad 

import cosmicfish as cf 

class forecast: 
    '''Forecasts Fisher/Covariance matrices from fid cosmo.'''
    
    def __init__(self, 
                 fiducialcosmo, 
                 z_steps, 
                 dNdz, 
                 dstep, 
                 classdir, 
                 datastore,
                 forecast_type,  
                 fsky=None, 
                 fcoverage_deg=None): 
        self.forecast = forecast_type
        self.fid = fiducialcosmo
        self.z_steps = z_steps
        self.n_densities = np.zeros(len(dNdz))
        self.dNdz = dNdz
        self.A_s_fid = fiducialcosmo['A_s']
        self.n_s_fid = fiducialcosmo['n_s']
        self.omega_b_fid = fiducialcosmo['omega_b']
        self.omega_cdm_fid = fiducialcosmo['omega_cdm']
        self.h_fid = fiducialcosmo['h']
        self.tau_reio_fid = fiducialcosmo['tau_reio']
        if forecast_type=="relic": 
            self.T_ncdm_fid = fiducialcosmo['T_ncdm'] # Units [T_cmb]
        self.m_ncdm_fid = fiducialcosmo['m_ncdm'] # Unit [eV] 
        self.M_ncdm_fid = 3.*fiducialcosmo['m_ncdm'] # Unit [eV] 
        if forecast_type=="relic": 
            self.omega_ncdm_fid = omega_ncdm(self.T_ncdm_fid, 
                                             self.m_ncdm_fid, 
                                             "relic")
        elif forecast_type=="neutrino": 
            self.omega_ncdm_fid = omega_ncdm(None, 
                                             self.M_ncdm_fid / 3., 
                                             "neutrino")
        self.dstep = dstep 
        self.classdir = classdir
        self.datastore = datastore
        self.forecast_type = forecast_type
        self.c = 2.9979e8 
        self.kp = 0.05 * self.h_fid # Units [Mpc^-1]

        if (fsky is None) and (fcoverage_deg is not None):
            self.fcoverage_deg = fcoverage_deg
            self.fsky = fcoverage_deg / 41253. 
            # ^^^ http://www.badastronomy.com/bitesize/bigsky.html
        elif (fsky is not None) and (fcoverage_deg is None):
            self.fcoverage_deg =  41253. * fsky
            self.fsky = fsky
        elif (fsky is not None) and (fcoverage_deg is not None):
            print("Both f_sky and sky coverage specified,"
                  + " using value for f_sky.")
            self.fcoverage_deg = 41253. * fsky
            self.fsky = fsky
        else:
            print("Assuming full sky survey.")
            self.fsky = 1.
            self.fcoverage_deg = 41253.

        # Generate spectra at each z for fid cosmo
        self.spectra_mid = [cf.spectrum(
                                cf.generate_data(
                                    dict(self.fid,      
                                         **{'z_pk' : j}),
                                    self.classdir,       
                                    self.datastore).replace(
                                        '/test_parameters.ini',''),
                                self.z_steps) 
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
        if forecast_type=="neutrino": 
            self.M_ncdm_high, self.M_ncdm_low = self.generate_spectra('m_ncdm')
        elif forecast_type=="relic": 
            self.T_ncdm_high, self.T_ncdm_low = self.generate_spectra('T_ncdm')

        # Calculate centered derivatives about fiducial cosmo at each z 
        self.dPdA_s, self.dlogPdA_s = dPs_array(self.A_s_low, 
                                                self.A_s_high, 
                                                self.fid['A_s']*self.dstep) 
        # ^^^ Replace w/ analytic result 
        self.dlogPdA_s = [[(1. / self.A_s_fid) 
                           for k in self.k_table] 
                          for z in self.z_steps] # Analytic form

        self.dPdn_s, self.dlogPdn_s = dPs_array(self.n_s_low, 
                                                self.n_s_high, 
                                                self.fid['n_s']*self.dstep) 
        # ^^^ Replace w/ analytic result
        self.dlogPdn_s = [[np.log( k / self.kp ) 
                           for k in self.k_table] 
                          for z in self.z_steps] # Analytic form

        self.dPdomega_b, self.dlogPdomega_b = dPs_array(
            self.omega_b_low, 
            self.omega_b_high, 
            self.fid['omega_b'] * self.dstep) 
        self.dPdomega_cdm, self.dlogPdomega_cdm = dPs_array(
            self.omega_cdm_low, 
            self.omega_cdm_high, 
            self.fid['omega_cdm']*self.dstep)
        self.dPdh, self.dlogPdh = dPs_array(
            self.h_low, 
            self.h_high, 
            self.fid['h']*self.dstep)
        self.dPdtau_reio, self.dlogPdtau_reio = dPs_array(
            self.tau_reio_low, 
            self.tau_reio_high, 
            self.fid['tau_reio']*self.dstep)
        if forecast_type=="neutrino": 
            self.dPdM_ncdm, self.dlogPdM_ncdm = dPs_array(
                self.M_ncdm_low, 
                self.M_ncdm_high, 
                3.*self.fid['m_ncdm']*self.dstep)   
                # ^^^CAUTION. Check factor of 3 in denominator.  
            self.dPdomega_ncdm = [np.array(self.dPdM_ncdm[k]) * 93.14
                                  for k in range(len(self.z_steps))]
            self.dlogPdomega_ncdm = [np.array(self.dlogPdM_ncdm[k]) * 93.14
                                     for k in range(len(self.z_steps))]
        elif forecast_type=="relic": 
            self.dPdT_ncdm, self.dlogPdT_ncdm = dPs_array(
                self.T_ncdm_low, 
                self.T_ncdm_high, 
                self.fid['T_ncdm']*self.dstep)
            self.dPdomega_ncdm = [np.array(self.dPdT_ncdm[k]) 
                                  / domega_ncdm_dT_ncdm(self.T_ncdm_fid, 
                                                        self.m_ncdm_fid)
                                  for k in range(len(self.z_steps))]
            self.dlogPdomega_ncdm = [np.array(self.dlogPdT_ncdm[k]) 
                                     / domega_ncdm_dT_ncdm(self.T_ncdm_fid, 
                                                           self.m_ncdm_fid)
                                     for k in range(len(self.z_steps))]
 
    def generate_spectra(self, param): 
        spectra_high = [cf.spectrum(
                            cf.generate_data(
                                dict(self.fid, 
                                     **{'z_pk' : j,
                                     param : (1.+self.dstep)*self.fid[param]}),
                                self.classdir,
                                self.datastore).replace(
                                    '/test_parameters.ini',''),
                                self.z_steps) 
                        for j in self.z_steps] 
        spectra_low = [cf.spectrum(
                           cf.generate_data(
                               dict(self.fid,
                                    **{'z_pk' : j,
                                    param : (1.-self.dstep)*self.fid[param]}),
                               self.classdir,
                               self.datastore).replace(
                                   '/test_parameters.ini',''),
                           self.z_steps) 
                       for j in self.z_steps] 
        return spectra_high, spectra_low

    def gen_rsd(self, mu): 
        '''Given mu, creates len(z_steps) array. Each elem is len(k_table).'''
        kfs_table = kfs(self.omega_ncdm_fid, 
                        self.h_fid, 
                        np.array(self.z_steps))
        self.RSD = [[rsd(self.omega_b_fid, 
                         self.omega_cdm_fid, 
                         self.omega_ncdm_fid, 
                         self.h_fid, 
                         kfs_table[zidx], 
                         zval, 
                         mu, 
                         kval) 
                     for kval in self.k_table] 
                    for zidx, zval in enumerate(self.z_steps)]   
        self.dlogRSDdomega_b = ((np.log([[rsd(((1.+self.dstep)
                                               * self.omega_b_fid), 
                                              self.omega_cdm_fid, 
                                              self.omega_ncdm_fid, 
                                              self.h_fid, 
                                              kfs_table[zidx], 
                                              zval, 
                                              mu, 
                                              kval) 
                                          for kval in self.k_table] 
                                         for zidx, zval in enumerate(
                                             self.z_steps)]) 
                                - np.log([[rsd(((1.-self.dstep)
                                                *self.omega_b_fid), 
                                               self.omega_cdm_fid, 
                                               self.omega_ncdm_fid, 
                                               self.h_fid, 
                                               kfs_table[zidx], 
                                               zval, 
                                               mu, 
                                               kval) 
                                           for kval in self.k_table] 
                                          for zidx, zval in enumerate(
                                              self.z_steps)])) 
                               / (2.*self.dstep*self.omega_b_fid))
        self.dlogRSDdomega_cdm = ((np.log([[rsd(self.omega_b_fid, 
                                                ((1.+self.dstep)
                                                 * self.omega_cdm_fid), 
                                                self.omega_ncdm_fid, 
                                                self.h_fid, 
                                                kfs_table[zidx], 
                                                zval, 
                                                mu, 
                                                kval) 
                                            for kval in self.k_table] 
                                           for zidx, zval in enumerate(
                                               self.z_steps)]) 
                                   - np.log([[rsd(self.omega_b_fid, 
                                                  ((1.-self.dstep)
                                                   * self.omega_cdm_fid), 
                                                  self.omega_ncdm_fid, 
                                                  self.h_fid, 
                                                  kfs_table[zidx], 
                                                  zval, 
                                                  mu, 
                                                  kval) 
                                              for kval in self.k_table] 
                                             for zidx, zval in enumerate(
                                                 self.z_steps)])) 
                                  / (2.* self.dstep * self.omega_cdm_fid))
        self.dlogRSDdomega_ncdm = ((np.log([[rsd(self.omega_b_fid, 
                                                 self.omega_cdm_fid, 
                                                 ((1.+self.dstep)
                                                  * self.omega_ncdm_fid), 
                                                 self.h_fid, 
                                                 kfs_table[zidx], 
                                                 zval, 
                                                 mu, 
                                                 kval) 
                                             for kval in self.k_table] 
                                            for zidx, zval in enumerate(
                                                self.z_steps)]) 
                                    - np.log([[rsd(self.omega_b_fid, 
                                                   self.omega_cdm_fid, 
                                                   ((1.-self.dstep)
                                                    * self.omega_ncdm_fid), 
                                                   self.h_fid, 
                                                   kfs_table[zidx], 
                                                   zval, 
                                                   mu, 
                                                   kval) 
                                               for kval in self.k_table] 
                                               for zidx, zval in enumerate(
                                                   self.z_steps)])) 
                                   / (2. * self.dstep * self.omega_ncdm_fid))
        self.dlogRSDdM_ncdm = self.dlogRSDdomega_ncdm * (1. / 93.14) 
        self.dlogRSDdh = ((np.log([[rsd(self.omega_b_fid, 
                                        self.omega_cdm_fid, 
                                        self.omega_ncdm_fid, 
                                        (1.+self.dstep)*self.h_fid, 
                                        kfs_table[zidx], 
                                        zval, 
                                        mu, 
                                        kval) 
                                    for kval in self.k_table] 
                                    for zidx, zval in enumerate(self.z_steps)]) 
                           - np.log([[rsd(self.omega_b_fid, 
                                          self.omega_cdm_fid, 
                                          self.omega_ncdm_fid, 
                                          (1.-self.dstep)*self.h_fid, 
                                          kfs_table[zidx], 
                                          zval, 
                                          mu, 
                                          kval) 
                                      for kval in self.k_table] 
                                      for zidx, zval in enumerate(
                                          self.z_steps)])) 
                          / (2.*self.dstep*self.h_fid))

    def gen_fog(self, mu):
        self.FOG = [[fog(self.h_fid, 
                         self.c, 
                         self.omega_b_fid, 
                         self.omega_cdm_fid, 
                         self.omega_ncdm_fid, 
                         zval, 
                         kval, 
                         mu) 
                     for kval in self.k_table] 
                     for zval in self.z_steps]
        h_high = np.log([[fog((1. + self.dstep) * self.h_fid, 
                              self.c, 
                              self.omega_b_fid,   
                              self.omega_cdm_fid, 
                              self.omega_ncdm_fid,    
                              zval, 
                              kval, 
                              mu)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        h_low = np.log([[fog((1. - self.dstep) * self.h_fid,
                             self.c,
                             self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             zval,
                             kval,
                             mu)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        omega_b_high = np.log([[fog(self.h_fid,
                                    self.c,
                                    (1. + self.dstep) * self.omega_b_fid,
                                    self.omega_cdm_fid,
                                    self.omega_ncdm_fid,
                                    zval,
                                    kval,
                                    mu)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        omega_b_low = np.log([[fog(self.h_fid,
                                   self.c,
                                   (1. - self.dstep) * self.omega_b_fid,
                                   self.omega_cdm_fid,
                                   self.omega_ncdm_fid,
                                   zval,
                                   kval,
                                   mu)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        omega_cdm_high = np.log([[fog(self.h_fid,
                                      self.c,
                                      self.omega_b_fid,
                                      (1. + self.dstep) * self.omega_cdm_fid,
                                      self.omega_ncdm_fid,
                                      zval,
                                      kval,
                                      mu)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        omega_cdm_low = np.log([[fog(self.h_fid,
                                     self.c,
                                     self.omega_b_fid,
                                     (1. - self.dstep) * self.omega_cdm_fid,
                                     self.omega_ncdm_fid,
                                     zval,
                                     kval,
                                     mu)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        omega_ncdm_high = np.log([[fog(self.h_fid,
                                        self.c,
                                        self.omega_b_fid,
                                        self.omega_cdm_fid,
                                        (1.+self.dstep) * self.omega_ncdm_fid,
                                        zval, 
                                        kval,
                                        mu)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        omega_ncdm_low = np.log([[fog(self.h_fid,
                                      self.c,
                                      self.omega_b_fid,
                                      self.omega_cdm_fid,
                                      (1. - self.dstep) * self.omega_ncdm_fid,
                                      zval, 
                                      kval,
                                      mu)
                     for kval in self.k_table]
                     for zval in self.z_steps])


        self.dlogFOGdh = (h_high - h_low) / (2. * self.dstep * self.h_fid)
        self.dlogFOGdomega_b = ((omega_b_high - omega_b_low)
                                / (2. * self.dstep * self.omega_b_fid))
        self.dlogFOGdomega_cdm = ((omega_cdm_high - omega_cdm_low)
                                  / (2. * self.dstep * self.omega_cdm_fid))
        self.dlogFOGdomega_ncdm = ((omega_ncdm_high - omega_ncdm_low)
                                   / (2. * self.dstep * self.omega_ncdm_fid))
        self.dlogFOGdM_ncdm = self.dlogFOGdomega_ncdm * (1. / 93.14) 

    def gen_ap(self): 
        h_high_omega_b = np.log([[H((1.+self.dstep)*self.omega_b_fid, 
                             self.omega_cdm_fid, 
                             self.omega_ncdm_fid, 
                             self.h_fid, 
                             zval)
                           for kval in self.k_table]  
                           for zval in self.z_steps]) 
        h_low_omega_b = np.log([[H((1.-self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        h_high_omega_cdm = np.log([[H(self.omega_b_fid,
                               (1.+self.dstep)*self.omega_cdm_fid,
                               self.omega_ncdm_fid,
                               self.h_fid,
                               zval)
                             for kval in self.k_table]
                             for zval in self.z_steps])       
        h_low_omega_cdm = np.log([[H(self.omega_b_fid,
                              (1.-self.dstep)*self.omega_cdm_fid,
                              self.omega_ncdm_fid,
                              self.h_fid,
                              zval)
                            for kval in self.k_table]
                            for zval in self.z_steps])  
        h_high_omega_ncdm = np.log([[H(self.omega_b_fid,
                                self.omega_cdm_fid,
                                (1.+self.dstep)*self.omega_ncdm_fid,
                                self.h_fid,
                                zval)
                              for kval in self.k_table]
                              for zval in self.z_steps])
        h_low_omega_ncdm = np.log([[H(self.omega_b_fid,
                               self.omega_cdm_fid,
                               (1.-self.dstep)*self.omega_ncdm_fid,
                               self.h_fid,
                               zval)
                             for kval in self.k_table]
                             for zval in self.z_steps])       
        h_high_h = np.log([[H(self.omega_b_fid,
                       self.omega_cdm_fid,
                       self.omega_ncdm_fid,
                       (1.+self.dstep)*self.h_fid,
                       zval)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        h_low_h = np.log([[H(self.omega_b_fid,
                      self.omega_cdm_fid,
                      self.omega_ncdm_fid,
                      (1.-self.dstep)*self.h_fid,
                      zval)
                    for kval in self.k_table]
                    for zval in self.z_steps])

        da_high_omega_b = np.log([[Da((1.+self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval,
                             self.c)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        da_low_omega_b = np.log([[Da((1.-self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval,
                             self.c)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        da_high_omega_cdm = np.log([[Da(self.omega_b_fid,
                               (1.+self.dstep)*self.omega_cdm_fid,
                               self.omega_ncdm_fid,
                               self.h_fid,
                               zval,
                               self.c)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        da_low_omega_cdm = np.log([[Da(self.omega_b_fid,
                              (1.-self.dstep)*self.omega_cdm_fid,
                              self.omega_ncdm_fid,
                              self.h_fid,
                              zval,
                              self.c)
                            for kval in self.k_table]
                            for zval in self.z_steps])
        da_high_omega_ncdm = np.log([[Da(self.omega_b_fid,
                                self.omega_cdm_fid,
                                (1.+self.dstep)*self.omega_ncdm_fid,
                                self.h_fid,
                                zval,
                                self.c)
                              for kval in self.k_table]
                              for zval in self.z_steps])
        da_low_omega_ncdm = np.log([[Da(self.omega_b_fid,
                               self.omega_cdm_fid,
                               (1.-self.dstep)*self.omega_ncdm_fid,
                               self.h_fid,
                               zval,
                               self.c)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        da_high_h = np.log([[Da(self.omega_b_fid,
                       self.omega_cdm_fid,
                       self.omega_ncdm_fid,
                       (1.+self.dstep)*self.h_fid,
                       zval,
                       self.c)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        da_low_h = np.log([[Da(self.omega_b_fid,
                      self.omega_cdm_fid,
                      self.omega_ncdm_fid,
                      (1.-self.dstep)*self.h_fid,
                      zval,
                      self.c)
                    for kval in self.k_table]
                    for zval in self.z_steps])
        
        self.dlogAPdomega_b = (((h_high_omega_b - h_low_omega_b)
                                / (2. * self.dstep * self.omega_b_fid))
                               - ((da_high_omega_b - da_low_omega_b)
                                  / (self.dstep * self.omega_b_fid)))
        self.dlogAPdomega_cdm = (((h_high_omega_cdm - h_low_omega_cdm)
                                / (2. * self.dstep * self.omega_cdm_fid))
                               - ((da_high_omega_cdm - da_low_omega_cdm)
                                  / (self.dstep * self.omega_cdm_fid)))
        self.dlogAPdomega_ncdm = (((h_high_omega_ncdm - h_low_omega_ncdm)
                                / (2. * self.dstep * self.omega_ncdm_fid))
                               - ((da_high_omega_ncdm - da_low_omega_ncdm)
                                  / (self.dstep * self.omega_ncdm_fid)))
        self.dlogAPdM_ncdm = self.dlogAPdomega_ncdm * (1. / 93.14) 
        self.dlogAPdh = (((h_high_h - h_low_h)
                                / (2. * self.dstep * self.h_fid))
                               - ((da_high_h - da_low_h)
                                  / (self.dstep * self.h_fid)))
                            
    def gen_cov(self, mu): 
        H_high_omega_b = np.array([[H((1.+self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        H_low_omega_b = np.array([[H((1.-self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        H_high_omega_cdm = np.array([[H(self.omega_b_fid,
                               (1.+self.dstep)*self.omega_cdm_fid,
                               self.omega_ncdm_fid,
                               self.h_fid,
                               zval)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        H_low_omega_cdm = np.array([[H(self.omega_b_fid,
                              (1.-self.dstep)*self.omega_cdm_fid,
                              self.omega_ncdm_fid,
                              self.h_fid,
                              zval)
                            for kval in self.k_table]
                            for zval in self.z_steps])
        H_high_omega_ncdm = np.array([[H(self.omega_b_fid,
                                self.omega_cdm_fid,
                                (1.+self.dstep)*self.omega_ncdm_fid,
                                self.h_fid,
                                zval)
                              for kval in self.k_table]
                              for zval in self.z_steps])
        H_low_omega_ncdm = np.array([[H(self.omega_b_fid,
                               self.omega_cdm_fid,
                               (1.-self.dstep)*self.omega_ncdm_fid,
                               self.h_fid,
                               zval)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        H_high_h = np.array([[H(self.omega_b_fid,
                       self.omega_cdm_fid,
                       self.omega_ncdm_fid,
                       (1.+self.dstep)*self.h_fid,
                       zval)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        H_low_h = np.array([[H(self.omega_b_fid,
                      self.omega_cdm_fid,
                      self.omega_ncdm_fid,
                      (1.-self.dstep)*self.h_fid,
                      zval)
                    for kval in self.k_table]
                    for zval in self.z_steps])
        Da_high_omega_b = np.array([[Da((1.+self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval,
                             self.c)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        Da_low_omega_b = np.array([[Da((1.-self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval,
                             self.c)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        Da_high_omega_cdm = np.array([[Da(self.omega_b_fid,
                               (1.+self.dstep)*self.omega_cdm_fid,
                               self.omega_ncdm_fid,
                               self.h_fid,
                               zval,
                               self.c)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        Da_low_omega_cdm = np.array([[Da(self.omega_b_fid,
                              (1.-self.dstep)*self.omega_cdm_fid,
                              self.omega_ncdm_fid,
                              self.h_fid,
                              zval,
                              self.c)
                            for kval in self.k_table]
                            for zval in self.z_steps])
        Da_high_omega_ncdm = np.array([[Da(self.omega_b_fid,
                                self.omega_cdm_fid,
                                (1.+self.dstep)*self.omega_ncdm_fid,
                                self.h_fid,
                                zval,
                                self.c)
                              for kval in self.k_table]
                              for zval in self.z_steps])
        Da_low_omega_ncdm = np.array([[Da(self.omega_b_fid,
                               self.omega_cdm_fid,
                               (1.-self.dstep)*self.omega_ncdm_fid,
                               self.h_fid,
                               zval,
                               self.c)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        Da_high_h = np.array([[Da(self.omega_b_fid,
                       self.omega_cdm_fid,
                       self.omega_ncdm_fid,
                       (1.+self.dstep)*self.h_fid,
                       zval,
                       self.c)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        Da_low_h = np.array([[Da(self.omega_b_fid,
                      self.omega_cdm_fid,
                      self.omega_ncdm_fid,
                      (1.-self.dstep)*self.h_fid,
                      zval,
                      self.c)
                    for kval in self.k_table]
                    for zval in self.z_steps])

        dkdH = [[(2. * kval * np.power(mu, 2.))
                      / H(self.omega_b_fid, self.omega_cdm_fid,
                          self.omega_ncdm_fid, self.h_fid, zval)
                      for kval in self.k_table]
                      for zval in self.z_steps]

        dkdDa = [[(-2. * kval * (1. - np.power(mu, 2.)))
                       / Da(self.omega_b_fid, self.omega_cdm_fid,
                            self.omega_ncdm_fid, self.h_fid, zval, self.c)
                       for kval in self.k_table]
                       for zval in self.z_steps]

        dHdomega_b = ((H_high_omega_b - H_low_omega_b) 
                      / (2. * self.dstep * self.omega_b_fid))
        dHdomega_cdm = ((H_high_omega_cdm - H_low_omega_cdm)
                        / (2. * self.dstep * self.omega_cdm_fid))
        dHdomega_ncdm = ((H_high_omega_ncdm - H_low_omega_ncdm)
                         / (2. * self.dstep * self.omega_ncdm_fid))
        dHdh = (H_high_h - H_low_h) / (2. * self.dstep * self.h_fid)

        dDdomega_b = ((Da_high_omega_b - Da_low_omega_b)
                      / (2. * self.dstep * self.omega_b_fid))
        dDdomega_cdm = ((Da_high_omega_cdm - Da_low_omega_cdm)
                        / (2. * self.dstep * self.omega_cdm_fid))
        dDdomega_ncdm = ((Da_high_omega_ncdm - Da_low_omega_ncdm)
                         / (2. * self.dstep * self.omega_ncdm_fid))
        dDdh = (Da_high_h - Da_low_h) / (2. * self.dstep * self.h_fid)

        dkdomega_b = dkdH * dHdomega_b + dkdDa * dDdomega_b
        dkdomega_cdm = dkdH * dHdomega_cdm + dkdDa * dDdomega_cdm
        dkdomega_ncdm = dkdH * dHdomega_ncdm + dkdDa * dDdomega_ncdm
        dkdh = dkdH * dHdh + dkdDa * dDdh

        dmudomega_b = [[(mu / kval) 
                        * dkdomega_b[zidx][kidx]
                        + (mu / H(self.omega_b_fid,
                                  self.omega_cdm_fid,
                                  self.omega_ncdm_fid,
                                  self.h_fid,
                                  zval))
                        * dHdomega_b[zidx][kidx]
                        for kidx, kval in enumerate(self.k_table)]
                        for zidx, zval in enumerate(self.z_steps)]
        dmudomega_cdm = [[(mu / kval)
                          * dkdomega_cdm[zidx][kidx]
                          + (mu / H(self.omega_b_fid,
                                    self.omega_cdm_fid,
                                    self.omega_ncdm_fid,
                                    self.h_fid,
                                    zval))
                          * dHdomega_cdm[zidx][kidx]
                          for kidx, kval in enumerate(self.k_table)]
                          for zidx, zval in enumerate(self.z_steps)]
        dmudomega_ncdm = [[(mu / kval)
                           * dkdomega_ncdm[zidx][kidx]
                           + (mu / H(self.omega_b_fid,
                                     self.omega_cdm_fid,
                                     self.omega_ncdm_fid,
                                     self.h_fid,
                                     zval))
                           * dHdomega_ncdm[zidx][kidx]
                           for kidx, kval in enumerate(self.k_table)]
                           for zidx, zval in enumerate(self.z_steps)]
        dmudh = [[(mu / kval)
                  * dkdh[zidx][kidx]
                  + (mu / H(self.omega_b_fid,
                            self.omega_cdm_fid,
                            self.omega_ncdm_fid,
                            self.h_fid,
                            zval))
                  * dHdh[zidx][kidx]
                  for kidx, kval in enumerate(self.k_table)]
                  for zidx, zval in enumerate(self.z_steps)]

        # Fix this, very redundant, slow, error prone
        self.gen_rsd(1.01 * mu)
        self.gen_fog(1.01 * mu)
        logP_mu_high = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG) 
        P_mu_high = np.array(self.Pg) * np.array(self.RSD) * np.array(self.FOG)       
 
        self.gen_rsd(0.99 * mu)
        self.gen_fog(0.99 * mu)
        logP_mu_low = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG)
        P_mu_low = np.array(self.Pg) * np.array(self.RSD) * np.array(self.FOG)
        
        self.gen_rsd(mu)
        self.gen_rsd(mu)
        logP_mid = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG)
        P_mid = np.array(self.Pg) * np.array(self.RSD) * np.array(self.FOG)

        dlogPdmu = (logP_mu_high - logP_mu_low) / (2. * 0.01 * mu) 
        dPdmu = (P_mu_high - P_mu_low) / (2. * 0.01 * mu)
        dlogPdk = np.zeros((len(self.z_steps), len(self.k_table)))
        dPdk = np.zeros((len(self.z_steps), len(self.k_table)))
        
        for zidx, zval in enumerate(self.z_steps): 
            for kidx, kval in enumerate(self.k_table[1:-1]):
                # Careful with this derivative definition, uneven spacing
                dlogPdk[zidx][kidx+1] = ((logP_mid[zidx][kidx+2] 
                                          - logP_mid[zidx][kidx])
                                         / (self.k_table[kidx+2]
                                            - self.k_table[kidx]))
                dPdk[zidx][kidx+1] = ((P_mid[zidx][kidx+2]
                                       - P_mid[zidx][kidx])
                                      / (self.k_table[kidx+2]
                                         - self.k_table[kidx]))

        # Careful with this substitution - is it appropriate? 
        for zidx, zval in enumerate(self.z_steps): 
            dlogPdk[zidx][0] = dlogPdk[zidx][1]
            dlogPdk[zidx][-1] = dlogPdk[zidx][-2]
            dPdk[zidx][0] = dPdk[zidx][1]
            dPdk[zidx][-1] = dPdk[zidx][-2]

        dlogPdomega_b = dlogPdk * dkdomega_b + dlogPdmu * dmudomega_b
        dlogPdomega_cdm = dlogPdk * dkdomega_cdm + dlogPdmu * dmudomega_cdm
        dlogPdomega_ncdm = dlogPdk * dkdomega_ncdm + dlogPdmu * dmudomega_ncdm
        dlogPdh = dlogPdk * dkdh + dlogPdmu * dmudh

        dPdomega_b = dPdk * dkdomega_b + dPdmu * dmudomega_b
        dPdomega_cdm = dPdk * dkdomega_cdm + dPdmu * dmudomega_cdm
        dPdomega_ncdm = dPdk * dkdomega_ncdm + dPdmu * dmudomega_ncdm
        dPdh = dPdk * dkdh + dPdmu * dmudh

        self.dlogCOVdomega_b = np.array(dlogPdomega_b)
        self.dlogCOVdomega_cdm = np.array(dlogPdomega_cdm)
        self.dlogCOVdomega_ncdm = np.array(dlogPdomega_ncdm)
        self.dlogCOVdM_ncdm = self.dlogCOVdomega_ncdm * (1. / 93.14) 
        self.dlogCOVdh = np.array(dlogPdh)
    
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
                   * np.power(self.c / (1000. * 100. * self.h_fid), 3.)
                   * np.power(z_integral_max, 3.))
        V_min = ((4. * np.pi / 3.)
                   * np.power(self.c / (1000. * 100. * self.h_fid), 3.)
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
                       0.84 / self.spectra_mid[zidx].D_table[zidx]))
        return 
        
    def gen_fisher(self, mu_step): # Really messy and inefficient
        fisher = np.zeros((7, 7))

        mu_vals = np.arange(-1., 1., mu_step)
        Pm = np.zeros((len(self.z_steps), 
                       len(self.k_table), 
                       len(mu_vals)))
        dlogPdA_s = np.zeros((len(self.z_steps), 
                              len(self.k_table), 
                              len(mu_vals)))
        dlogPdn_s = np.zeros((len(self.z_steps), 
                              len(self.k_table), 
                              len(mu_vals)))
        dlogPdomega_b = np.zeros((len(self.z_steps), 
                                  len(self.k_table), 
                                  len(mu_vals)))
        dlogPdomega_cdm = np.zeros((len(self.z_steps), 
                                    len(self.k_table), 
                                    len(mu_vals)))
        dlogPdh = np.zeros((len(self.z_steps), 
                            len(self.k_table), 
                            len(mu_vals)))
        dlogPdtau_reio = np.zeros((len(self.z_steps), 
                                   len(self.k_table), 
                                   len(mu_vals)))
        dlogPdomega_ncdm = np.zeros((len(self.z_steps), 
                                     len(self.k_table), 
                                     len(mu_vals)))

        
        for muidx, muval in enumerate(mu_vals): 
            #self.gen_rsd(muval)
            self.gen_fog(muval)
            self.gen_ap()
            self.gen_cov(muval)
            for zidx, zval in enumerate(self.z_steps): 
                for kidx, kval in enumerate(self.k_table):
                    Pm[zidx][kidx][muidx] = (
                        self.spectra_mid[zidx].ps_table[kidx] 
                        #* self.RSD[zidx][kidx] 
                        * self.FOG[zidx][kidx]
                        #### * self.COV[zidx][kidx] # Need to make this term
                        )
                    dlogPdA_s[zidx][kidx][muidx] = (
                        self.dlogPdA_s[zidx][kidx]
                        )
                    dlogPdn_s[zidx][kidx][muidx] = (
                        self.dlogPdn_s[zidx][kidx]
                        )
                    dlogPdomega_b[zidx][kidx][muidx] = (
                        self.dlogPdomega_b[zidx][kidx]
                        #+ self.dlogRSDdomega_b[zidx][kidx]
                        + self.dlogFOGdomega_b[zidx][kidx]
                        + self.dlogAPdomega_b[zidx][kidx]
                        + self.dlogCOVdomega_b[zidx][kidx]
                        )
                    dlogPdomega_cdm[zidx][kidx][muidx] = (
                        self.dlogPdomega_cdm[zidx][kidx]
                        #+ self.dlogRSDdomega_cdm[zidx][kidx]
                        + self.dlogFOGdomega_cdm[zidx][kidx]
                        + self.dlogAPdomega_cdm[zidx][kidx]
                        + self.dlogCOVdomega_cdm[zidx][kidx]
                        )
                    dlogPdh[zidx][kidx][muidx] = (
                        self.dlogPdh[zidx][kidx]
                        #+ self.dlogRSDdh[zidx][kidx] 
                        + self.dlogFOGdh[zidx][kidx]
                        + self.dlogAPdh[zidx][kidx]
                        + self.dlogCOVdh[zidx][kidx]
                        )
                    dlogPdtau_reio[zidx][kidx][muidx] = (
                        self.dlogPdtau_reio[zidx][kidx]
                        ) 
                    dlogPdomega_ncdm[zidx][kidx][muidx] = (
                        self.dlogPdomega_ncdm[zidx][kidx]
                        #+ self.dlogRSDdomega_ncdm[zidx][kidx]
                        + self.dlogFOGdomega_ncdm[zidx][kidx]
                        + self.dlogAPdomega_ncdm[zidx][kidx]
                        + self.dlogCOVdomega_ncdm[zidx][kidx]
                        )
                    dlogPdM_ncdm = dlogPdomega_ncdm * (1. / 93.14) 
                    # ^^^Careful, this overwrites the earlier dP_g value. 
                    # we do this to reflect corrections in the contours
                    # for M_ncdm 

        self.Pm = Pm 

        if self.forecast=="neutrino": 
            paramvec = [dlogPdomega_b, 
                        dlogPdomega_cdm,  
                        dlogPdn_s, 
                        dlogPdA_s,
                        dlogPdtau_reio, 
                        dlogPdh, 
                        dlogPdM_ncdm]
        elif self.forecast=="relic": 
            paramvec = [dlogPdomega_b, 
                        dlogPdomega_cdm,  
                        dlogPdn_s, 
                        dlogPdA_s,
                        dlogPdtau_reio, 
                        dlogPdh, 
                        dlogPdomega_ncdm] 

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
                
    def print_P_table(self): 
        for zidx, zval in enumerate(self.z_steps): 
            print((("For z = {0:.2f},\t") + 
                   (" P(0.2h, 0) = {1:.2f}\n")).format(zval, 
                                                        self.Pm[zidx][69][20]))

def H(omega_b, omega_cdm, omega_ncdm, h, z):
    # Returns H in units of m/s/Mpc
    omega_m = omega_b + omega_cdm + omega_ncdm
    omega_lambda = np.power(h, 2.) - omega_m
    Hval = 1000.*100.*h*np.sqrt(omega_m * np.power(1. + z, 3.) + omega_lambda)
    return Hval  

def Da(omega_b, omega_cdm, omega_ncdm, h, z, c): 
    prefactor = c / (1. + z) 
    def integrand(zval): 
        return 1. / H(omega_b, omega_cdm, omega_ncdm, h, zval)
    integral, error = quad(integrand, 0., z) 
    return prefactor * integral 

def neff(ndens, Pg): 
    # ndens at specific z, Pm at specific k and z 
    n = np.power((ndens * Pg) / (ndens*Pg + 1.), 2.)
    return n 

def fog(h, c, omega_b, omega_cdm, omega_ncdm, z, k, mu): 
    #sigma_fog_0 = 250. # Units [km*s^-1]
    sigma_fog_0 = 250000. # Units [m*s^-1] 
    # ^^^^ CAUTION! Convince self which is correct.
    sigma_z = 0.001
    sigma_fog = sigma_fog_0 * np.sqrt(1.+z)
    sigma_v = ((1. + z) 
               * np.sqrt((np.power(sigma_fog, 2.) / 2.) 
                         + np.power(c*sigma_z, 2.)))
    F = np.exp(-1. 
               * np.power((k*mu*sigma_v) 
               / H(omega_b, omega_cdm, omega_ncdm, h, z), 2.))
    return F 
        

def kfs(omega_ncdm, h, z): 
    k_fs = (940. * 0.08 * omega_ncdm * h) / np.sqrt(1. + z)
    return k_fs

def rsd(omega_b, omega_cdm, omega_ncdm, h, k_fs, z, mu, k):
    gamma = 0.55
    b1L = 0.5 
    Deltaq = 1.6
    q = (5. * k) / k_fs
    epsilon = (omega_b + omega_cdm) / np.power(h, 2.) 
    f = np.power((epsilon * np.power(1.+z, 3.)) 
                 / ( epsilon * np.power(1.+z, 3.) - epsilon + 1.), gamma)
    DeltaL = (0.6 * omega_ncdm) / (omega_b + omega_cdm) 
    g = 1. + (DeltaL / 2.) * np.tanh(1. + (np.log(q) / Deltaq))
    b1tilde = 1. + b1L * g 

    R = np.power((b1tilde + np.power(mu, 2.) * f), 2.)  
    return R  


def dPs_array(low, high, step): 
    dPs = [(high[zval].ps_table - low[zval].ps_table)/(2.*step) 
           for zval in range(len(high))]
    dlogPs = [(high[zval].log_ps_table - low[zval].log_ps_table)/(2.*step) 
              for zval in range(len(high))] 
    return dPs, dlogPs 

def omega_ncdm(T_ncdm, m_ncdm, forecast_type): 
    """Returns omega_ncdm as a function of T_ncdm, m_ncdm.

    T_ncdm : relic temperature in units [K]
    m_ncdm : relic mass in units [eV]
    
    omega_ncdm : relative relic abundance. Unitless. 
    """
    if forecast_type=="neutrino": 
        omega_ncdm = 3. * (m_ncdm / 93.14)
    if forecast_type=="relic": 
        omega_ncdm = np.power(T_ncdm / 1.95, 3.) * (m_ncdm / 94.) 
    return omega_ncdm 


def T_ncdm(omega_ncdm, m_ncdm): 
    # RELICS ONLY? 
    """Returns T_ncdm as a function of omega_ncdm, m_ncdm. 

    omega_ncdm : relative relic abundance. Unitless. 
    m_ncdm : relic mass in units [eV]. 
    
    T_ncdm : relic temperature in units [K]
    """

    T_ncdm = np.power( 94. * omega_ncdm / m_ncdm, 1./3.) * 1.95
    return T_ncdm 


def domega_ncdm_dT_ncdm(T_ncdm, m_ncdm): 
    # RELICS ONLY? 
    """Returns derivative of omega_ncdm wrt T_ncdm. 

    T_ncdm : relic temperature in units [K]
    m_ncdm : relic mass in units [eV]
    
    deriv : derivative of relic abundance wrt relic temp in units [K]^(-1)  
    """

    deriv = (3. * m_ncdm / 94.) * np.power(T_ncdm, 2.) * np.power(1.95, -3.) 
    return deriv

def dT_ncdm_domega_ncdm(omega_ncdm, m_ncdm): 
    # RELICS ONLY? 
    """Returns derivative of T_ncdm wrt  omega_ncdm.

    omega_ncdm : relative relic abundance. Unitless. 
    m_ncdm : relic mass in units [eV]. 
    
    deriv : derivative of relic temp wrt relic abundance in units [K] 
    """

    deriv = ((1.95 / 3) 
             * np.power(94. / m_ncdm, 1./3.) 
             * np.power(omega_ncdm, -2./3.))
    return deriv 
        
