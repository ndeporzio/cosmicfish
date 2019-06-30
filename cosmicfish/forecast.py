import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import quad 

import cosmicfish as cf 

class relic_forecast: 
    """Given fiducial cosmology, generates forecasted Fisher and Covariance matrices."""
    
    def __init__(self, fiducialcosmo, z_steps, n_densities, dstep, classdir, datastore): 
        self.fid = fiducialcosmo
        self.z_steps = z_steps
        self.n_densities = n_densities
        self.A_s_fid = fiducialcosmo['A_s']
        self.n_s_fid = fiducialcosmo['n_s']
        self.omega_b_fid = fiducialcosmo['omega_b']
        self.omega_cdm_fid = fiducialcosmo['omega_cdm']
        self.h_fid = fiducialcosmo['h']
        self.tau_reio_fid = fiducialcosmo['tau_reio']
        self.T_ncdm_fid = fiducialcosmo['T_ncdm'] #Units [T_cmb]
        self.m_ncdm_fid = fiducialcosmo['m_ncdm'] #Unit [eV] 
        self.omega_ncdm_fid = cf.omega_ncdm(self.T_ncdm_fid, self.m_ncdm_fid)
        self.dstep = dstep 
        self.classdir = classdir
        self.datastore = datastore
        self.c = 2.9979e8 
        self.kp = 0.05 * self.h_fid #Units [Mpc^-1]

        #Generate spectra at each z for fid cosmo
        self.spectra_mid = [cf.spectrum(cf.generate_data(dict(self.fid,      
                                                              **{'z_pk' : j}),
                                                         self.classdir,       
                                                         self.datastore).replace('/test_parameters.ini',''),
                                        self.z_steps) for j in self.z_steps]
        self.k_table = self.spectra_mid[0].k_table #All spectra should have same k_table
        self.Pg = [self.spectra_mid[zidx].ps_table for zidx, zval in enumerate(self.z_steps)]

        #Generate variation spectra at each z    
        self.A_s_high, self.A_s_low = self.generate_spectra('A_s') 
        self.n_s_high, self.n_s_low = self.generate_spectra('n_s')
        self.omega_b_high, self.omega_b_low = self.generate_spectra('omega_b')
        self.omega_cdm_high, self.omega_cdm_low = self.generate_spectra('omega_cdm')
        self.h_high, self.h_low = self.generate_spectra('h')
        self.tau_reio_high, self.tau_reio_low = self.generate_spectra('tau_reio')
        self.T_ncdm_high, self.T_ncdm_low = self.generate_spectra('T_ncdm')

        #Calculate centered derivatives about fiducial cosmo at each z 
        self.dPdA_s, self.dlogPdA_s = dPs_array(self.A_s_low, self.A_s_high, self.fid['A_s']*self.dstep) #Replace w/ analytic result 
        self.dlogPdA_s = [[(1. / self.A_s_fid) for k in self.k_table] for z in self.z_steps] #Analytic form

        self.dPdn_s, self.dlogPdn_s = dPs_array(self.n_s_low, self.n_s_high, self.fid['n_s']*self.dstep) #Replace w/ analytic result
        self.dlogPdn_s = [[np.log( k / self.kp ) for k in self.k_table] for z in self.z_steps] #Analytic form

        self.dPdomega_b, self.dlogPdomega_b = dPs_array(self.omega_b_low, self.omega_b_high, self.fid['omega_b']*self.dstep)
        self.dPdomega_cdm, self.dlogPdomega_cdm = dPs_array(self.omega_cdm_low, self.omega_cdm_high, self.fid['omega_cdm']*self.dstep)
        self.dPdh, self.dlogPdh = dPs_array(self.h_low, self.h_high, self.fid['h']*self.dstep)
        self.dPdtau_reio, self.dlogPdtau_reio = dPs_array(self.tau_reio_low, self.tau_reio_high, self.fid['tau_reio']*self.dstep)
        self.dPdT_ncdm, self.dlogPdT_ncdm = dPs_array(self.T_ncdm_low, self.T_ncdm_high, self.fid['T_ncdm']*self.dstep)    
    
        self.dPdomega_ncdm = [np.array(self.dPdT_ncdm[k]) / domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.m_ncdm_fid)
                              for k in range(len(self.z_steps))]
        self.dlogPdomega_ncdm = [np.array(self.dlogPdT_ncdm[k]) / domega_ncdm_dT_ncdm(self.T_ncdm_fid, self.m_ncdm_fid) 
                              for k in range(len(self.z_steps))]
 
    def generate_spectra(self, param): 
        spectra_high = [cf.spectrum(cf.generate_data(dict(self.fid, 
                                                          **{'z_pk' : j,
                                                             param : (1.+self.dstep)*self.fid[param]}),
                                                     self.classdir,
                                                     self.datastore).replace('/test_parameters.ini',''),
                                    self.z_steps) for j in self.z_steps] 
        spectra_low = [cf.spectrum(cf.generate_data(dict(self.fid,
                                                          **{'z_pk' : j,
                                                             param : (1.-self.dstep)*self.fid[param]}),
                                                    self.classdir,
                                                    self.datastore).replace('/test_parameters.ini',''),
                                   self.z_steps) for j in self.z_steps] 
        return spectra_high, spectra_low

    def gen_rsd(self, mu): 
        """For given val of mu, generates array w/ len(z_steps) elems. Each elem is len(k_table)."""
        kfs_table = kfs(self.omega_ncdm_fid, self.h_fid, np.array(self.z_steps))
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
                                              *self.omega_b_fid), 
                                             self.omega_cdm_fid, 
                                             self.omega_ncdm_fid, 
                                             self.h_fid, 
                                             kfs_table[zidx], 
                                             zval, 
                                             mu, 
                                             kval) 
                                          for kval in self.k_table] 
                                          for zidx, zval in enumerate(self.z_steps)]) 
                                - np.log([[rsd((1.-self.dstep)*self.omega_b_fid, 
                                               self.omega_cdm_fid, 
                                               self.omega_ncdm_fid, 
                                               self.h_fid, 
                                               kfs_table[zidx], 
                                               zval, 
                                               mu, 
                                               kval) 
                                           for kval in self.k_table] 
                                           for zidx, zval in enumerate(self.z_steps)])) 
                               / (2.*self.dstep*self.omega_b_fid))
        self.dlogRSDdomega_cdm = ((np.log([[rsd(self.omega_b_fid, 
                                               (1.+self.dstep)*self.omega_cdm_fid, 
                                               self.omega_ncdm_fid, 
                                               self.h_fid, 
                                               kfs_table[zidx], 
                                               zval, 
                                               mu, 
                                               kval) 
                                           for kval in self.k_table] 
                                           for zidx, zval in enumerate(self.z_steps)]) 
                                   - np.log([[rsd(self.omega_b_fid, 
                                                 (1.-self.dstep)*self.omega_cdm_fid, 
                                                 self.omega_ncdm_fid, 
                                                 self.h_fid, 
                                                 kfs_table[zidx], 
                                                 zval, 
                                                 mu, 
                                                 kval) 
                                             for kval in self.k_table] 
                                             for zidx, zval in enumerate(self.z_steps)])) 
                                  / (2.*self.dstep*self.omega_cdm_fid))
        self.dlogRSDdomega_ncdm = ((np.log([[rsd(self.omega_b_fid, 
                                                 self.omega_cdm_fid, 
                                                 (1.+self.dstep)*self.omega_ncdm_fid, 
                                                 self.h_fid, 
                                                 kfs_table[zidx], 
                                                 zval, 
                                                 mu, 
                                                 kval) 
                                             for kval in self.k_table] 
                                             for zidx, zval in enumerate(self.z_steps)]) 
                                    - np.log([[rsd(self.omega_b_fid, 
                                                   self.omega_cdm_fid, 
                                                   (1.-self.dstep)*self.omega_ncdm_fid, 
                                                   self.h_fid, 
                                                   kfs_table[zidx], 
                                                   zval, 
                                                   mu, 
                                                   kval) 
                                               for kval in self.k_table] 
                                               for zidx, zval in enumerate(self.z_steps)])) 
                                   / (2.*self.dstep*self.omega_ncdm_fid))
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
                                      for zidx, zval in enumerate(self.z_steps)])) 
                          / (2.*self.dstep*self.h_fid))

    def gen_fog(self, mu):
        self.FOG = [[fog(self.h_fid, self.c, zval, kval, mu) 
                     for kval in self.k_table] 
                     for zval in self.z_steps]
        high = np.log([[fog((1.+self.dstep)*self.h_fid, self.c, zval, kval, mu) 
                        for kval in self.k_table] 
                        for zval in self.z_steps])
        low = np.log([[fog((1.-self.dstep)*self.h_fid, self.c, zval, kval, mu) 
                       for kval in self.k_table] 
                       for zval in self.z_steps])
        diff = high - low
        np.nan_to_num(diff)
        denom = 2.*self.dstep*self.h_fid
        self.dlogFOGdh = diff / denom

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
        self.dlogAPdh = (((h_high_h - h_low_h)
                                / (2. * self.dstep * self.h_fid))
                               - ((da_high_h - da_low_h)
                                  / (self.dstep * self.h_fid)))
                            
    def gen_cov(self, mu): 
        h_high_omega_b = np.array([[H((1.+self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        h_low_omega_b = np.array([[H((1.-self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        h_high_omega_cdm = np.array([[H(self.omega_b_fid,
                               (1.+self.dstep)*self.omega_cdm_fid,
                               self.omega_ncdm_fid,
                               self.h_fid,
                               zval)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        h_low_omega_cdm = np.array([[H(self.omega_b_fid,
                              (1.-self.dstep)*self.omega_cdm_fid,
                              self.omega_ncdm_fid,
                              self.h_fid,
                              zval)
                            for kval in self.k_table]
                            for zval in self.z_steps])
        h_high_omega_ncdm = np.array([[H(self.omega_b_fid,
                                self.omega_cdm_fid,
                                (1.+self.dstep)*self.omega_ncdm_fid,
                                self.h_fid,
                                zval)
                              for kval in self.k_table]
                              for zval in self.z_steps])
        h_low_omega_ncdm = np.array([[H(self.omega_b_fid,
                               self.omega_cdm_fid,
                               (1.-self.dstep)*self.omega_ncdm_fid,
                               self.h_fid,
                               zval)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        h_high_h = np.array([[H(self.omega_b_fid,
                       self.omega_cdm_fid,
                       self.omega_ncdm_fid,
                       (1.+self.dstep)*self.h_fid,
                       zval)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        h_low_h = np.array([[H(self.omega_b_fid,
                      self.omega_cdm_fid,
                      self.omega_ncdm_fid,
                      (1.-self.dstep)*self.h_fid,
                      zval)
                    for kval in self.k_table]
                    for zval in self.z_steps])
        da_high_omega_b = np.array([[Da((1.+self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval,
                             self.c)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        da_low_omega_b = np.array([[Da((1.-self.dstep)*self.omega_b_fid,
                             self.omega_cdm_fid,
                             self.omega_ncdm_fid,
                             self.h_fid,
                             zval,
                             self.c)
                           for kval in self.k_table]
                           for zval in self.z_steps])
        da_high_omega_cdm = np.array([[Da(self.omega_b_fid,
                               (1.+self.dstep)*self.omega_cdm_fid,
                               self.omega_ncdm_fid,
                               self.h_fid,
                               zval,
                               self.c)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        da_low_omega_cdm = np.array([[Da(self.omega_b_fid,
                              (1.-self.dstep)*self.omega_cdm_fid,
                              self.omega_ncdm_fid,
                              self.h_fid,
                              zval,
                              self.c)
                            for kval in self.k_table]
                            for zval in self.z_steps])
        da_high_omega_ncdm = np.array([[Da(self.omega_b_fid,
                                self.omega_cdm_fid,
                                (1.+self.dstep)*self.omega_ncdm_fid,
                                self.h_fid,
                                zval,
                                self.c)
                              for kval in self.k_table]
                              for zval in self.z_steps])
        da_low_omega_ncdm = np.array([[Da(self.omega_b_fid,
                               self.omega_cdm_fid,
                               (1.-self.dstep)*self.omega_ncdm_fid,
                               self.h_fid,
                               zval,
                               self.c)
                             for kval in self.k_table]
                             for zval in self.z_steps])
        da_high_h = np.array([[Da(self.omega_b_fid,
                       self.omega_cdm_fid,
                       self.omega_ncdm_fid,
                       (1.+self.dstep)*self.h_fid,
                       zval,
                       self.c)
                     for kval in self.k_table]
                     for zval in self.z_steps])
        da_low_h = np.array([[Da(self.omega_b_fid,
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

        dkdDa = [[(2. * kval * (1. - np.power(mu, 2.)))
                       / Da(self.omega_b_fid, self.omega_cdm_fid,
                            self.omega_ncdm_fid, self.h_fid, zval, self.c)
                       for kval in self.k_table]
                       for zval in self.z_steps]

        dHdomega_b = ((h_high_omega_b - h_low_omega_b) 
                      / (2. * self.dstep * self.omega_b_fid))
        dHdomega_cdm = ((h_high_omega_cdm - h_low_omega_cdm)
                        / (2. * self.dstep * self.omega_cdm_fid))
        dHdomega_ncdm = ((h_high_omega_ncdm - h_low_omega_ncdm)
                         / (2. * self.dstep * self.omega_ncdm_fid))
        dHdh = (h_high_h - h_low_h) / (2. * self.dstep * self.h_fid)

        dDdomega_b = ((da_high_omega_b - da_low_omega_b)
                      / (2. * self.dstep * self.omega_b_fid))
        dDdomega_cdm = ((da_high_omega_cdm - da_low_omega_cdm)
                        / (2. * self.dstep * self.omega_cdm_fid))
        dDdomega_ncdm = ((da_high_omega_ncdm - da_low_omega_ncdm)
                         / (2. * self.dstep * self.omega_ncdm_fid))
        dDdh = (da_high_h - da_low_h) / (2. * self.dstep * self.h_fid)

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

        #Fix this, very redundant, slow, error prone
        self.gen_rsd(1.01 * mu)
        self.gen_fog(1.01 * mu)
        logP_mu_high = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG) 
        P_mu_high = np.array(self.Pg) + np.array(self.RSD) + np.array(self.FOG)       
 
        self.gen_rsd(0.99 * mu)
        self.gen_fog(0.99 * mu)
        logP_mu_low = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG)
        P_mu_low = np.array(self.Pg) + np.array(self.RSD) + np.array(self.FOG)
        
        self.gen_rsd(mu)
        self.gen_rsd(mu)
        logP_mid = np.log(self.Pg) + np.log(self.RSD) + np.log(self.FOG)
        P_mid = np.array(self.Pg) + np.array(self.RSD) + np.array(self.FOG)

        dlogPdmu = (logP_mu_high - logP_mu_low) / (2. * 0.01 * mu) 
        dPdmu = (P_mu_high - P_mu_low) / (2. * 0.01 * mu)
        dlogPdk = np.zeros((len(self.z_steps), len(self.k_table)))
        dPdk = np.zeros((len(self.z_steps), len(self.k_table)))
        
        for zidx, zval in enumerate(self.z_steps): 
            for kidx, kval in enumerate(self.k_table[1:-1]):
                #Careful with this derivative definition, uneven spacing
                dlogPdk[zidx][kidx+1] = ((logP_mid[zidx][kidx+2] 
                                          - logP_mid[zidx][kidx])
                                         / (self.k_table[kidx+2]
                                            - self.k_table[kidx]))
                dPdk[zidx][kidx+1] = ((P_mid[zidx][kidx+2]
                                       - P_mid[zidx][kidx])
                                      / (self.k_table[kidx+2]
                                         - self.k_table[kidx]))

        #Careful with this substitution - is it appropriate? 
        for zidx, zval in enumerate(self.z_steps): 
            dlogPdk[zidx][0] = dlogPdk[zidx][1]
            dlogPdk[zidx][-1] = dlogPdk[zidx][-2]
            dPdk[zidx][0] = dPdk[zidx][1]
            dPdk[zidx][-1] = dPdk[zidx][-2]

        #print(np.shape(dlogPdk))
        #print(np.shape(dkdomega_b))
        #print(np.shape(dlogPdmu))
        #print(np.shape(dmudomega_b))

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
        self.dlogCOVdh = np.array(dlogPdh)
    
        #print(np.shape(self.dlogCOVdomega_b))
        #self.COV = 


    def veff(self, zidx):
        omega_m = self.omega_b_fid + self.omega_cdm_fid + self.omega_ncdm_fid
        omega_lambda = np.power(self.h_fid, 2.) - omega_m
        V = ((4. * np.pi / 3.) 
             * np.power(self.c / (100* self.h_fid), 3.)
             * np.power((self.h_fid*(self.z_steps[1] - self.z_steps[0])) 
                        / np.sqrt(omega_m 
                                  * np.power(1.+self.z_steps[zidx], 3.) 
                                  + omega_lambda), 3.))
        return V  
        
    def gen_fisher(self, mu_step): #Really messy and inefficient
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
            self.gen_rsd(muval)
            self.gen_fog(muval)
            self.gen_ap()
            self.gen_cov(muval)
            for zidx, zval in enumerate(self.z_steps): 
                for kidx, kval in enumerate(self.k_table):
                    Pm[zidx][kidx][muidx] = (self.spectra_mid[zidx].ps_table[kidx] 
                                             * self.RSD[zidx][kidx] 
                                             * self.FOG[zidx][kidx])
                                             #* self.COV[zidx][kidx]) #Need to make this term
                    dlogPdA_s[zidx][kidx][muidx] = self.dlogPdA_s[zidx][kidx]
                    dlogPdn_s[zidx][kidx][muidx] = self.dlogPdn_s[zidx][kidx]
                    dlogPdomega_b[zidx][kidx][muidx] = (self.dlogPdomega_b[zidx][kidx] 
                                                        + self.dlogRSDdomega_b[zidx][kidx]
                                                        + self.dlogAPdomega_b[zidx][kidx]
                                                        + self.dlogCOVdomega_b[zidx][kidx])
                    dlogPdomega_cdm[zidx][kidx][muidx] = (self.dlogPdomega_cdm[zidx][kidx] 
                                                          + self.dlogRSDdomega_cdm[zidx][kidx]
                                                          + self.dlogAPdomega_cdm[zidx][kidx]
                                                          + self.dlogCOVdomega_cdm[zidx][kidx])
                    dlogPdh[zidx][kidx][muidx] = (self.dlogPdh[zidx][kidx] 
                                                  + self.dlogRSDdh[zidx][kidx] 
                                                  + self.dlogFOGdh[zidx][kidx]
                                                  + self.dlogAPdh[zidx][kidx]
                                                  + self.dlogCOVdh[zidx][kidx])
                    dlogPdtau_reio[zidx][kidx][muidx] = self.dlogPdtau_reio[zidx][kidx]
                    dlogPdomega_ncdm[zidx][kidx][muidx] = (self.dlogPdomega_ncdm[zidx][kidx] 
                                                           + self.dlogRSDdomega_ncdm[zidx][kidx]
                                                           + self.dlogAPdomega_ncdm[zidx][kidx]
                                                           + self.dlogCOVdomega_ncdm[zidx][kidx])

        Pm = np.nan_to_num(Pm)
        dlogPdA_s = np.nan_to_num(dlogPdA_s)
        dlogPdn_s = np.nan_to_num(dlogPdn_s)
        dlogPdomega_b = np.nan_to_num(dlogPdomega_b)
        dlogPdomega_cdm = np.nan_to_num(dlogPdomega_cdm)
        dlogPdh = np.nan_to_num(dlogPdh)
        dlogPdtau_reio = np.nan_to_num(dlogPdtau_reio)
        dlogPdomega_ncdm = np.nan_to_num(dlogPdomega_ncdm)

        paramvec = [dlogPdA_s, dlogPdn_s, dlogPdomega_b, dlogPdomega_cdm, 
                    dlogPdh, dlogPdtau_reio, dlogPdomega_ncdm] 

        #Highly inefficient set of loops 
        for pidx1, p1 in enumerate(paramvec): 
            for pidx2, p2 in enumerate(paramvec):
                integrand = np.zeros((len(self.z_steps), len(self.k_table)))
                #integrand = np.array(integrand, dtype=np.float128) #Necessary? 
                integral = np.zeros(len(self.z_steps))
                for zidx, zval in enumerate(self.z_steps):
                    V = self.veff(zidx) 
                    for kidx, kval in enumerate(self.k_table): #Center this integral? 
                        integrand[zidx][kidx] = np.sum(mu_step *2. *np.pi
                                                       * paramvec[pidx1][zidx][kidx]
                                                       * paramvec[pidx2][zidx][kidx]
                                                       * np.power(self.k_table[kidx]/(2.*np.pi), 3.)
                                                       * np.power((self.n_densities[zidx] 
                                                                   * self.Pg[zidx][kidx])
                                                                  / (self.n_densities[zidx] 
                                                                     * self.Pg[zidx][kidx] 
                                                                     + 1.), 
                                                                  2.) 
                                                       * V 
                                                       * (1. / self.k_table[kidx]))
#                        if np.isnan(integrand[zidx][kidx]): 
#                            print("Is nan encountered, printing all components of multiply...")
#                            print("Param 1 index: ", pidx1)
#                            print("Param 2 index: ", pidx2) 
#                            print("Z index: ", zidx)
#                            print("k index: ", kidx)
#                            print(mu_step)
#                            print(np.pi)
#                            print(paramvec[pidx1][zidx][kidx])
#                            print(paramvec[pidx2][zidx][kidx])
#                            print(self.k_table[kidx])
#                            print(self.n_densities[zidx])
#                            print(Pm[zidx][kidx])
#                            print(V)
                            
                for zidx, zval in enumerate(self.z_steps): 
                    val = 0 
                    for kidx, kval in enumerate(self.k_table[:-1]): #Center this integral? 
                        val += (((integrand[zidx][kidx] + integrand[zidx][kidx+1])/2.)  
                                * (self.k_table[kidx+1]-self.k_table[kidx]))
                    integral[zidx] = val  
                fisher[pidx1][pidx2] = np.sum(integral)
                print("Fisher element (", pidx1, ", ", pidx2,") calculated...") 
        self.fisher=fisher
                


def H(omega_b, omega_cdm, omega_ncdm, h, z):
    #Returns H in units of m/s/Mpc
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
    #ndens at specific z, Pm at specific k and z 
    n = np.power((ndens * Pg) / (ndens*Pg + 1.), 2.)
    return n 

def fog(h, c, z, k, mu): 
    sigma_fog_0 = 250 #Units [km*s^-1] 
    sigma_z = 0.001
    sigma_fog = sigma_fog_0 * np.sqrt(1.+z)
    sigma_v = (1. + z) * np.sqrt((np.power(sigma_fog, 2.)/2.) + np.power(c*sigma_z, 2.))
    F = np.exp(-1. * np.power((k*mu*sigma_v) / (100.*h), 2.))
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
    f = np.power((epsilon * np.power(1.+z, 3.)) / ( epsilon * np.power(1.+z, 3.) - epsilon + 1.), gamma)
    DeltaL = (0.6 * omega_ncdm) / (omega_b + omega_cdm) 
    g = 1. + (DeltaL / 2.) * np.tanh(1. + (np.log(q) / Deltaq))
    b1tilde = 1. + b1L * g 

    R = np.power((b1tilde + np.power(mu, 2.) * f), 2.)  
    return R  


def dPs_array(low, high, step): 
    dPs = [(high[zval].ps_table - low[zval].ps_table)/(2.*step) for zval in range(len(high))]
    dlogPs = [(high[zval].log_ps_table - low[zval].log_ps_table)/(2.*step) for zval in range(len(high))] 
    return dPs, dlogPs 

def omega_ncdm(T_ncdm, m_ncdm): 
    """Returns omega_ncdm as a function of T_ncdm, m_ncdm.

    T_ncdm : relic temperature in units [K]
    m_ncdm : relic mass in units [eV]
    
    omega_ncdm : relative relic abundance. Unitless. 
    """

    omega_ncdm = np.power(T_ncdm / 1.95, 3.) * (m_ncdm / 94)
    return omega_ncdm 


def T_ncdm(omega_ncdm, m_ncdm): 
    """Returns T_ncdm as a function of omega_ncdm, m_ncdm. 

    omega_ncdm : relative relic abundance. Unitless. 
    m_ncdm : relic mass in units [eV]. 
    
    T_ncdm : relic temperature in units [K]
    """

    T_ncdm = np.power( 94. * omega_ncdm / m_ncdm, 1./3.) * 1.95
    return T_ncdm 


def domega_ncdm_dT_ncdm(T_ncdm, m_ncdm): 
    """Returns derivative of omega_ncdm wrt T_ncdm. 

    T_ncdm : relic temperature in units [K]
    m_ncdm : relic mass in units [eV]
    
    deriv : derivative of relic abundance wrt relic temp in units [K]^(-1)  
    """

    deriv = (3. * m_ncdm / 94.) * np.power(T_ncdm, 2.) * np.power(1.95, -3.) 
    return deriv

def dT_ncdm_domega_ncdm(omega_ncdm, m_ncdm): 
    """Returns derivative of T_ncdm wrt  omega_ncdm.

    omega_ncdm : relative relic abundance. Unitless. 
    m_ncdm : relic mass in units [eV]. 
    
    deriv : derivative of relic temp wrt relic abundance in units [K] 
    """

    deriv = (1.95 / 3) * np.power(94. / m_ncdm, 1./3.) * np.power(omega_ncdm, -2./3.)
    return deriv 
        
