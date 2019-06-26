import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

        #Generate spectra at each z for fid cosmo
        self.spectra_mid = [cf.spectrum(cf.generate_data(dict(self.fid,      
                                                              **{'z_pk' : j}),
                                                         self.classdir,       
                                                         self.datastore).replace('/test_parameters.ini',''),
                                        self.z_steps) for j in self.z_steps]
        self.k_table = spectra_mid[0].k_table #All spectra should have same k_table

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
        self.dPdn_s, self.dlogPdn_s = dPs_array(self.n_s_low, self.n_s_high, self.fid['n_s']*self.dstep) #Replace w/ analytic result
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
                     kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)]   
        self.dlogRSDdomega_b = (np.log([[rsd((1.+self.dstep)*self.omega_b_fid, self.omega_cdm_fid, self.omega_ncdm_fid, self.h_fid, kfs_table[zidx], zval, mu, kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)]) - np.log([[rsd((1.-self.dstep)*self.omega_b_fid, self.omega_cdm_fid, self.omega_ncdm_fid, self.h_fid, kfs_table[zidx], zval, mu, kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)])) / (2.*self.dstep*self.omega_b_fid)
        self.dlogRSDdomega_cdm = (np.log([[rsd(self.omega_b_fid, (1.+self.dstep)*self.omega_cdm_fid, self.omega_ncdm_fid, self.h_fid, kfs_table[zidx], zval, mu, kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)]) - np.log([[rsd(self.omega_b_fid, (1.-self.dstep)*self.omega_cdm_fid, self.omega_ncdm_fid, self.h_fid, kfs_table[zidx], zval, mu, kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)])) / (2.*self.dstep*self.omega_cdm_fid)
        self.dlogRSDdomega_ncdm = (np.log([[rsd(self.omega_b_fid, self.omega_cdm_fid, (1.+self.dstep)*self.omega_ncdm_fid, self.h_fid, kfs_table[zidx], zval, mu, kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)]) - np.log([[rsd(self.omega_b_fid, self.omega_cdm_fid, (1.-self.dstep)*self.omega_ncdm_fid, self.h_fid, kfs_table[zidx], zval, mu, kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)])) / (2.*self.dstep*self.omega_ncdm_fid)
        self.dlogRSDdh = (np.log([[rsd(self.omega_b_fid, self.omega_cdm_fid, self.omega_ncdm_fid, (1.+self.dstep)*self.h_fid, kfs_table[zidx], zval, mu, kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)]) - np.log([[rsd(self.omega_b_fid, self.omega_cdm_fid, self.omega_ncdm_fid, (1.-self.dstep)*self.h_fid, kfs_table[zidx], zval, mu, kval) for kval in self.k_table] for zidx, zval in enumerate(self.z_steps)])) / (2.*self.dstep*self.h_fid)
            

    def gen_fog(self, mu):
        self.FOG = [[fog(self.h_fid, self.c, zval, kval, mu) for kval in self.k_table] for zval in self.z_steps]
        self.dlogFOGdh = ((np.log([[fog((1.+self.dstep)*self.h_fid, self.c, zval, kval, mu) for kval in self.k_table] for zval in self.z_steps]) 
                    - np.log([[fog((1.-self.dstep)*self.h_fid, self.c, zval, kval, mu) for kval in self.k_table] for zval in self.z_steps]))
                    / (2.*self.dstep*self.h_fid)) 

    def gen_fisher(self, mu_step): #Really messy and inefficient
        fisher = np.zeros((7, 7))
        mu_vals = np.arange(-1., 1., mu_step)
        Pm = np.zeros((len(self.z_steps), len(self.k_table, len(mu_vals))))
        dlogPdA_s = np.zeros((len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdn_s = np.zeros((len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdomega_b = np.zeros((len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdomega_cdm = np.zeros((len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdh = np.zeros((len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdtau_reio = np.zeros((len(self.z_steps), len(self.k_table), len(mu_vals)))
        dlogPdomega_ncdm = np.zeros((len(self.z_steps), len(self.k_table), len(mu_vals)))

        
        for muidx, muval in mu_vals: 
            gen_rsd(muval)
            gen_fog(muval)
            for zidx, zval in enumerate(self.z_steps): 
                for kidx, kval in enumerate(self.k_table):
                    Pm[zidx][kidx][muidx] = self.spectra_mid[zidx].ps_table[k_idx] * self.RSD[zidx][kidx] * self.FOG[zidx][kidx]
                    dlogPdA_s[zidx][kidx][muidx] = self.dlogPdA_s[zidx][kidx]
                    dlogPdn_s[zidx][kidx][muidx] = self.dlogPdn_s[zidx][kidx]
                    dlogPdomega_b[zidx][kidx][muidx] = self.dlogPdomega_b[zidx][kidx] + self.dlogRSDdomega_b[zidx][kidx]
                    dlogPdomega_cdm[zidx][kidx][muidx] = self.dlogPdomega_cdm[zidx][kidx] + self.dlogRSDdomega_cdm[zidx][kidx]
                    dlogPdh[zidx][kidx][muidx] = self.dlogPdh[zidx][kidx] + self.dlogRSDdh[zidx][kidx] + self.dlogFOGdh[zidx][kidx]
                    dlogPdtau_reio[zidx][kidx][muidx] = self.dlogPdtau_reio[zidx][kidx]
                    dlogPdomega_ncdm[zidx][kidx][muidx] = self.dlogPdomega_ncdm[zidx][kidx] + self.dlogRSDdomega_ncdm[zidx][kidx]

        for zidx, zval in enumerate(self.z_steps): 
            fisher_z = np.zeros((7, 7))
               #TEST 

        def veff(self, zidx): 
            omega_m = self.omega_b_fid + self.omega_cdm_fid + self.omega_ncdm_fid
            omega_lambda = np.power(self.h_fid, 2.) - omega_m
            return ((4. * np.pi / 3.) * np.power(self.c / (100* self.h_fid), 3.) 
                    * np.power((self.h_fid*(self.z_steps[1] - self.z_steps[0])) 
                                / np.sqrt(omega_m * np.power(1.+self.z_steps[zidx], 3.) + omega_lambda), 3.)) 
                    


     
def neff(ndens, Pm): 
    #ndens at specific z, Pm at specific k and z 
    n = np.power((ndens * Pm) / (ndens*Pm + 1.), 2.)
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
    dPs = [(high[k].ps_table - low[k].ps_table)/(2.*step) for k in range(len(high))]
    dlogPs = [(high[k].log_ps_table - low[k].log_ps_table)/(2.*step) for k in range(len(high))] 
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
        
