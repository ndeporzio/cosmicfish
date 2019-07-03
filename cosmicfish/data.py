import numpy as np
import pandas as pd

from .io import correct_path

class spectrum: 
   
    def __init__(self, datadirectory, z_table): 

        #While this spectrum is for a specific z value, how we bin z
        #in analysis determines range of k table
        self.z_table = z_table 
        self.A_s = None #Unitless ?? 
        self.n_s = None #Unitless
        self.omega_b = None #Unitless ??
        self.omega_cdm = None #Unitless ??
        self.tau_reio = None
        self.h = None #Unitless
        self.m_ncdm = None #Units ov [eV]
        self.T_ncdm = None #Units of [T_cmb]
        self.T_cmb = None #Units of [K] 
        self.z_pk = None #Unitless
        self.k_pivot = None #Units [Mpc^-1]
        self.rawdata = None
        self.b_interp_table = None #Unitless ??
        self.cdm_interp_table = None #Unitless ??
        self.prim_table = None #Units of [Mpc^3]
        self.ps_table = None #Units of [Mpc^3]
        self.log_ps_table = None
        self.class_pk = None 
        self.dataconfig = correct_path(datadirectory + "/test_parameters.ini")
        self.datapath = correct_path(datadirectory + "/test_tk.dat")

        #Import data 
        self.input()

        #Derive k_table 
        self.v_eff = gen_v_eff(self.h, self.omega_b, 
                               self.omega_cdm, self.z_table, 
                               self.T_ncdm, self.m_ncdm, c=2.9979e8) #Units [Mpc^3]
        self.k_table = gen_k_table(self.v_eff, self.h, k_max=0.1, k_steps=100) #Units [Mpc^-1]

        #Derive power spectrum 
        self.interpolate()
        self.gen_primordial_table()
        self.gen_power_spectrum()

    def input(self): 
        with open(self.dataconfig) as f: 
            for line in f:
                if line.startswith("z_pk"):
                    self.z_pk = float(line.split(' = ')[1]) 
                if line.startswith("A_s"): 
                    self.A_s = float(line.split(' = ')[1])
                if line.startswith("n_s"):
                    self.n_s = float(line.split(' = ')[1])
                if line.startswith("omega_b"):
                    self.omega_b = float(line.split(' = ')[1])
                if line.startswith("omega_cdm"):
                    self.omega_cdm = float(line.split(' = ')[1])
                if line.startswith("tau_reio"):
                    self.tau_reio = float(line.split(' = ')[1])
                if line.startswith("h"):
                    self.h = float(line.split(' = ')[1])
                if line.startswith("m_ncdm"):
                    self.m_ncdm = float(line.split(' = ')[1])
                if line.startswith("T_ncdm"):
                    self.T_ncdm = float(line.split(' = ')[1])
                if line.startswith("T_cmb"):
                    self.T_cmb = float(line.split(' = ')[1])
                if line.startswith("k_pivot"): 
                    self.k_pivot = float(line.split(' = ')[1]) 
        self.rawdata = pd.read_csv(self.datapath, 
                                    skiprows=11, 
                                    skipinitialspace=True, 
                                    #sep="     ", 
                                    delim_whitespace=True,
                                    usecols=[0, 2, 3], 
                                    header=None, 
                                    engine="python",
                                    names=["k (h/Mpc)", "d_b", "d_cdm"])
        self.class_pk = pd.read_csv(self.datapath.replace("/test_tk.dat", "/test_pk.dat"), 
                                    skiprows=4,
                                    skipinitialspace=True,
                                    #sep="     ", 
                                    delim_whitespace=True,
                                    header=None,
                                    engine="python",
                                    names=["k (h/Mpc)", "P (Mpc/h)^3"])
        

    def interpolate(self):
        self.b_interp_table = np.interp(self.k_table, self.h*self.rawdata['k (h/Mpc)'], self.rawdata['d_b'])
        self.cdm_interp_table = np.interp(self.k_table, self.h*self.rawdata['k (h/Mpc)'], self.rawdata['d_cdm']) 

    def gen_primordial_table(self):
        table = self.A_s * 2. * np.power(np.pi, 2.) * np.power(self.k_table, -3.) * np.power(self.k_table / self.k_pivot, self.n_s - 1)
        self.prim_table=table #Units of [Mpc^3] ?? 
 
    def gen_power_spectrum(self):
        fb = self.omega_b / (self.omega_b + self.omega_cdm) #Unitless
        fcdm = self.omega_cdm / (self.omega_b + self.omega_cdm) #Unitless
        table = np.power(self.b_interp_table*fb + self.cdm_interp_table*fcdm, 2.) * self.prim_table
        self.ps_table = table #Units of [Mpc^3] ??
        self.log_ps_table = np.log(table) #Units of log[Mpc^3]

    def print_cosmo(self): 
        print('z_pk = ', self.z_pk)
        print('A_s = ', self.A_s)
        print('n_s = ', self.n_s)
        print('omega_b = ', self.omega_b)
        print('omega_cdm = ', self.omega_cdm)
        print('tau_reio = ', self.tau_reio)
        print('h = ', self.h)
        print('m_ncdm = ', self.m_ncdm)
        print('T_ncdm = ', self.T_ncdm)
        print('k_pivot = ', self.k_pivot)
        print('v_eff = ', self.v_eff)

def gen_v_eff(h, omega_b, omega_cdm, z_table, T_ncdm=None, m_ncdm=0, c=2.9979e8):
    # T_ncdm in units of [K]
    # m_ncdm in units of [eV]
    # c in units of [m*s^-1] 
    # returns v_eff in units [Mpc^3]
    H = 1000. * 100. * h #H has units of [m*s^-1*Mpc^-1]
    if T_ncdm is not None:
        omega_chi = np.power(T_ncdm/1.95, 3.) * (m_ncdm/94.)
    else: 
        omega_chi = 0
    omega_m = omega_b + omega_cdm + omega_chi #Unitless
    omega_lambda = np.power(h, 2.) - omega_m #Unitless

    zmax = z_table[-1]
    zmin = z_table[-2]
    zsteps = 100.
    dz = zmin / zsteps
    z_table_max = np.arange(0., zmax, dz)
    z_table_min = np.arange(0., zmin, dz)
    z_integrand_max = (h * dz) /  np.sqrt(omega_m * np.power(1. + z_table_max, 3.) + omega_lambda)
    z_integrand_min = (h * dz) /  np.sqrt(omega_m * np.power(1. + z_table_min, 3.) + omega_lambda)
    z_integral_max = np.sum(z_integrand_max)
    z_integral_min = np.sum(z_integrand_min)
    v_max = ((4. * np.pi / 3.)
         * np.power(c / (1000. * 100. * h), 3.)
         * np.power(z_integral_max, 3.))
    v_min = ((4. * np.pi / 3.)
         * np.power(c / (1000. * 100. * h), 3.)
         * np.power(z_integral_min, 3.))
    v = v_max - v_min
    #print(v * np.power(h, 3.) / 1e9)
    return v #Units [Mpc^3]

def gen_k_table(v_eff, h, k_max, k_steps):
    # v_eff in units of [Mpc^3]
    # returns k_table in units [Mpc^-1]
    k_table = np.linspace((np.pi / h) * np.power(v_eff, -1./3.), k_max, k_steps)
    return k_table #Units [Mpc^-1]        
