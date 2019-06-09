import numpy as np
import pandas as pd

from .io import correct_path

class spectrum: 
   
    def __init__(self, datadirectory, z_table): 

        #While this spectrum is for a specific z value, how we bin z
        #in analysis determines range of k table
        self.z_table = z_table 
        self.A_s = None
        self.n_s = None
        self.omega_b = None
        self.omega_cdm = None
        self.tau_reio = None
        self.h = None
        self.m_ncdm = None
        self.T_ncdm = None
        self.z_pk = None 
        self.k_pivot = None
        self.rawdata = None
        self.b_interp_table = None
        self.cdm_interp_table = None
        self.prim_table = None
        self.ps_table = None
        self.log_ps_table = None
        self.class_pk = None 
        self.dataconfig = correct_path(datadirectory + "/test_parameters.ini")
        self.datapath = correct_path(datadirectory + "/test_tk.dat")

        #Import data 
        self.input()

        #Derive k_table 
        self.v_eff = gen_v_eff(self.h, self.omega_b, 
                               self.omega_cdm, self.z_table, 
                               self.T_ncdm, self.m_ncdm, c=2.9979e8)
        self.k_table = gen_k_table(self.v_eff, self.h, k_max=0.1, k_steps=100)

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
        self.b_interp_table = np.interp(self.k_table, self.rawdata['k (h/Mpc)'], self.rawdata['d_b'])
        self.cdm_interp_table = np.interp(self.k_table, self.rawdata['k (h/Mpc)'], self.rawdata['d_cdm']) 

    def gen_primordial_table(self):
        table = self.A_s * 2. * np.power(np.pi, 2.) * np.power(self.k_table, -3.) * np.power(self.k_table / self.k_pivot, self.n_s - 1)
        self.prim_table=table
 
    def gen_power_spectrum(self):
        fb = self.omega_b / (self.omega_b + self.omega_cdm)
        fcdm = self.omega_cdm / (self.omega_b + self.omega_cdm)
        table = np.power(self.b_interp_table*fb + self.cdm_interp_table*fcdm, 2.) * self.prim_table
        self.ps_table = table
        self.log_ps_table = np.log(table)

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
    H = 1000. * 100. * h
    if T_ncdm is not None:
        omega_chi = np.power(T_ncdm/1.95, 3.) * (m_ncdm/94.)
    else: 
        omega_chi = 0
    omega_m = omega_b + omega_cdm + omega_chi
    omega_lambda = np.power(h, 2.) - omega_m
    v_eff = ((4.*np.pi/3.)*np.power(c/H, 3.)
              * np.power(np.trapz(h/np.sqrt(omega_m*(1+z_table)**3. + omega_lambda), z_table), 3.))
    return v_eff

def gen_k_table(v_eff, h, k_max, k_steps):
    k_table = np.linspace((np.pi / h) * np.power(v_eff, -1./3.), k_max, k_steps)
    return k_table        
