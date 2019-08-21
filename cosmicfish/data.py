import numpy as np
import pandas as pd
import scipy
import cosmicfish as cf 

from .io import correct_path

class spectrum: 
   
    def __init__(self, datadirectory, z_table, fsky=None): 

        # While this spectrum is for a specific z value, how we bin z
        # in analysis determines V and thus range of k table. 
        # 
        # Instantiation variables
        #
        self.dataconfig = correct_path(datadirectory + "/test_parameters.ini")
        self.datapath = correct_path(datadirectory + "/test_tk.dat")
        self.background_data = correct_path(datadirectory 
                                            + "/test_background.dat")
        self.z_table = z_table # Unitless 
        self.fsky = fsky # Unitless  
        #
        # Values read from CLASS output. 
        #
        self.z_pk = None # Unitless
        self.A_s = None # Unitless  
        self.n_s = None # Unitless
        self.omega_b = None # Unitless 
        self.omega_cdm = None # Unitless
        self.tau_reio = None
        self.h = None # Unitless
        self.m_ncdm = None # Units ov [eV]
        self.T_ncdm = None # Units of [T_cmb]
        self.N_ncdm = None # Unitless
        self.T_cmb = None # Units of [K] 
        self.k_pivot = None # Units [Mpc^-1]
        #
        # Values interptreted from CLASS output.
        #
        self.rawdata = None
        self.b_interp_table = None # Unitless 
        self.cdm_interp_table = None # Unitless 
        self.prim_table = None # Units of [Mpc^3]
        self.ps_table = None # Units of [Mpc^3]
        self.log_ps_table = None # Units of log([Mpc^3])
        self.class_pk = None # Units of [Mpc^3]
        self.D_table = None # Unitless 
        # 
        # Import data 
        # 
        self.input()
        self.growthfactor()
        #
        #Derive k_table 
        #
        self.V = cf.gen_V(
            self.h, 
            self.omega_b, 
            self.omega_cdm, 
            self.z_table, 
            self.N_ncdm, 
            self.T_ncdm, 
            self.m_ncdm, 
            c=cf.C, 
            fsky=self.fsky) #Units [Mpc^3]
        self.k_table = cf.gen_k_table(
            self.V, 
            self.h, 
            k_max=cf.K_MAX, 
            k_steps=100) #Units [Mpc^-1]
        #
        #Derive power spectrum 
        #
        self.interpolate()
        self.gen_primordial_table()
        self.gen_power_spectrum()

    def growthfactor(self): 
        rawdata = pd.read_csv(self.background_data, 
                              delim_whitespace=True, 
                              skipinitialspace=True, 
                              skiprows=4, 
                              header=None, 
                              usecols=[0,20], 
                              names = ["z", "D"])
        interpolator = scipy.interpolate.interp1d(rawdata['z'], rawdata['D'])
        self.D_table = interpolator(self.z_table)
        #print(self.D_table) 

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
                    self.m_ncdm = float(line.split(' = ')[1][0:4])
                if line.startswith("T_ncdm"):
                    self.T_ncdm = float(line.split(' = ')[1])
                if line.startswith("N_ncdm"):
                    self.N_ncdm = float(line.split(' = ')[1])
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
        self.class_pk = pd.read_csv(self.datapath.replace("/test_tk.dat", 
                                                          "/test_pk.dat"), 
                                    skiprows=4,
                                    skipinitialspace=True,
                                    #sep="     ", 
                                    delim_whitespace=True,
                                    header=None,
                                    engine="python",
                                    names=["k (h/Mpc)", "P (Mpc/h)^3"])
        

    def interpolate(self):
        self.b_interpolator = scipy.interpolate.interp1d(
                             self.h * self.rawdata['k (h/Mpc)'], 
                             self.rawdata['d_b'])
        self.cdm_interpolator = scipy.interpolate.interp1d(
                               self.h * self.rawdata['k (h/Mpc)'], 
                               self.rawdata['d_cdm'])
        self.b_interp_table = self.b_interpolator(self.k_table)
        self.cdm_interp_table = self.cdm_interpolator(self.k_table) 

    def gen_primordial_table(self):
        table = (self.A_s 
                 * 2. 
                 * np.power(np.pi, 2.) 
                 * np.power(self.k_table, -3.) 
                 * np.power(self.k_table / self.k_pivot, self.n_s - 1))
        self.prim_table=table #Units of [Mpc^3] ?? 
 
    def gen_power_spectrum(self):
        fb = self.omega_b / (self.omega_b + self.omega_cdm) # Unitless
        fcdm = self.omega_cdm / (self.omega_b + self.omega_cdm) # Unitless
        table = (np.power(self.b_interp_table*fb 
                         + self.cdm_interp_table*fcdm, 2.) 
                 * self.prim_table)
        self.ps_table = table # Units of [Mpc^3]
        self.log_ps_table = np.log(table) # Units of log[Mpc^3]

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
        print('volume = ', self.V)

if __name__ == '__main__':   
    # Actions to perform only if this module, 'data.py', is called
    # directly (e.g. '$ python data.py'). These actions aren't 
    # performed if the module is imported by another module.      
    print("End __main__ execution of 'data.py'...")  
