import numpy as np
import pandas as pd

from .io import correct_path

class spectrum: 
   
    def __init__(self, datadirectory): 
        self.A_s = None
        self.n_s = None
        self.omega_b = None
        self.omega_cdm = None
        self.tau_reio = None
        self.h = None
        self.m_ncdm = None
        self.T_ncdm = None
        self.rawdata = None
        self.b_interp_table = None
        self.cdm_interp_table = None
        self.dataconfig = correct_path(datadirectory + "/test_parameters.ini")
        self.datapath = correct_path(datadirectory + "/test_tk.dat")
        self.input()

    def input(self): 
        with open(self.dataconfig) as f: 
            for line in f: 
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
        self.rawdata = pd.read_csv(self.datapath, 
                                    skiprows=11, 
                                    skipinitialspace=True, 
                                    #sep="     ", 
                                    delim_whitespace=True,
                                    usecols=[0, 2, 3], 
                                    header=None, 
                                    engine="python",
                                    names=["k (h/Mpc)", "d_b", "d_cdm"])

    def interpolate(self, k_table):
        self.b_interp_table = np.interp(k_table, self.rawdata['k (h/Mpc)'], self.rawdata['d_b'])
        self.cdm_interp_table = np.interp(k_table, self.rawdata['k (h/Mpc)'], self.rawdata['d_cdm']) 

        
    
            



