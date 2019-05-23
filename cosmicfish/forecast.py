import numpy as np

import cosmicfish as cf 

class analysis: 

    def __init__(self, name=None, z_table=None, k_table=None, classdir=None, datastore=None): 
        self.name = name
        self.classdir = classdir
        self.datastore = datastore
        self.z_table = z_table
        self.k_table = k_table
        self.fid = []
        self.h_var = []
        self.omega_b_var = []
        self.omega_cdm_var = []
        self.tau_reio_var = []
        self.A_s_var = []
        self.n_s_var = []
        self.T_ncdm_var = []

    def generate_fiducials(self): 
        generators = list(map(lambda z : {"h" : 0.70148, 
                                            "omega_b" : 0.02226, 
                                            "omega_cdm" : 0.11271, 
                                            "N_ncdm" : 1, 
                                            "tau_reio" : 0.059888, 
                                            "A_s" : 2.2321e-9, 
                                            "n_s" : 0.96659, 
                                            "z_pk" : z}, self.z_table))
        for g in generators: 
            d = cf.io.generate_data(g, self.classdir, self.datastore).replace('/test_parameters.ini','')
            self.fid.append(cf.data.spectrum(d))
        for ps in self.fid : 
            ps.interpolate(self.k_table)
            ps.generate_power_spectrum(ps.omega_b, ps.omega_cdm, self.k_table, 
                primordial_table_generator(self.k_table, ps.A_s, ps.n_s, 0.05 * ps.h))
    
    def generate_non_fiducials(self, **kwargs): 
        for key, value in kwargs.items():          
            if key[0:3]=='rel': 
                genlist = getattr(self.fid[0], key[8:]) * np.array(value)
            if key[0:3]=='abs': 
                genlist = value
            generators = list(map(lambda z : {"h" : 0.70148,
                                            "omega_b" : 0.02226,
                                            "omega_cdm" : 0.11271,
                                            "N_ncdm" : 1,        
                                            "tau_reio" : 0.059888,        
                                            "A_s" : 2.2321e-9,        
                                            "n_s" : 0.96659,        
                                            "z_pk" : z}, self.z_table))
            for g in generators:
                if key[8:]=="h":
                    generators2 = list(map(lambda y : {"h" : y,
                                            "omega_b" : 0.02226,
                                            "omega_cdm" : 0.11271,
                                            "N_ncdm" : 1,
                                            "tau_reio" : 0.059888,
                                            "A_s" : 2.2321e-9,
                                            "n_s" : 0.96659,
                                            "z_pk" : g['z_pk']}, genlist))
                    self.h_var.append(list(map(lambda x : cf.data.spectrum(
                        cf.io.generate_data(x, self.classdir, self.datastore).replace('/test_parameters.ini','')), generators2)))
                if key[8:]=="omega_b":
                    generators2 = list(map(lambda y : {"h" : 0.70148,
                                            "omega_b" : y,
                                            "omega_cdm" : 0.11271,
                                            "N_ncdm" : 1,
                                            "tau_reio" : 0.059888,
                                            "A_s" : 2.2321e-9,
                                            "n_s" : 0.96659,
                                            "z_pk" : g['z_pk']}, genlist))
                    self.omega_b_var.append(list(map(lambda x : cf.data.spectrum(
                        cf.io.generate_data(x, self.classdir, self.datastore).replace('/test_parameters.ini','')), generators2)))
                if key[8:]=="omega_cdm":
                    generators2 = list(map(lambda y : {"h" : 0.70148,
                                            "omega_b" : 0.02226,
                                            "omega_cdm" : y,
                                            "N_ncdm" : 1,
                                            "tau_reio" : 0.059888,
                                            "A_s" : 2.2321e-9,
                                            "n_s" : 0.96659,
                                            "z_pk" : g['z_pk']}, genlist))
                    self.omega_cdm_var.append(list(map(lambda x : cf.data.spectrum(
                        cf.io.generate_data(x, self.classdir, self.datastore).replace('/test_parameters.ini','')), generators2)))
                if key[8:]=="tau_reio":
                    generators2 = list(map(lambda y : {"h" : 0.70148,
                                            "omega_b" : 0.02226,
                                            "omega_cdm" : 0.11271,
                                            "N_ncdm" : 1,
                                            "tau_reio" : y,
                                            "A_s" : 2.2321e-9,
                                            "n_s" : 0.96659,
                                            "z_pk" : g['z_pk']}, genlist))
                    self.tau_reio_var.append(list(map(lambda x : cf.data.spectrum(
                        cf.io.generate_data(x, self.classdir, self.datastore).replace('/test_parameters.ini','')), generators2)))
                if key[8:]=="T_ncdm":
                    generators2 = list(map(lambda y : {"h" : 0.70148,
                                            "omega_b" : 0.02226,
                                            "omega_cdm" : 0.11271,
                                            "N_ncdm" : 1,
                                            "tau_reio" : 0.059888,
                                            "A_s" : 2.2321e-9,
                                            "n_s" : 0.96659,
                                            "z_pk" : g['z_pk'],
                                            "T_ncdm" : y}, genlist))
                    self.T_ncdm_var.append(list(map(lambda x : cf.data.spectrum(
                        cf.io.generate_data(x, self.classdir, self.datastore).replace('/test_parameters.ini','')), generators2)))
        for idx, el1 in enumerate(self.h_var): 
            for el2 in el1: 
                el2.interpolate(self.k_table)
                el2.generate_power_spectrum(el2.omega_b, el2.omega_cdm, self.k_table,
                primordial_table_generator(self.k_table, el2.A_s, el2.n_s, 0.05 * el2.h))
        for idx, el1 in enumerate(self.omega_b_var):
            for el2 in el1:
                el2.interpolate(self.k_table)
                el2.generate_power_spectrum(el2.omega_b, el2.omega_cdm, self.k_table,
                primordial_table_generator(self.k_table, el2.A_s, el2.n_s, 0.05 * el2.h))
        for idx, el1 in enumerate(self.omega_cdm_var):
            for el2 in el1:
                el2.interpolate(self.k_table)
                el2.generate_power_spectrum(el2.omega_b, el2.omega_cdm, self.k_table,
                primordial_table_generator(self.k_table, el2.A_s, el2.n_s, 0.05 * el2.h))
        for idx, el1 in enumerate(self.tau_reio_var):
            for el2 in el1:
                el2.interpolate(self.k_table)
                el2.generate_power_spectrum(el2.omega_b, el2.omega_cdm, self.k_table,
                primordial_table_generator(self.k_table, el2.A_s, el2.n_s, 0.05 * el2.h))
        for idx, el1 in enumerate(self.T_ncdm_var):
            for el2 in el1:
                el2.interpolate(self.k_table)
                el2.generate_power_spectrum(el2.omega_b, el2.omega_cdm, self.k_table,
                primordial_table_generator(self.k_table, el2.A_s, el2.n_s, 0.05 * el2.h))               

def v_effective_table_generator(z_table, omega_m_table, omega_lambda_table, c, h, H): 
    table = list(map(lambda y, z: (4.*np.pi/3.)*np.power(c/H, 3.) 
        * np.power(np.trapz(h/np.sqrt(y*(1+z_table)**3. + z), z_table), 3.), 
        omega_m_table, omega_lambda_table))
    return table

def k_table_generator(v_eff_table, h, k_max, k_steps): 
    table = np.linspace((np.pi / h) * np.power(v_eff_table, -1./3.), k_max, k_steps)
    return table 

def primordial_table_generator(k_table, A_s, n_s, k_p): 
    table = A_s * 2. * np.power(np.pi, 2.) * np.power(k_table, -3.) * np.power(k_table / k_p, n_s - 1)
    return table  


