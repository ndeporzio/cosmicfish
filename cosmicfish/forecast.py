import numpy as np


def v_effective_table_generator(z_table, omega_m_table, omega_lambda_table, c, h, H): 
    table = list(map(lambda y, z: (4.*np.pi/3.)*np.power(c/H, 3.) 
        * np.power(np.trapz(h/np.sqrt(y*(1+z_table)**3. + z), z_table), 3.), 
        omega_m_table, omega_lambda_table))
    return table

def k_table_generator(v_eff_table, h, k_max, k_steps): 
    table = np.linspace((np.pi / h) * np.power(v_eff_table, -1./3.), k_max, k_steps)
    return table  
