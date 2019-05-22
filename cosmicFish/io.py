import os
import numpy as np
import pandas as pd






def is_data(path, 
            A_s, 
            n_s, 
            omega_b, 
            omega_cdm, 
            tau_reio, 
            h, 
            N_ncdm, 
            m_ncdm=None, 
            T_ncdm=None): 
    '''Check existence of CLASS output with specified parameters.'''
    #Either both m_ncdm and T_ncdm defined, or neither 
    if (m_ncdm is None) ^ (T_ncdm is None): 
        raise Exception("Must define m_ncdm with T_ncdm, and vice versa.") 
        
    check = 0
    text = open(correct_path(path)).read()
    if "A_s = {}".format(A_s).replace("e-0","e-") not in text: 
        print("A_s = {}".format(A_s))
        check += 1
    if "n_s = {}".format(n_s) not in text:
        print("n_s = {}".format(n_s))
        check += 1
    if "omega_b = {}".format(omega_b) not in text:
        print("omega_b = {}".format(omega_b))
        check += 1
    if "omega_cdm = {}".format(omega_cdm) not in text:
        print("omega_cdm = {}".format(omega_cdm))
        check += 1
    if "tau_reio = {}".format(tau_reio) not in text:
        print("tau_reio = {}".format(tau_reio))
        check += 1
    if "h = {}".format(h) not in text:
        print("h = {}".format(h))
        check += 1
    if "N_ncdm = {}".format(N_ncdm) not in text:
        print("N_ncdm = {}".format(N_ncdm))
        check += 1
    if m_ncdm is not None: 
        if "m_ncdm = {}".format(m_ncdm) not in text:
            print("m_ncdm = {}".format(m_ncdm))
            check += 1
        if "T_ncdm = {}".format(T_ncdm) not in text:
            print("T_ncdm = {}".format(T_ncdm))
            check += 1
    if check > 0: 
        print(check)
        return False 
    else:   
        return True 
        
def correct_path(pathname): 
    """Expands and checks file paths."""
    fix1 = os.path.expanduser(pathname)
    fix2 = os.path.expandvars(fix1)
    fix3 = os.path.normpath(fix2)
    fix4 = os.path.abspath(fix3)
    if not os.path.isfile(fix4): 
        raise Exception("Invalid path: {}".format(fix4))
    return fix4    
