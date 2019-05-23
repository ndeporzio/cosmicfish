import os
import time
import random 
import numpy as np
import pandas as pd


def generate_data(fiducial, classdir, datastore, **kwargs):
    '''Generates a CLASS dataset with specified parameters.'''
    if check_data(datastore, **kwargs)==False:
        datastore = correct_path(datastore)
        classdir = correct_path(classdir) 
        start_ini = os.path.join(config_directory(),"prime.ini")
        end_ini = os.path.join(config_directory(),"in.ini")
        os.system('cp ' + start_ini + ' ' + end_ini)
        modify = {}
        for key, value in fiducial.items(): 
            if key not in kwargs.items(): 
                modify[key] = value
        for key, value in kwargs.items(): 
            modify[key] = value
        for key, value in modify.items():
            print('#'+key+'-->'+ key + ' = ' + str(value))
            replace_text(end_ini, '#'+key, key + ' = ' + str(value))
        newdatapath = os.path.join(datastore, str(time.time())+"{0:.6f}".format(random.random()))
        os.system('mkdir ' + newdatapath)
        os.system('cd ' + classdir)
        os.chdir(classdir)
        print(os.getcwd())
        os.system('pwd')
        os.system('./class ' + end_ini)
        print("Dataset generated at: " + newdatapath)
        os.system('mv ' + os.path.join(classdir, 'output') + '/* ' + newdatapath)
        os.system('rm ' + end_ini)
    return None  
                

def check_data(datastore, **kwargs):             
    '''Checks if CLASS spectrum exists with specified parameters.'''
    return False        
         

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
    if not (os.path.isfile(fix4) or os.path.isdir(fix4)): 
        raise Exception("Invalid path: {}".format(fix4))
    return fix4    

def config_directory(): 
    """Returns absolute path of data directory in package distribution."""
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    path = os.path.join(path, 'config')
    return path

def replace_text(document, old_text, new_text):
    '''Replace text in a specified file.'''
    with open(document) as f:
        new_doc=f.read().replace(old_text, new_text)
    with open(document, "w") as f:
        f.write(new_doc)
