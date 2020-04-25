import os
import time
import random
import subprocess 
import numpy as np
import pandas as pd

import cosmicfish as cf

def install_class(classdir): 
    '''Installs CLASS to specified directory.
    
    Args: 
        classdir (str) : path to directory to install CLASS at.

    Returns: 
        No values returned. 
    '''
    print("Installing CLASS, please wait up to 2 minutes.")
    classdir = correct_path(classdir)
    os.chdir(classdir)
    os.system('git clone git@github.com:lesgourg/class_public.git')
    # subprocess.check_output(
    #     'git clone git@github.com:lesgourg/class_public.git', 
    #     shell=True)
    os.system('mv class_public class') 
    os.chdir(os.path.join(classdir, 'class'))
    os.system('make clean')
    os.system('make -j class')
    # subprocess.check_output('make -j class', shell=True) 
    print("To check for proper installation, run: '$ ./class explanatory.ini'") 

def generate_data(fiducial, classdir, datastore, **kwargs):
    '''Generates a CLASS dataset with specified parameters.

    Only generates a dataset if it doesn't already exist. If a dataset already
    exists, only the path to that dataset is identified and returned. 
    
    Args: 
        fiducial (dict) : fiducial cosmology that will be overwritten 
            to the default .ini file to generate CLASS spectra. If 
            additional entries are passed as keyword arguments, 
            they will take precedence over the same values listed
            here. 
        classdir (str) : path to directory containing an installed
            build of CLASS. 
        datastore (str) : path to directory where you would like 
            output generated by CLASS to be stored.  
    Returns: 
        Path to the 'test_parameter.ini' file generated according to the 
        specified cosmology. 
    '''
    modify = {}
    #
    # Add fiducial cosmology
    #
    for key, value in fiducial.items():
        if (key in cf.CLASSVARS) and (key not in kwargs):
            modify[key] = value
    #
    # Add to/overwrite fiducial cosmology with specified arguments.
    # 
    for key, value in kwargs.items():
        if (key in cf.CLASSVARS): 
            modify[key] = value

    check, existingdata = check_data(datastore, **modify)

    if check==False:
        datastore = correct_path(datastore)
        classdir = correct_path(classdir)
        tstamp=str(time.time()) 
        start_ini = os.path.join(config_directory(),"prime.ini")
        end_ini = os.path.join(config_directory(), tstamp+".ini")
        os.system('cp ' + start_ini + ' ' + end_ini)
        for key, value in modify.items():
            if (key=='m_ncdm') or (key=='T_ncdm') or (key=='deg_ncdm'):
                if modify['N_ncdm']==3: # Degenerate neutrinos
                    print('#'+key+'-->'
                          + key + ' = ' + str(value) 
                          + ", " + str(value) 
                          + ", " + str(value))
                    replace_text(end_ini, '#'+key, key + ' = ' + str(value) 
                                                   + ", " + str(value) 
                                                   + ", " + str(value))
                if modify['N_ncdm']==1: # Light relics
                    print('#'+key+'-->'+ key + ' = ' + str(value))
                    replace_text(end_ini, '#'+key, key + ' = ' + str(value)) 
            else: 
                print('#'+key+'-->'+ key + ' = ' + str(value))
                replace_text(end_ini, '#'+key, key + ' = ' + str(value))
        newdatapath = os.path.join(datastore, 
                                   str(time.time())
                                       +"{0:.6f}".format(random.random()))
        os.system('mkdir ' + newdatapath)
        os.system('cd ' + classdir)
        os.chdir(classdir)
        os.system('pwd')
        os.system('./class ' + end_ini)
        print("Dataset generated at: " + newdatapath)
        os.system('mv ' 
                  + os.path.join(classdir, 'output') 
                  + '/* ' + newdatapath)
        os.system('rm ' + end_ini)
        datasetpath = os.path.join(newdatapath, 'test_parameters.ini') 
    else: 
        datasetpath = existingdata 
    return datasetpath  
                

def check_data(datastore, **kwargs):             
    '''Checks if CLASS spectrum exists with specified parameters.

    Args: 
        datastore (str) : path to directory where you would like                
            output generated by CLASS to be stored.
        **kwargs (dict) : values to pass to `is_data` function 
            to check if CLASS output already exist for those
            values. 
    Returns: 
        Two values are returned. 
        
        check (bool) : `True` if data output already exists for the specified 
            input cosmology values. `False` otherwise.  

        preexistingdata (str) : if `check` is `True`, this is the path to the 
            corresponding dataset. Else, this value is `None`. 
    '''
    datastore = correct_path(datastore)
    check = False
    preexistingdata = None

    # Look in each subdirectory of 'datastore'
    for datapath in next(os.walk(datastore))[1]:
        # Look at the contained 'test_parameters.ini' file. 
        testfile = os.path.join(datastore, datapath, 'test_parameters.ini') 
        if is_data(testfile, **kwargs): 
            print("Dataset already exists at: " 
                  + os.path.join(datastore, datapath))
            check = True 
            preexistingdata = testfile 
            break 
    return (check, preexistingdata)         
         
def is_data(path, **kwargs): 
    '''Returns true if datafile exists w/ correct parameter values.

    Args:                                                                       
        path (str) : path to the `test_parameter.ini` file you want to check
            for values matching those provided to **kwargs.                              
        **kwargs (dict) : values to check if CLASS output matches. 
                           
    Returns:                                                                    
        `True` is returned if the target .ini file contains *at least* the 
        values specified in **kwargs. Note, the  target .ini file may contain
        additional valus not specified in **kwargs and `True` will still be 
        returned. Else, 'False' is returned. 
    ''' 
    check = 0
    try: 
        correct_path(path)
    except: 
        print("Data was not generated by CLASS for this run!" +
              "A directory exists without data files.") 
        return False
    else: 
        # Open the data file                                                   
        text = open(correct_path(path)).read()                                 
        # Check if it contains provided parameter values                       
        for key, val in kwargs.items():
            if (key=='m_ncdm') or (key=='T_ncdm') or (key=='deg_ncdm'):
                if (kwargs['N_ncdm']==3): 
                    test = ('\n'
                            + key 
                            + ' = '
                            + (int(kwargs['N_ncdm'])-1) * (str(val) + ', ')
                            + str(val)
                            + '\n')                                         
                else: 
                    test = ('\n' + key + ' = ' + str(val) + '\n')
            else: 
                test = (key+' = '+str(val)+'\n')     
            if test not in text:
                check += 1                                                     
        if check > 0:                                                          
            return False                                                       
        else:                                                                  
            return True 

def correct_path(pathname): 
    """Expands and checks exisence of file paths.

    Args: 
        pathname (str) : Absolute or relative path. 

    Returns: 
        Absolute path with no special characters (e.g. tilde). 
    """
    fix1 = os.path.expanduser(pathname)
    fix2 = os.path.expandvars(fix1)
    fix3 = os.path.normpath(fix2)
    fix4 = os.path.abspath(fix3)
    if not (os.path.isfile(fix4) or os.path.isdir(fix4)): 
        raise Exception("Invalid path: {}".format(fix4))
    return fix4    

def config_directory(): 
    '''Returns path of data directory in package distribution.
    
    Returns: 
        Path to the `config` directory of the local cosmicfish package 
        distribution. 
    '''
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    path = os.path.join(path, 'config')
    return path

def priors_directory():
    '''Returns path of priors directory in package distribution.                 
                                                                                
    Returns:                                                                    
        Path to the `priors` directory of the local cosmicfish package          
        distribution.                                                           
    '''                                                                         
    path = os.path.abspath(__file__)                                            
    path = os.path.dirname(path)                                                
    path = os.path.join(path, 'priors')                                         
    return path 

def replace_text(document, old_text, new_text):
    '''Replace text in a specified file.

    Args: 
        document (str) : Path to file to be edited. 
        
        old_text (str) : String in original file to be replaced. 

        new_text (str) : String to replace `old_text` with in the file.

    Returns: 
        Nothing returned. 
    '''
    with open(document) as f:
        new_doc=f.read().replace(old_text, new_text)
    with open(document, "w") as f:
        f.write(new_doc)

def citation(): 
    path = os.path.abspath(__file__)
    path = os.path.join(path, "..") 
    path = os.path.abspath(path)                                            
    path = os.path.dirname(path)                                                
    path = os.path.join(path, 'CITATION')

    with open(path, 'r') as text: 
        for line in text: 
            print(line)  

def makedirectory(path): 
    try: 
        os.stat(path)
    except:
        os.makedirs(path, exist_ok=True)
    return 

if __name__ == '__main__':
    # Actions to perform only if this module, 'io.py', is called
    # directly (e.g. '$ python io.py'). These actions aren't 
    # performed if the module is imported by another module.
    print("End __main__ execution of 'io.py'...")
