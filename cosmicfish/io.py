import os
import time
import random
import subprocess 
import numpy as np
import pandas as pd

def install_class(classdir): 
    '''Installs CLASS to specified directory.'''
    print("Installing CLASS, please wait up to 2 minutes.")
    classdir = correct_path(classdir)
    os.chdir(classdir)
    os.system('git clone git@github.com:lesgourg/class_public.git')
    #subprocess.check_output('git clone git@github.com:lesgourg/class_public.git', shell=True)
    os.system('mv class_public class') 
    os.chdir(os.path.join(classdir, 'class'))
    os.system('make clean')
    os.system('make -j class')
    #subprocess.check_output('make -j class', shell=True) 
    print("To check for proper installation, run: '$ ./class explanatory.ini'") 

def generate_data(fiducial, classdir, datastore, **kwargs):
    '''Generates a CLASS dataset with specified parameters.'''
    modify = {}
    for key, value in fiducial.items():
        if key not in kwargs.items():
            modify[key] = value
    for key, value in kwargs.items():
        modify[key] = value
    
    check, existingdata = check_data(datastore, **modify)
    if check==False:
        datastore = correct_path(datastore)
        classdir = correct_path(classdir) 
        start_ini = os.path.join(config_directory(),"prime.ini")
        end_ini = os.path.join(config_directory(),"in.ini")
        os.system('cp ' + start_ini + ' ' + end_ini)
        for key, value in modify.items():
            print('#'+key+'-->'+ key + ' = ' + str(value))
            replace_text(end_ini, '#'+key, key + ' = ' + str(value))
        newdatapath = os.path.join(datastore, str(time.time())+"{0:.6f}".format(random.random()))
        os.system('mkdir ' + newdatapath)
        os.system('cd ' + classdir)
        os.chdir(classdir)
        os.system('pwd')
        os.system('./class ' + end_ini)
        print("Dataset generated at: " + newdatapath)
        os.system('mv ' + os.path.join(classdir, 'output') + '/* ' + newdatapath)
        os.system('rm ' + end_ini)
        datasetpath = os.path.join(newdatapath, 'test_parameters.ini') 
    else: 
        datasetpath = existingdata 
    return datasetpath  
                

def check_data(datastore, **kwargs):             
    '''Checks if CLASS spectrum exists with specified parameters.'''
    datastore = correct_path(datastore)
    check = False
    preexistingdata = None
    for datapath in next(os.walk(datastore))[1]:
        testfile = os.path.join(datastore, datapath, 'test_parameters.ini') 
        if is_data(testfile, **kwargs): 
            print("Dataset already exists at: " + os.path.join(datastore, datapath))
            check = True 
            preexistingdata = testfile 
    return (check, preexistingdata)         
         
def is_data(path, **kwargs): 
    check = 0
    try: 
        correct_path(path)
    except: 
        return False
    else: 
        text = open(correct_path(path)).read()
        for key, val in kwargs.items(): 
            #print(key, str(val)) 
            test = key + ' = ' + str(val)
            if test not in text: 
                check += 1
        if check > 0:
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
