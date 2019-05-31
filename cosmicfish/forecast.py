import numpy as np

import cosmicfish as cf

#class lightrelicanalysis: 
#  
#    def __init__(self, 
#                 name=None, 
#                 fid=None,
#                 nonfid=None, 
#                 z_table=None, 
#                 classdir=None, 
#                 datastore=None, 
#                 testconvergence=None, 
#                 paramvary=None): 
#        self.name=name
#        self.fid=fid
#        self.z_table=z_table
#        self.classdir=classdir
#        self.datastore=datastore
#        self.testconvergence=testconvergence
#        self.paramvary=paramvary
#        self.spectra=[]
#
#    def generate(self): 
#        if testconvergence==True: 
#            print('Convergence package in development...') 
#        else:   
            
            
                 

class relic_convergence_analysis: 

    def __init__(self, fid, param, varytype, varyvals, z_table, m_ncdm,
                 classdir, datastore): 
        self.name = param + "convergence analysis for light relic" 
        self.z_table = z_table
        self.m_ncdm = m_ncdm
        self.classdir = classdir
        self.datastore = datastore
        self.fid = fid

        if varytype=="abs":     
            self.variants = varyvals
        if varytype=="rel": 
            self.variants = self.fid[param] * varyvals

        #First index is redshift, second index is variation 
        if param=='T_ncdm':
            self.spectra = [[cf.spectrum(cf.generate_data(dict(self.fid,
                                                               **{param : i, 'N_ncdm' : 1, 'm_ncdm' : self.m_ncdm, 'z_pk' : j}),
                                                          self.classdir,
                                                          self.datastore).replace('/test_parameters.ini',''),
                                         self.z_table) for i in self.variants] for j in self.z_table] 

        else: 
            self.spectra = [[cf.spectrum(cf.generate_data(dict(self.fid, 
                                                               **{param : i, 'z_pk' : j}), 
                                                          self.classdir, 
                                                          self.datastore).replace('/test_parameters.ini',''), 
                                         self.z_table) for i in self.variants] for j in self.z_table]

def dPs(self, fid_ps_table, step, theta):
    table = (self.ps_table - fid_ps_table) / step
    self.dps_table[theta] = table
    self.log_dps_table[theta] = np.log(table)
