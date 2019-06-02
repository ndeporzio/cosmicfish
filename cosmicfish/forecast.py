import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.param = param

        #Calculate parameter variations
        if varytype=="abs":     
            self.variants = varyvals
        if varytype=="rel": 
            self.variants = self.fid[param] * varyvals

        #Calculate fiducial spectra
        self.fid_spectra = [cf.spectrum(cf.generate_data(dict(self.fid, **{'z_pk' : j}),
                                                          self.classdir, 
                                                          self.datastore).replace('/test_parameters.ini',''), 
                                        self.z_table) for j in self.z_table]

        #Calculate variation spectra 
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


        #Calculate derivatives 
        if param=='T_ncdm': 
            self.dps = [[dPs(self.fid_spectra[j].ps_table, 
                             self.spectra[j][i].ps_table, 
                             self.variants[i]) for i in range(len(self.variants))] for j in range(len(self.z_table))]
            self.dlogps = [[dlogPs(self.fid_spectra[j].ps_table,
                                   self.spectra[j][i].ps_table,
                                   self.variants[i]) for i in range(len(self.variants))] for j in range(len(self.z_table))]
        else: 
            self.dps = [[dPs(self.fid_spectra[j].ps_table, 
                             self.spectra[j][i].ps_table, 
                             self.variants[i]-self.fid[param]) for i in range(len(self.variants))] for j in range(len(self.z_table))]
            self.dlogps = [[dlogPs(self.fid_spectra[j].ps_table,
                                   self.spectra[j][i].ps_table,
                                   self.variants[i]-self.fid[param]) for i in range(len(self.variants))] for j in range(len(self.z_table))]


    def plot_ps(self, z_index=0):
        sns.set() 
        sns.set_palette("Blues_d", n_colors=len(self.variants))
        plt.figure(figsize=(15, 7.5))

        ax1 = plt.subplot(1, 2, 1)
        for idx, ps in enumerate(self.spectra[z_index]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm = {0:.2f}[K]'.format(self.variants[idx])
            else:
                plotlabel = r'$\delta$' + self.param + ' = {0:.2f}%'.format((self.variants[idx]/self.fid[self.param]-1))
            ax1.plot(ps.k_table, ps.ps_table, label=plotlabel)
        ax1.set_title(r'$P_g$ for $z={0:.2f}$'.format(self.z_table[z_index]))
        ax1.set_xlabel(r'k [Mpc$^{-1}$]')
        ax1.set_ylabel(r'[Mpc$^3$]')
        ax1.legend()

        ax2= plt.subplot(1, 2, 2)
        for idx, ps in enumerate(self.spectra[z_index]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm = {0:.2f}[K]'.format(self.variants[idx])
            else:
                plotlabel = r'$\delta$' + self.param + ' = {0:.2f}%'.format((self.variants[idx]/self.fid[self.param]-1))
            ax2.plot(ps.k_table, ps.log_ps_table, label=plotlabel)
        ax2.set_title(r'$log(P_g)$ for $z={0:.2f}$'.format(self.z_table[z_index]))
        ax2.set_xlabel(r'k [Mpc$^{-1}$]')
        ax2.set_ylabel(r'[Mpc$^3$]')
        ax2.legend()
        
        plt.show()

#        if self.param=='T_ncdm': 
#            plt.figure(figsize=(15, 7.5))
#
#            ax1 = plt.subplot(1, 2, 1)
#            for idx, ps in enumerate(self.spectra[z_index]):
#                if self.param=='T_ncdm':
#                    plotlabel = r'T_ncdm = {0:.2f}[K]'.format(self.variants[idx])
#                else:
#                    plotlabel = r'$\delta$' + self.param + ' = {0:.2f}%'.format((self.variants[idx]/self.fid[self.param]-1))
#                ax1.plot(ps.k_table, ps.ps_table, label=plotlabel)
#            ax1.set_title(r'$P_g$ for $z={0:.2f}$'.format(self.z_table[z_index]))
#            ax1.set_xlabel(r'k [Mpc$^{-1}$]')
#            ax1.set_ylabel(r'[Mpc$^3$]')
#            ax1.legend()
#
#            ax2= plt.subplot(1, 2, 2)
#            for idx, ps in enumerate(self.spectra[z_index]):
#                if self.param=='T_ncdm':
#                    plotlabel = r'T_ncdm = {0:.2f}[K]'.format(self.variants[idx])
#                else:
#                    plotlabel = r'$\delta$' + self.param + ' = {0:.2f}%'.format((self.variants[idx]/self.fid[self.param]-1))
#                ax2.plot(ps.k_table, ps.log_ps_table, label=plotlabel)
#            ax2.set_title(r'$log(P_g)$ for $z={0:.2f}$'.format(self.z_table[z_index]))
#            ax2.set_xlabel(r'k [Mpc$^{-1}$]')
#            ax2.set_ylabel(r'[Mpc$^3$]')
#            ax2.legend()        
#        
#            plt.show() 
                  
    def plot_dps(self, z_index=0):
        sns.set()
        sns.set_palette("Blues_d", n_colors=len(self.variants))
        plt.figure(figsize=(15, 7.5))

        ax1 = plt.subplot(1, 2, 1)
        for idx, dps in enumerate(self.dps[z_index]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm = {0:.2f}[K]'.format(self.variants[idx])
            else:
                plotlabel = r'$\delta$' + self.param + ' = {0:.2f}%'.format((self.variants[idx]/self.fid[self.param]-1))
            ax1.plot(self.spectra[z_index][idx].k_table, dps, label=plotlabel)
        ax1.set_title(r'$\partial P_g / \partial$' + self.param + ' for $z={0:.2f}$'.format(self.z_table[z_index]))
        ax1.set_xlabel(r'k [Mpc$^{-1}$]')
        ax1.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
        ax1.legend()

        ax2= plt.subplot(1, 2, 2)
        for idx, dlogps in enumerate(self.dlogps[z_index]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm = {0:.2f}[K]'.format(self.variants[idx])
            else:
                plotlabel = r'$\delta$' + self.param + ' = {0:.2f}%'.format((self.variants[idx]/self.fid[self.param]-1))
            ax2.plot(self.spectra[z_index][idx].k_table, dlogps, label=plotlabel)
        ax2.set_title(r'$\partial log(P_g) / \partial$'+self.param+' for $z={0:.2f}$'.format(self.z_table[z_index]))
        ax2.set_xlabel(r'k [Mpc$^{-1}$]')
        ax2.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
        ax2.legend()

        plt.show()

        if self.m_ncdm is not 0: 
            plt.figure(figsize=(15, 7.5))

            ax1 = plt.subplot(1, 2, 1)
            for idx, dps in enumerate(self.dps[z_index]):
                
                domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm, 2.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                plotlabel = r'!!omega_ncdm = {0:.2f}[K]'.format(self.variants[idx])
                ax1.plot(self.spectra[z_index][idx].k_table, dps/domega_chi_dT_chi, label=plotlabel)
            ax1.set_title(r'$\partial P_g / \partial$ omega_ncdm' + ' for $z={0:.2f}$'.format(self.z_table[z_index]))
            ax1.set_xlabel(r'k [Mpc$^{-1}$]')
            ax1.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
            ax1.legend()

            ax2= plt.subplot(1, 2, 2)
            for idx, dlogps in enumerate(self.dlogps[z_index]):
                plotlabel = r'!!omega_ncdm = {0:.2f}[K]'.format(self.variants[idx])
                ax2.plot(self.spectra[z_index][idx].k_table, dlogps/domega_chi_dT_chi, label=plotlabel)
            ax2.set_title(r'$\partial log(P_g) / \partial$ omega_ncdm'+' for $z={0:.2f}$'.format(self.z_table[z_index]))
            ax2.set_xlabel(r'k [Mpc$^{-1}$]')
            ax2.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
            ax2.legend()

            plt.show()



    def plot_delta_dps(self, z_index=0): 
        sns.set()
        sns.set_palette("Blues_d", n_colors=len(self.variants))
        plt.figure(figsize=(15, 7.5))

        ax1 = plt.subplot(1, 2, 1)
        dps_comp = list(self.dps[z_index][:-1])
        for idx, dps in enumerate(dps_comp):
            if self.param=='T_ncdm':
                plotlabel = r'\Delta for T_ncdm @ {0:.2f}[K]'.format((self.variants[idx+1]+self.variants[idx])/2)
            else:
                plotlabel = (r'$\Delta$ for ' 
                             + self.param 
                             + ' @ ' + r'$\delta$' 
                             + self.param
                             + ' = {0:.2f}%'.format(((self.variants[idx+1]/self.fid[self.param]-1)
                                                     + (self.variants[idx+1]/self.fid[self.param]-1)) / 2.))
            ax1.plot(self.spectra[z_index][idx].k_table, self.dps[z_index][idx+1]-self.dps[z_index][idx], label=plotlabel)
        ax1.set_title(r'$\Delta (\partial P_g / \partial$' + self.param + ') for $z={0:.2f}$'.format(self.z_table[z_index]))
        ax1.set_xlabel(r'k [Mpc$^{-1}$]')
        ax1.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
        ax1.legend()

        ax2= plt.subplot(1, 2, 2)
        dlogps_comp = list(self.dlogps[z_index][:-1])
        for idx, dlogps in enumerate(dlogps_comp):
            if self.param=='T_ncdm':
                plotlabel = r'\Delta for  T_ncdm @ {0:.2f}[K]'.format((self.variants[idx+1]+self.variants[idx])/2)
            else:
                plotlabel = (r'$\Delta$ for ' 
                             + self.param 
                             + ' @ ' + r'$\delta$'  
                             + self.param
                             + ' = {0:.2f}%'.format(((self.variants[idx+1]/self.fid[self.param]-1)
                                                     + (self.variants[idx+1]/self.fid[self.param]-1)) / 2.))
            ax2.plot(self.spectra[z_index][idx].k_table, self.dlogps[z_index][idx+1]-self.dlogps[z_index][idx], label=plotlabel)
        ax2.set_title(r'$\Delta (\partial log(P_g) / \partial$'+self.param+') for $z={0:.2f}$'.format(self.z_table[z_index]))
        ax2.set_xlabel(r'k [Mpc$^{-1}$]')
        ax2.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
        ax2.legend()

        plt.show()

        if self.m_ncdm is not 0:
            plt.figure(figsize=(15, 7.5))

            ax1 = plt.subplot(1, 2, 1)
            dps_comp = list(self.dps[z_index][:-1])
            for idx, dps in enumerate(dps_comp):
                domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm, 2.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                plotlabel = r'!!omega_ncdm = {0:.2f}[K]'.format(self.variants[idx])

                ax1.plot(self.spectra[z_index][idx].k_table, dps/domega_chi_dT_chi, label=plotlabel)
            ax1.set_title(r'$\partial P_g / \partial$ omega_ncdm' + ' for $z={0:.2f}$'.format(self.z_table[z_index]))
            ax1.set_xlabel(r'k [Mpc$^{-1}$]')
            ax1.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
            ax1.legend()

            ax2= plt.subplot(1, 2, 2)
            dlogps_comp = list(self.dlogps[z_index][:-1])
            for idx, dlogps in enumerate(self.dlogps[z_index]):
                plotlabel = r'!!omega_ncdm = {0:.2f}[K]'.format(self.variants[idx])
                ax2.plot(self.spectra[z_index][idx].k_table, dlogps/domega_chi_dT_chi, label=plotlabel)
            ax2.set_title(r'$\partial log(P_g) / \partial$ omega_ncdm'+' for $z={0:.2f}$'.format(self.z_table[z_index]))
            ax2.set_xlabel(r'k [Mpc$^{-1}$]')
            ax2.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
            ax2.legend()

            plt.show()

def dPs(fid_ps_table, var_ps_table, step):
    table = (var_ps_table - fid_ps_table) / step
    dps_table = table
    return dps_table

def dlogPs(fid_ps_table, var_ps_table, step):
    table = (np.log(var_ps_table) - np.log(fid_ps_table)) / step
    dlogps_table = table
    return dlogps_table
