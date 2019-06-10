import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cosmicfish as cf

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
                                                               **{'T_ncdm' : i,
                                                                  'N_ncdm' : 1,
                                                                  'm_ncdm' : self.m_ncdm, 
                                                                  'z_pk' : j}),
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


    def plot_ps(self, z_index=0, xscale='linear', plotdata=False):
        sns.set() 
        sns.set_palette("Blues_d", n_colors=len(self.variants)+1)
        plt.figure(figsize=(15, 7.5))

        ax1 = plt.subplot(1, 2, 1)
        for idx, ps in enumerate(self.spectra[z_index]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm = {:.3f}[K]'.format(self.variants[idx]*ps.T_cmb)
            else:
                plotlabel = (r'$\delta$' \
                             + self.param \
                             + ' = {:.3f}%'.format(100*(self.variants[idx]/self.fid[self.param]-1)))
            ax1.plot(ps.k_table, ps.ps_table, label=plotlabel)
            if plotdata==True and idx==0: #idx==0 is just to plot a single dataset 
                ax1.plot(ps.class_pk['k (h/Mpc)']*ps.h, 
                         ps.class_pk['P (Mpc/h)^3']*np.power(ps.h, -3), 
                         label='CLASS P(k) Data',
                         marker='x', 
                         linestyle=':')
        ax1.set_title(r'$P_g$ for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
        ax1.set_xlabel(r'k [Mpc$^{-1}$]')
        ax1.set_ylabel(r'$P_g$ [Mpc$^3$]')
        ax1.set_xlim(0, 1.1 * np.max(self.spectra[z_index][0].k_table))
        ax1.legend()
        ax1.set_xscale(xscale) 

        ax2= plt.subplot(1, 2, 2)
        for idx, ps in enumerate(self.spectra[z_index]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm = {:.3f}[K]'.format(self.variants[idx]*ps.T_cmb)
            else:
                plotlabel = (r'$\delta$' \
                             + self.param \
                             + ' = {:.3f}%'.format(100*(self.variants[idx]/self.fid[self.param]-1)))
            ax2.plot(ps.k_table, ps.log_ps_table, label=plotlabel)
            if plotdata==True and idx==0: #idx==0 is just to plot a single dataset 
                ax2.plot(ps.class_pk['k (h/Mpc)']*ps.h, 
                         np.log(ps.class_pk['P (Mpc/h)^3']*np.power(ps.h, -3)), 
                         label='CLASS Log(P(k)) Data', 
                         marker='x', 
                         linestyle=':')
        ax2.set_title(r'$log(P_g)$ for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
        ax2.set_xlabel(r'k [Mpc$^{-1}$]')
        ax2.set_ylabel(r'log($P_g$) [Mpc$^3$]')
        ax2.set_xlim(0, 1.1 * np.max(self.spectra[z_index][0].k_table))
        ax2.legend()
        ax2.set_xscale(xscale) 
        
        plt.show()

    def plot_dps(self, z_index=0, xscale='linear'):
        sns.set()
        sns.set_palette("Blues_d", n_colors=len(self.variants))
        plt.figure(figsize=(15, 7.5))

        ax1 = plt.subplot(1, 2, 1)
        for idx, dps in enumerate(self.dps[z_index]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm = {:.3f}[K]'.format(self.variants[idx]*self.spectra[z_index][idx].T_cmb)
            else:
                plotlabel = r'$\delta$' + self.param + ' = {:.3f}%'.format(100*(self.variants[idx]/self.fid[self.param]-1))
            ax1.plot(self.spectra[z_index][idx].k_table, dps, label=plotlabel)
        ax1.set_title(r'$\partial P_g / \partial$' + self.param + ' for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
        ax1.set_xlabel(r'k [Mpc$^{-1}$]')
        ax1.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
        ax1.legend()
        ax1.set_xscale(xscale)

        ax2= plt.subplot(1, 2, 2)
        for idx, dlogps in enumerate(self.dlogps[z_index]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm = {:.3f}[K]'.format(self.variants[idx]*self.spectra[z_index][idx].T_cmb)
            else:
                plotlabel = r'$\delta$' + self.param + ' = {:.3f}%'.format(100*(self.variants[idx]/self.fid[self.param]-1))
            ax2.plot(self.spectra[z_index][idx].k_table, dlogps, label=plotlabel)
        ax2.set_title(r'$\partial log(P_g) / \partial$'+self.param+' for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
        ax2.set_xlabel(r'k [Mpc$^{-1}$]')
        ax2.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
        ax2.legend()
        ax2.set_xscale(xscale) 

        plt.show()

        if self.m_ncdm is not None: 
            plt.figure(figsize=(15, 7.5))

            ax1 = plt.subplot(1, 2, 1)
            for idx, dps in enumerate(self.dps[z_index]):
                
                domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm*self.spectra[z_index][idx].T_cmb, 2.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                #domega_chi_dT_chi = np.power(self.spectra[z_index][idx].T_ncdm/1.95, 3.) * (self.m_ncdm/94.) / self.spectra[z_index][idx].T_ncdm #simplified approximation domega/dT =  omega/T
                #domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm, 1.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                #domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm*self.spectra[z_index][idx].T_cmb, 0.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                plotlabel = r'T_ncdm = {:.3f}[K]'.format(self.variants[idx]*self.spectra[z_index][idx].T_cmb)
                plotdata = np.array(dps) * (1/domega_chi_dT_chi)
                ax1.plot(self.spectra[z_index][idx].k_table, plotdata, label=plotlabel)
                #print('dT/domega @ T_ncdm = {:.3f}[K] is: '.format(self.variants[idx]*self.spectra[z_index][idx].T_cmb) + str(1/domega_chi_dT_chi))
            ax1.set_title(r'$\partial P_g / \partial$ omega_ncdm' + ' for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
            ax1.set_xlabel(r'k [Mpc$^{-1}$]')
            ax1.set_ylabel(r'[Mpc$^3$ / (units of omega_ncdm)]')
            ax1.legend()
            if xscale=='log': 
                ax1.set_xscale('log') 

            ax2= plt.subplot(1, 2, 2)
            for idx, dlogps in enumerate(self.dlogps[z_index]):
                domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm*self.spectra[z_index][idx].T_cmb, 2.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                plotlabel = r'T_ncdm = {:.3f}[K]'.format(self.variants[idx]*self.spectra[z_index][idx].T_cmb)
                plotdata = np.array(dlogps) * (1/domega_chi_dT_chi)
                ax2.plot(self.spectra[z_index][idx].k_table, plotdata, label=plotlabel)
            ax2.set_title(r'$\partial log(P_g) / \partial$ omega_ncdm'+' for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
            ax2.set_xlabel(r'k [Mpc$^{-1}$]')
            ax2.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
            ax2.legend()
            if xscale=='log': 
                ax2.set_xscale('log') 

            plt.show()



    def plot_delta_dps(self, z_index=0, xscale='linear'): 
        sns.set()
        sns.set_palette("Blues_d", n_colors=len(self.variants))
        plt.figure(figsize=(15, 7.5))

        ax1 = plt.subplot(1, 2, 1)
        for idx, dps in enumerate(self.dps[z_index][:-1]):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm @ {:.3f}[K]'.format(self.spectra[z_index][idx].T_cmb*(self.variants[idx+1]+self.variants[idx])/2)
            else:
                plotlabel = (r'$\delta$' 
                             + self.param
                             + ' @ {:.3f}%'.format((100*(self.variants[idx+1]/self.fid[self.param]-1)
                                                     + 100*(self.variants[idx]/self.fid[self.param]-1)) / 2.))
            ax1.plot(self.spectra[z_index][idx].k_table, self.dps[z_index][idx+1]-self.dps[z_index][idx], label=plotlabel)
        ax1.set_title(r'$\Delta (\partial P_g / \partial$' + self.param + ') for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
        ax1.set_xlabel(r'k [Mpc$^{-1}$]')
        ax1.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
        ax1.legend()
        if xscale=='log': 
            ax1.set_xscale('log') 

        ax2= plt.subplot(1, 2, 2)
        dlogps_comp = list(self.dlogps[z_index][:-1])
        for idx, dlogps in enumerate(dlogps_comp):
            if self.param=='T_ncdm':
                plotlabel = r'T_ncdm @ {:.3f}[K]'.format(self.spectra[z_index][idx].T_cmb*(self.variants[idx+1]+self.variants[idx])/2)
            else:
                plotlabel = (r'$\delta$'  
                             + self.param
                             + ' @ {:.3f}%'.format((100*(self.variants[idx+1]/self.fid[self.param]-1)
                                                     + 100*(self.variants[idx]/self.fid[self.param]-1)) / 2.))
            ax2.plot(self.spectra[z_index][idx].k_table, self.dlogps[z_index][idx+1]-self.dlogps[z_index][idx], label=plotlabel)
        ax2.set_title(r'$\Delta (\partial log(P_g) / \partial$'+self.param+') for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
        ax2.set_xlabel(r'k [Mpc$^{-1}$]')
        ax2.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
        ax2.legend()
        if xscale=='log': 
            ax2.set_xscale('log') 

        plt.show()

        if self.m_ncdm is not None:
            plt.figure(figsize=(15, 7.5))

            ax1 = plt.subplot(1, 2, 1)
            for idx, dps in enumerate(self.dps[z_index][:-1]):
                domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm*self.spectra[z_index][idx].T_cmb, 2.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                #domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm, 2.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                plotlabel = r'T_ncdm @ {:.3f}[K]'.format(self.spectra[z_index][idx].T_cmb*(self.variants[idx+1]+self.variants[idx])/2)

                ax1.plot(self.spectra[z_index][idx].k_table, (self.dps[z_index][idx+1]-self.dps[z_index][idx])/domega_chi_dT_chi, label=plotlabel)
            ax1.set_title(r'$\Delta(\partial P_g / \partial$ omega_ncdm)' + ' for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
            ax1.set_xlabel(r'k [Mpc$^{-1}$]')
            ax1.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
            ax1.legend()
            if xscale=='log': 
                ax1.set_xscale('log') 

            ax2= plt.subplot(1, 2, 2)
            dlogps_comp = list(self.dlogps[z_index][:-1])
            for idx, dlogps in enumerate(dlogps_comp):
                domega_chi_dT_chi = (3. * np.power(self.spectra[z_index][idx].T_ncdm*self.spectra[z_index][idx].T_cmb, 2.) * self.m_ncdm) / (np.power(1.95, 3.) * 94.)
                plotlabel = r'T_ncdm @ {:.3f}[K]'.format(self.spectra[z_index][idx].T_cmb*(self.variants[idx+1]+self.variants[idx])/2)
                ax2.plot(self.spectra[z_index][idx].k_table, (self.dlogps[z_index][idx+1]-self.dlogps[z_index][idx])/domega_chi_dT_chi, label=plotlabel)
            ax2.set_title(r'$\Delta(\partial log(P_g) / \partial$ omega_ncdm)'+' for $z={:.3f}$, m_ncdm={:.3f} [eV]'.format(self.z_table[z_index], self.m_ncdm))
            ax2.set_xlabel(r'k [Mpc$^{-1}$]')
            ax2.set_ylabel(r'[Mpc$^3$ / (units of '+self.param+')]')
            ax2.legend()
            if xscale=='log': 
                ax2.set_xscale('log') 

            plt.show()

def dPs(fid_ps_table, var_ps_table, step):
    table = (var_ps_table - fid_ps_table) / step
    dps_table = table
    return dps_table

def dlogPs(fid_ps_table, var_ps_table, step):
    table = (np.log(var_ps_table) - np.log(fid_ps_table)) / step
    dlogps_table = table
    return dlogps_table
