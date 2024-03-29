import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cosmicfish as cf

class convergence: 

    def __init__(
        self, 
        classdir, 
        datastore, 
        forecast_type,
        fiducialcosmology,
        z_steps,
        lss_survey_name, 
        dNdz,
        fisher_order,
        fsky=None,
        fcoverage_deg=None,
        RSD=True,
        FOG=True,
        AP=True,
        COV=True,
        mu_step=0.05,
        varyfactors=[],
        showplots=True,
        saveplots=False,
        savepath=None,
        plotparams=None):  
                

        self.classdir = classdir
        self.datastore = datastore
        self.type = forecast_type
        self.fid = fiducialcosmology
        self.z_steps = z_steps
        self.lss_survey_name = lss_survey_name
        self.dNdz = dNdz
        self.fsky, self.fcoverage_deg = cf.set_sky_cover(fsky,  fcoverage_deg)    
        self.use_rsd = RSD
        self.use_fog = FOG
        self.use_ap = AP
        self.use_cov = COV
        self.varyfactors = varyfactors
        self.showplots = showplots
        self.saveplots = saveplots
        self.savepath = savepath

        self.mu_step = mu_step
        self.plotparams = plotparams

        if self.plotparams==None: 
            self.plotparams = [
                'A_s',
                'n_s',
                'omega_b',
                'omega_cdm',
                'h',
                'tau_reio',
                'omega_ncdm',
                'sigmafog',
                'b0',
                'alphak2'] 
            

        # Calculate fiducial spectra
        self.forecasts = [cf.forecast(
                            classdir=self.classdir, 
                            datastore=self.datastore,
                            forecast_type=self.type,
                            fiducialcosmo=self.fid,
                            z_steps=self.z_steps,
                            lss_survey_name=self.lss_survey_name,
                            dNdz=self.dNdz,
                            fsky=self.fsky,
                            dstep=vfactor,  
                            RSD=self.use_rsd,
                            FOG=self.use_fog,
                            AP=self.use_ap,
                            COV=self.use_cov)
                            for vidx, vfactor in enumerate(self.varyfactors)]

        for fcst in self.forecasts: 
            fcst.gen_fisher(
                fisher_order,
                mu_step=self.mu_step,
                skipgen=False) 
                            

    def plot_ps(self, z_index=0, mu_index=0, xscale='linear', plotdata=False):
        sns.set() 
        sns.set_palette("Blues_d", n_colors=len(self.varyfactors)+1)
        plt.figure(figsize=(15, 7.5))

        ax1 = plt.subplot(1, 2, 1)
        for idx, fcst in enumerate(self.forecasts):
            data = np.array(fcst.Pg)
            plotlabel = (r'$\Delta$' + str(100.*self.varyfactors[idx]) + r'%') 
            ax1.plot(fcst.k_table[z_index], data[z_index, :, mu_index], 
                label=plotlabel)
        ax1.set_title(r'$P_g$ for z={:.3f}, $\mu$={:0.2f}'.format(
            self.z_steps[z_index], np.arange(-1, 1, self.mu_step)[mu_index]))
        ax1.set_xlabel(r'k [Mpc$^{-1}$]')
        ax1.set_ylabel(r'$P_g$ [Mpc$^3$]')
        ax1.legend()
        ax1.set_xscale(xscale) 

        ax2 = plt.subplot(1, 2, 2)                                              
        for idx, fcst in enumerate(self.forecasts):                             
            data =  np.log(fcst.Pg)                                            
            plotlabel = (r'$\Delta$' + str(100.*self.varyfactors[idx]) + r'\%') 
            ax2.plot(fcst.k_table[z_index], data[z_index, :, mu_index], 
                label=plotlabel) 
        ax2.set_title(r'$log(P_g)$ for z={:.3f}, $\mu$={:0.2f}'.format(               
            self.z_steps[z_index], np.arange(-1, 1, self.mu_step)[mu_index]))    
        ax2.set_xlabel(r'k [Mpc$^{-1}$]')                                       
        ax2.set_ylabel(r'log($P_g$) [log(Mpc$^3$)]')                                      
        ax2.legend()                                                            
        ax2.set_xscale(xscale) 

        if self.saveplots==True:
            plt.savefig(self.savepath + "/Convergence_Pg.png")
        if self.showplots==True:
            plt.show()

    def plot_dps(self, paramname, z_index=0, mu_index=0, xscale='linear', 
        plotdata=False):  

        sns.set()                                                               
        sns.set_palette("Blues_d", n_colors=len(self.varyfactors)+1)               
        plt.figure(figsize=(15, 7.5))                                           
                                                                                
        for idx, fcst in enumerate(self.forecasts):                             
            data = np.array(getattr(fcst, 'dlogPgd'+paramname))                                            
            plotlabel = (r'$\Delta $' + str(100.*self.varyfactors[idx]) + r'%') 
            plt.plot(fcst.k_table[z_index], data[z_index, :, mu_index], 
                label=plotlabel) 
        plt.title((r'$dlogP_g/d$'
            +paramname+r' for z={:.3f}, $\mu$={:0.2f}').format(               
                self.z_steps[z_index], 
                np.arange(-1, 1, self.mu_step)[mu_index]))    
        plt.xlabel(r'k [Mpc$^{-1}$]')                                       
        plt.ylabel(r'$dlogP_g/d$'+paramname+r' [Mpc$^3$]')                                      
        plt.legend()                                                            
        plt.xscale(xscale)                                                  
                                                                                
        if self.saveplots==True:                                                
            plt.savefig(self.savepath + "/Convergence_dlogPgd" + paramname 
                + ".png")                  
        if self.showplots==True:                                                
            plt.show()

    def plot_delta_dps(self, paramname, z_index=0, mu_index=0, xscale='linear',       
        plotdata=False):                                                        
                                                                                
        sns.set()                                                               
        sns.set_palette("Blues_d", n_colors=len(self.varyfactors)+1)               
        plt.figure(figsize=(15, 7.5))                                           
                                                                                
        for idx, fcst in enumerate(self.forecasts[0:-1]):
            datahigh = np.array(
                getattr(self.forecasts[idx+1], 'dlogPgd'+paramname))                             
            datalow = np.array(
                getattr(self.forecasts[idx], 'dlogPgd'+paramname)) 
            data = datahigh - datalow                               
            plotlabel = (r'$\Delta$' + str(100.*self.varyfactors[idx]) + r'%') 
            plt.plot(fcst.k_table[z_index], data[z_index, :, mu_index], 
                label=plotlabel) 
        plt.title((r'$\Delta$ $dlogP_g/d$'                                               
            +paramname+r' for z={:.3f}, $\mu$={:0.2f}').format(                  
                self.z_steps[z_index],                                          
                np.arange(-1, 1, self.mu_step)[mu_index]))                      
        plt.xlabel(r'k [Mpc$^{-1}$]')                                           
        plt.ylabel(r'\Delta $dlogP_g/d$'+paramname+r' [Mpc$^3$]')                                          
        plt.legend()                                                            
        plt.xscale(xscale)                                                      
                                                                                
        if self.saveplots==True:                                                
            plt.savefig(self.savepath + "/Convergence_Delta_dlogPgd" 
                + paramname + ".png")                  
        if self.showplots==True:                                                
            plt.show()

    def gen_all_plots(self, z_index=0, mu_index=0, xscale='linear', 
        plotdata=False):
 
        self.plot_ps()
    
        for parameter in self.plotparams: 
            self.plot_dps(parameter)
            self.plot_delta_dps(parameter)



 
