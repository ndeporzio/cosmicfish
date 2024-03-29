import numpy as np
import scipy 
from scipy.integrate import quad
import cosmicfish as cf

def rsd(omega_b, omega_cdm, omega_ncdm, h, z, mu, k, b0, D, alphak2, 
    relic, T_ncdm, lss_survey_name, M_chi, m_nu, step=True, 
    delta_L=cf.RSD_DELTA_L_NUMERATOR_FACTOR, beta0=1.7, beta1=1.0):

    #if relic==False:             
    #    m_ncdm = cf.m_ncdm(omega_ncdm/3., cf.RELIC_TEMP_SCALE)     
    #    k_fs = cf.kfs(m_ncdm, T_ncdm, h, z)
    #elif relic==True: 
    #    m_ncdm = cf.m_ncdm(omega_ncdm, T_ncdm)
    #    k_fs = cf.kfs(m_ncdm, T_ncdm, h, z) 

    if step==True:                                                              
        delta_L_numerator = delta_L                                             
        rlcdm_unnorm = cf.rlambdacdm(h, k)                                        
    else:                                                                       
        delta_L_numerator = 0.                                                  
        rlcdm_unnorm = 1. 

    rlcdm_norm = cf.rlambdacdm(h, cf.BIAS_NORMALIZATION_SCALE*h)
    rlcdm = rlcdm_unnorm/rlcdm_norm  

    m_ncdm = float(relic)
    f = cf.fgrowth(omega_b, omega_cdm, h, z) 

    k_fs_1 = cf.kfs(M_chi, T_ncdm, h, z) 
    g1_unnorm = cf.ggrowth(k, k_fs_1, h, omega_b, omega_cdm, omega_ncdm, 
        step, delta_L)  
    g1_norm = cf.ggrowth(cf.BIAS_NORMALIZATION_SCALE*h, k_fs_1, h, 
        omega_b, omega_cdm, omega_ncdm, step, delta_L)                
    g1 = g1_unnorm / g1_norm

    k_fs_2 = cf.kfs(m_nu, T_ncdm, h, z)                                        
    g2_unnorm = cf.ggrowth(k, k_fs_2, h, omega_b, omega_cdm, omega_ncdm,        
        step, delta_L)                                                          
    g2_norm = cf.ggrowth(cf.BIAS_NORMALIZATION_SCALE*h, k_fs_2, h,             
        omega_b, omega_cdm, omega_ncdm, step, delta_L)                          
    g2 = g2_unnorm / g2_norm

    bl = cf.bL(b0, D) 
                                                                              
    if lss_survey_name=='DESI':                                                 
        b1tilde =  (1. + (bl * rlcdm * g1 * g2) + alphak2 * np.power(k, 2.))                    
        #b1tilde =  (1. + (b0*np.power((1.+z), 0.5*1.0)-1.)*(rlcdm * g1 * g2) + alphak2 * np.power(k, 2.))
    elif lss_survey_name=='EUCLID':                                             
        b1tilde = (1. + (beta0*np.power((1.+z), 0.5*beta1)-1.)*(rlcdm * g1 * g2) + alphak2*np.power(k, 2.))                                      
    else:                                                                       
        print("ERROR: bias function not defined for given LSS survey.") 
#    print(b1tilde) 
    R = np.power((b1tilde + np.power(mu, 2.) * f), 2.)                          
    return R                                                                    
                                                                                
def log_rsd(omega_b, omega_cdm, omega_ncdm, h, z, mu, k, b0, D, alphak2,            
    relic, T_ncdm, lss_survey_name, M_chi, m_nu, step=True,                     
    delta_L=cf.RSD_DELTA_L_NUMERATOR_FACTOR, beta0=1.7, beta1=1.0
    ):                 
    return np.log(cf.rsd(omega_b, omega_cdm, omega_ncdm, h, z, mu, k, b0, D, alphak2,            
    relic, T_ncdm, lss_survey_name, M_chi, m_nu, step,                     
    delta_L, beta0, beta1
        ))

def fog(omega_b, omega_cdm, omega_ncdm, h, z, k, mu, sigma_fog_0):              
    sigma_z = cf.SIGMA_Z                                                        
    sigma_fog = cf.sigma_fog(sigma_fog_0,  z)                                      
    sigma_v = cf.sigma_v(sigma_fog, sigma_z, z)                                                    
    F = np.exp(-1.                                                              
               * np.power((k*mu*sigma_v)                                        
               / cf.H(omega_b, omega_cdm, omega_ncdm, h, z), 2.))                  
    return F                                                                    

def rlambdacdm(h, k): 
    val = 1. + cf.RSD_DELTA_LAMBDACDM * np.tanh(
        (cf.RSD_ALPHA * k) / (cf.RSD_KEQ_PREFACTOR * h)) 
    return val

def sigma_fog(sigma_fog_0, z): 
    val = sigma_fog_0 *  np.sqrt(1. + z)
    return val

def sigma_v(sigma_fog, sigma_z, z): 
    val = ((1. + z) 
        * np.sqrt((np.power(sigma_fog, 2.) / 2.)                         
        + np.power(cf.C * sigma_z, 2.))) 
    return  val
                                                                                
def log_fog(omega_b, omega_cdm, omega_ncdm, h, z, k, mu, sigma_fog_0):          
    return np.log(cf.fog(omega_b, omega_cdm, omega_ncdm, h, z, k, mu,              
        sigma_fog_0))

def ap(omega_b, omega_cdm, omega_ncdm, h, z,
    omega_b_fid, omega_cdm_fid, omega_ncdm_fid, h_fid, z_fid): 

    H_fid = cf.H(omega_b_fid, omega_cdm_fid, omega_ncdm_fid, h_fid, z_fid)
    Da_fid = cf.Da(omega_b_fid,  omega_cdm_fid, omega_ncdm_fid, h_fid, z_fid)

    H_nonfid = cf.H(omega_b, omega_cdm, omega_ncdm, h, z)
    Da_nonfid = cf.Da(omega_b, omega_cdm, omega_ncdm, h, z) 

    ap_value = ((H_nonfid * np.power(Da_fid, 2.))
        / (H_fid * np.power(Da_nonfid, 2.))) 

    return ap_value

def log_ap(omega_b, omega_cdm, omega_ncdm, h, z, 
    omega_b_fid, omega_cdm_fid, omega_ncdm_fid, h_fid, z_fid): 
    return np.log(cf.ap(omega_b, omega_cdm, omega_ncdm, h, z,                                    
        omega_b_fid, omega_cdm_fid, omega_ncdm_fid, h_fid, z_fid))

def cov(): 
    return 1.  

def log_cov():
    return np.log(cf.cov())

def cov_dkdH(omega_b, omega_cdm, omega_ncdm, h, z, mu, k):
    return  (k * np.power(mu, 2.) 
        / cf.H(omega_b, omega_cdm, omega_ncdm, h, z)) 

def cov_dkdDa(omega_b, omega_cdm, omega_ncdm, h, z, mu, k):
    return (-1. * k * (1. - np.power(mu, 2.)) 
        / cf.Da(omega_b, omega_cdm, omega_ncdm, h, z)) 

def H(omega_b, omega_cdm, omega_ncdm, h, z):                                    
    # Returns H in units of m/s/Mpc                                             
    omega_m = omega_b + omega_cdm + omega_ncdm                                  
    omega_lambda = np.power(h, 2.) - omega_m                                    
    Hval = 1000.* 100. * np.sqrt(omega_m * np.power(1. + z, 3.) + omega_lambda)  
    return Hval 

def Da(omega_b, omega_cdm, omega_ncdm, h, z):                                
    prefactor = cf.C / (1. + z)                                                    
    def integrand(zval):                                                        
        return 1. / H(omega_b, omega_cdm, omega_ncdm, h, zval)                  
    integral, error = quad(integrand, 0., z)                                    
    return prefactor * integral   

def neff(ndens, Pg):                                                            
    # ndens at specific z, Pg at specific k and z                               
    n = np.power((ndens * Pg) / (ndens*Pg + 1.), 2.)                            
    return n 

def kfs(m_ncdm, T_ncdm, h, z):                                                      
    k_fs = ((
        (cf.KFS_NUMERATOR_FACTOR * h * m_ncdm) 
        / (cf.KFS_DENOMINATOR_FACTOR * np.sqrt(1. + z))) 
        * np.power(T_ncdm/cf.RELIC_TEMP_SCALE, -1.))           
    return k_fs 

def omega_ncdm(T_ncdm, m_ncdm, N_ncdm, forecast_type):                                  
    """Returns omega_ncdm as a function of T_ncdm, m_ncdm.                      
                                                                                
    T_ncdm : relic temperature in units [K]                                     
    m_ncdm : relic mass in units [eV]                                           
                                                                                
    omega_ncdm : relative relic abundance. Unitless.                            
    """                                                                         
    if forecast_type=="neutrino":                                               
        omega_ncdm = 3. * (m_ncdm / cf.NEUTRINO_SCALE_FACTOR)                                      
    if forecast_type=="relic":                                                  
        omega_ncdm = np.power(T_ncdm / cf.RELIC_TEMP_SCALE, 3.) * (m_ncdm 
            / cf.NEUTRINO_SCALE_FACTOR) * N_ncdm                
    return omega_ncdm

def T_ncdm(omega_ncdm, m_ncdm):                                                 
    # RELICS ONLY?                                                              
    """Returns T_ncdm as a function of omega_ncdm, m_ncdm.                      
                                                                                
    omega_ncdm : relative relic abundance. Unitless.                            
    m_ncdm : relic mass in units [eV].                                          
                                                                                
    T_ncdm : relic temperature in units [K]                                     
    """                                                                         
                                                                                
    T_ncdm = (np.power( cf.NEUTRINO_SCALE_FACTOR * omega_ncdm / m_ncdm, 1./3.) 
              * cf.RELIC_TEMP_SCALE)                  
    return T_ncdm

def m_ncdm(omega_ncdm, T_ncdm): 
    val = (
        cf.NEUTRINO_SCALE_FACTOR 
        * omega_ncdm 
        * np.power(cf.RELIC_TEMP_SCALE / T_ncdm, 3.))
    return val 

def domega_ncdm_dM_ncdm(T_ncdm): 
    deriv = (np.power(T_ncdm / cf.RELIC_TEMP_SCALE, 3.) 
        / cf.NEUTRINO_SCALE_FACTOR)
    return deriv 

def dM_ncdm_domega_ncdm(T_ncdm):
    deriv = cf.NEUTRINO_SCALE_FACTOR * np.power(
        cf.RELIC_TEMP_SCALE / T_ncdm, 3.) 
    return deriv 

def domega_ncdm_dT_ncdm(T_ncdm, M_ncdm): 
    deriv = (3 * np.power(T_ncdm, 2.) * np.power(cf.RELIC_TEMP_SCALE, -3.) 
        * M_ncdm / cf.NEUTRINO_SCALE_FACTOR)
    return deriv

def dT_ncdm_domega_ncdm(T_ncdm, M_ncdm):  
    deriv = (np.power(cf.RELIC_TEMP_SCALE, 3.) * cf.NEUTRINO_SCALE_FACTOR /
        (3. * np.power(T_ncdm, 2.) * M_ncdm))
    return deriv

def dT_ncdm_dN_ncdm(m_ncdm, omega_ncdm, N_ncdm): 
    deriv = (
        np.power(cf.NEUTRINO_SCALE_FACTOR / m_ncdm, 1./3.)
        * cf.RELIC_TEMP_SCALE
        * np.power(3./2., 1./3.)
        * (-1./3.) 
        * np.power(N_ncdm, -2./3.)
        * np.power(omega_ncdm, 1./3.) 
        ) 
    return deriv 

def dT_dTcmb(T_cmb):
    return T_cmb

def sigmafog(z, sigma_fog_0):                                                                
    """Returns sigma_fog as function of redshift.                               
                                                                                
    Value of sigma_fog^(0) hardcoded here.                                      
                                                                                
    Args:                                                                       
        z : redshift to evaluate at                                             
                                                                                
    Returns:                                                                    
        Value of sigma_fog                                                      
    """                                                                         
    #sigfog0 = 250000. # Units [m s^-1]                                          
    sigfog = sigma_fog_0 * np.sqrt(1. + z)                                          
    return sigfog 

def sigmav(z, sigma_fog_0):                                                                  
    """Returns sigma_v as function of redshift.                                 
                                                                                
    Value of sigma_z hardcoded here.                                            
                                                                                
    Args:                                                                       
        z : redshift to evaluate at                                             
                                                                                
    Returns:                                                                    
        Value of sigma_v                                                        
    """                                                                         
    sigmav = ((1. + z) 
              * np.sqrt(0.5 * np.power(sigmafog(z, sigma_fog_0), 2.)                   
                        + np.power(cf.C * cf.SIGMA_Z, 2.)))                       
    # careful units!!!                                                          
    return sigmav          

def fgrowth(omega_b, omega_cdm, h, z):                                          
    """Returns f growth factor.                                                 
                                                                                
    Args:                                                                       
        omega_b : little omega baryon abundance                                 
                                                                                
        omega_cdm: little omega cdm abundance                                   
                                                                                
        h : little h Hubble constant                                            
                                                                                
        z : redshift                                                            
                                                                                
    Returns:                                                                    
        Growth factor f(z)                                                      
    """                                                                         
                                                                                
    gamma = cf.RSD_GAMMA                                                                
    epsilon = (omega_b + omega_cdm) / np.power(h, 2.)                           
    inner = ((epsilon * np.power(1. + z, 3.))                                   
             / (epsilon * np.power(1. + z, 3.) - epsilon + 1.))                 
    f = np.power(inner, gamma)                                                  
    return f              

def ggrowth(k, k_fs, h, omega_b, omega_cdm, omega_ncdm, step=True, 
    delta_L=cf.RSD_DELTA_L_NUMERATOR_FACTOR):                           
    """Returns g growth factor(?).                                              
                                                                                
    Args:                                                                       
        k : wave number in Mpc^-1                                               
                                                                                
        k_fs : free streaming scale                                             
                                                                                
        omega_b : little omega baryon abundance                                 
                                                                                
        omega_cdm : little omega cdm abunance                                   
                                                                                
        omega_ncdm : little omega ncdm abundance                                
                                                                                
    Returns:                                                                    
        Growth factor (?) g                                                     
    """                                                                         
    
    if step==True:  
        delta_L_numerator = delta_L 
    else: 
        delta_L_numerator = 0. 
                                                                                
    Delta_q = cf.RSD_DELTA_Q                                                               
    q = cf.RSD_Q_NUMERATOR_FACTOR * k / k_fs                                          
    Delta_L =  (delta_L_numerator * omega_ncdm 
                / (omega_b + omega_cdm))                         
    g = ((1. + (Delta_L / 2.) * np.tanh(1. + (np.log(q) / Delta_q))))               
    return g


def bL(b0, D): 
    val = (b0 / D) - 1.
    return val 
    

def gen_V(h, omega_b, omega_cdm, z, N_ncdm, T_ncdm=None, m_ncdm=0,        
          c=cf.C, fsky=None, z_spacing=cf.DEFAULT_Z_BIN_SPACING):                                               
    # T_ncdm in units of [K]                                                    
    # m_ncdm in units of [eV]                                                   
    # c in units of [m*s^-1]                                                    
    # returns V in units [Mpc^3]                                                
                                                                                
    if fsky==None:                                                              
        fsky = 1.                                                               
    H = 1000. * 100. * h # H has units of [m*s^-1*Mpc^-1]                       
    if m_ncdm is not None:                                                      
        if N_ncdm==3.: # Degenerate neutrinos CAUTION                                  
            omega_chi = cf.omega_ncdm(T_ncdm, m_ncdm, N_ncdm, "neutrino")                                   
        else: # Light relic CAUTION                                         
            omega_chi = cf.omega_ncdm(T_ncdm, m_ncdm, N_ncdm, "relic")               
       # else:                                                                   
       #     print("N_ncdm must be 1 (relics) or 3 (degenerate neutrinos).")
    else:                                                                       
        omega_chi = 0                                                           
    omega_m = omega_b + omega_cdm + omega_chi # Unitless                        
    omega_lambda = np.power(h, 2.) - omega_m # Unitless                         
                                                                                
    zmax = z + (0.5 * z_spacing)                                                          
    zmin = z - (0.5 * z_spacing)                                                            
    zsteps = 10000.                                                               
    dz = ((zmin+zmax)/2.) / zsteps                                                          
    z_table_max = np.arange(0., zmax, dz)                                       
    z_table_min = np.arange(0., zmin, dz)                                       
    z_integrand_max = (h * dz) /  np.sqrt(omega_m                               
                                          * np.power(1. + z_table_max, 3.)      
                                          + omega_lambda)                       
    z_integrand_min = (h * dz) /  np.sqrt(omega_m                               
                                          * np.power(1. + z_table_min, 3.)      
                                          + omega_lambda)                       
    z_integral_max = np.sum(z_integrand_max)                                    
    z_integral_min = np.sum(z_integrand_min)                                    
    v_max = ((4. * np.pi / 3.)                                                  
             * np.power(c / H, 3.)                                 
             * np.power(z_integral_max, 3.))                                    
    v_min = ((4. * np.pi / 3.)                                                  
             * np.power(c / H, 3.)                                 
             * np.power(z_integral_min, 3.))                                    
    v = (v_max - v_min) * fsky                     
    # Incorporate fsky into volume calculation.                                 
    return v # Units [Mpc^3]                                                    

                                                                                
def gen_k_table(volume, z, h, n_s, k_steps, scaling='log'):                                     
    # volume in units of [Mpc^3]                                                
    # returns k_table in units [Mpc^-1]                                         

    #print('k_min = ', (np.pi / h) * np.power(volume, -1./3.))
    #k_table = np.linspace((np.pi / h) * np.power(volume, -1./3.),               
    #                      k_max,                                                
    #                      k_steps)       

    kmin = np.pi * np.power(volume, -1./3.) 
    #kmin = np.power(10., -4) * h
    #kmin = 1. / (160. * np.power(h, -1.)) #SDSS analysis 
    #kmax = 1. / (20. * np.power(h, -1.)) #SDSS analysis
    #kmax = cf.K_MAX_PREFACTOR * np.power(1.+z, 2./(2.+n_s)) * h                                   
    kmax = cf.K_MAX_PREFACTOR * h 

    if scaling=='linear': 
        k_table = np.linspace(kmin, kmax, k_steps) 
    elif scaling=='log':  
        k_table = np.geomspace(kmin, kmax, k_steps)
    return k_table #Units [Mpc^-1]


def set_sky_cover(fsky=None, fcoverage_deg=None): 
    if (fsky is None) and (fcoverage_deg is not None):                      
        fdeg = fcoverage_deg                                  
        ffrac = fcoverage_deg / cf.FULL_SKY_DEGREES                     
        # ^^^ http://www.badastronomy.com/bitesize/bigsky.html              
    elif (fsky is not None) and (fcoverage_deg is None):                    
        fdeg =  cf.FULL_SKY_DEGREES * fsky                    
        ffrac = fsky                                                    
    elif (fsky is not None) and (fcoverage_deg is not None):                
        print("Both f_sky and sky coverage specified,"                      
            + " using value for f_sky.")                                  
        fdeg = cf.FULL_SKY_DEGREES * fsky                     
        ffrac = fsky                                                    
    else:                                                                   
        print("Assuming full sky survey.")                                  
        ffrac = 1.                                                      
        fdeg = cf.FULL_SKY_DEGREES
    return ffrac, fdeg


def k_pivot(): 
    kp = cf.KP_PREFACTOR 
    return kp  


def dlogPdAs(As_fid): 
    val = 1. / As_fid
    return val 


def dlogPdns(k, kp): 
    val = np.log(k / kp) 
    return val 



















