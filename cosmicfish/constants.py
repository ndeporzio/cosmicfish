# All hardcoded constants should appear here. 

CLASSVARS = ['h', 'T_cmb', 'omega_b', 'N_eff', 'omega_cdm', 'N_ncdm',       
    'm_ncdm', 'T_ncdm', 'Omega_k', 'Omega_fld', 'YHe', 'recombination',     
    'reio_parametrization', 'tau_reio', 'reionization_exponent',            
    'reionization_width', 'helium_fullreio_redshift',                       
    'helium_fullreio_width', 'annihilation', 'decay', 'output',             
    'l_max_scalars', 'modes', 'lensing', 'ic', 'P_k_ini type', 'k_pivot',   
    'A_s', 'n_s', 'alpha_s', 'P_k_max_h/Mpc', 'z_pk', 'root',               
    'write background', 'write parameters', 'background_verbose',           
    'thermodynamics_verbose', 'perturbations_verbose', 'transfer_verbose',  
    'primordial_verbose', 'spectra_verbose',  'nonlinear_verbose',          
    'lensing_verbose', 'output_verbose', 'deg_ncdm']

C = 2.9979e8 # Speed of light in [m s^-1]
FULL_SKY_DEGREES = 41253. # Sky size in [deg^2]
NEUTRINO_SCALE_FACTOR = 93.14 # Neutrino mass/abundance rel. factor in [eV]
SIGMA_Z = 0.001 #Appears in FOG term. Units of [c]. 
KFS_NUMERATOR_FACTOR = 0.08
KFS_DENOMINATOR_FACTOR = 0.1
RSD_GAMMA = 0.55
RSD_DELTA_Q = 1.6
RSD_Q_NUMERATOR_FACTOR = 5.

RSD_DELTA_L_NUMERATOR_FACTOR = 0.6
#RSD_DELTA_L_NUMERATOR_FACTOR = 0.

RELIC_TEMP_SCALE = 1.95
KP_PREFACTOR = 0.05


DB_ELG = 0.84

K_MAX_PREFACTOR = 0.2

ANALYTIC_A_S = False
ANALYTIC_N_S = False

RSD_DELTA_LAMBDACDM = 4.8e-3                                                    
#RSD_DELTA_LAMBDACDM = 0.  

RSD_ALPHA = 4.                                                                  
RSD_KEQ_PREFACTOR = 0.015  
BIAS_NORMALIZATION_SCALE = 1.e-5 

DEFAULT_K_TABLE_STEPS = 100
DEFAULT_Z_BIN_SPACING = 0.1
