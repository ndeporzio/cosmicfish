import numpy as np
import cosmicfish as cf
from unittest import TestCase

ATOL = 0.000
RTOL = 0.0001

omega_b = 0.02226
omega_cdm = 0.11271
omega_ncdm = 0.06/93.14
tau_reio = 0.059888
h = 0.70148
sigma_fog_0 = 250000.
b1L = 0.7
alphak2 = 1.0
c = 2.9979e8 

z = 1.0
mu = 0.1
k = 0.2 

class TestEquations(TestCase): 

    def test_kfs(self):  

        TEST = 0.07936340237867828
        EVAL = cf.kfs(omega_ncdm, h, z) 

        print("Test output: ", EVAL)
        print("Benchmark value: ", TEST)
 
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))

    def test_fgrowth(self):  

        TEST = 0.8545772603876                            
        EVAL = cf.fgrowth(omega_b,  omega_cdm, h, z)                                         

        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))  

    def test_ggrowth(self):
        test_k = 0.1

        TEST = 1.0014156220263413                                          
        EVAL = cf.ggrowth(z, test_k,  h,  omega_b, omega_cdm, omega_ncdm)                            

        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))

    def test_rsd_1(self): 
        test_mu = 0.5
        test_k = 0.1

        TEST = 6.933862940488465
        EVAL = cf.rsd(omega_b, omega_cdm, omega_ncdm, h, z,  test_mu, test_k, 
            b1L, alphak2) 

        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)

        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))         
    
    def test_rsd_2(self):                                                       
        test_mu = 0.0                                                              
        test_k = 0.1                                                                 
                                                                                
        TEST = 5.854360619860679                                          
        EVAL = cf.rsd(omega_b, omega_cdm, omega_ncdm, h, z,  test_mu, test_k, 
            b1L, alphak2)                                                            

        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))     

    def test_fog_1(self):  

        TEST = 0.9832285224721687
        EVAL = cf.fog(omega_b, omega_cdm, omega_ncdm, h, z, k, mu, sigma_fog_0)

        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))

    def test_logfog(self):
                                                                                
        TEST = np.log(0.9832285224721687)                                
        EVAL = cf.log_fog(
            omega_b, omega_cdm, omega_ncdm, h, z, k, mu, sigma_fog_0)
                                                                                
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))

    def test_ap_1(self):

        TEST = 1.0                                       
        EVAL = cf.ap(omega_b, omega_cdm, omega_ncdm, h, z, 
            omega_b,  omega_cdm, omega_ncdm, h, z)           
                                                                                
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))  

    def  test_logap_1(self):  
        TEST = 0.                                                              
        EVAL = cf.log_ap(omega_b, omega_cdm, omega_ncdm, h, z,                      
            omega_b,  omega_cdm, omega_ncdm, h, z)                              
                                                                                
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))

    def test_ap_2(self): 

        omega_b_nf = 1.1 * omega_b
        omega_cdm_nf = 1.1 * omega_cdm
        omega_ncdm_nf = 1.1 * omega_ncdm
        h_nf = 1.1 * h
        z_nf = 1.1 * z

        TEST = 1.2622065637538376
        EVAL = cf.ap(omega_b_nf, omega_cdm_nf, omega_ncdm_nf, h_nf, z_nf, 
            omega_b, omega_cdm,  omega_ncdm, h, z)
         
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))

    def test_cov(self):
                                                                                
        TEST = 1.0                                                              
        EVAL = cf.cov()                      
                                                                                
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL)) 

    def test_log_cov(self):                                                         
                                                                                
        TEST = 0.                                                              
        EVAL = cf.log_cov()                                                         
                                                                                
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL)) 

    def test_H(self):

        TEST = 120057.21682503408
        EVAL = cf.H(omega_b, omega_cdm, omega_ncdm, h, z)
        
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))        

    def test_Da(self): 
    
        TEST = 1672.990829337001                                               
        EVAL = cf.Da(omega_b, omega_cdm, omega_ncdm, h, z)                       
                                                                                
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL)) 

    def test_neff(self):
        Pg = 1315.71
        nbar = 465448.76455

        TEST = 0.9999999967341373
        EVAL = cf.neff(nbar, Pg) 

        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL))  

    def  test_genV(self):
        
        TEST = 1.2478295308979862E10                                  
        EVAL = cf.gen_V(h, omega_b, omega_cdm, 1.05, 3., None, 0.02, c, 0.34, 0.1)

        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)

        self.assertTrue(np.isclose(TEST, EVAL, RTOL, ATOL)) 







