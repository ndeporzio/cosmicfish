import numpy as np
import cosmicfish as cf
from unittest import TestCase

class TestEquations(TestCase): 

    def test_kfs(self):  
        omega_ncdm = 0.06 / 93.14 
        h = 0.70148
        z = 1.0

        TEST = 0.23809020713603482
        EVAL = cf.kfs(omega_ncdm, h, z) 
        print("Test output: ", EVAL)
        print("Benchmark value: ", TEST)
 
        self.assertTrue(np.isclose(TEST, EVAL))

    def test_fgrowth(self):  
        omega_b = 0.02226
        omega_cdm = 0.11271
        h =  0.70148
        z = 1.0

        TEST = 0.8545772603876                            
        EVAL = cf.fgrowth(omega_b,  omega_cdm, h, z)                                         
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL))  

    def test_ggrowth(self):
        z = 1.0
        k = 0.1
        h =  0.70148                                                      
        omega_b = 0.02226                                                       
        omega_cdm = 0.11271    
        omega_ncdm = 0.06 / 93.14                                                  

        TEST = 1.001286340493369                                              
        EVAL = cf.ggrowth(z,  k,  h,  omega_b, omega_cdm, omega_ncdm)                            
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL))

    def test_rsd_1(self): 
        omega_b = 0.02226
        omega_cdm = 0.11271
        omega_ncdm = 0.06 / 93.14
        h = 0.70148
        z = 1.0
        mu = 0.5
        k = 0.1
        b1L = 0.7
        alphak2 = 1.0

        TEST = 6.933862940488465
        EVAL = cf.rsd(omega_b, omega_cdm, omega_ncdm, h, z,  mu, k, b1L, 
            alphak2) 
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)

        self.assertTrue(np.isclose(TEST, EVAL))         
    
    def test_rsd_2(self):                                                       
        omega_b = 0.02226                                                       
        omega_cdm = 0.11271                                                     
        omega_ncdm = 0.06 / 93.14                                               
        h = 0.70148                                                             
        z = 1.0                                                                 
        mu = 0.0                                                              
        k = 0.1                                                                 
        b1L = 0.7                                                               
        alphak2 = 1.0                                                           
                                                                                
        TEST = 5.854360619860679                                          
        EVAL = cf.rsd(omega_b, omega_cdm, omega_ncdm, h, z,  mu, k, b1L,        
            alphak2)                                                            
        print("Test output: ", EVAL)                                            
        print("Benchmark value: ", TEST)                                        
                                                                                
        self.assertTrue(np.isclose(TEST, EVAL))     


