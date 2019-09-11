import numpy as np

def dPs_array(low, high, step):                                                 
    dPs = np.array([(high[zval].ps_table - low[zval].ps_table)/(2.*step)                 
           for zval in range(len(high))])                                        
    dlogPs = np.array([(high[zval].log_ps_table - low[zval].log_ps_table)/(2.*step)      
              for zval in range(len(high))])                                     
    return dPs, dlogPs    

def dPs(fid_ps_table, var_ps_table, step, centered=False):                      
    if centered==False:                                                         
        dps_table = (var_ps_table - fid_ps_table) / step                        
    elif centered==True:                                                        
        var_high = var_ps_table                                                 
        var_low = fid_ps_table                                                  
        dps_table = (var_high - var_low)/(2 * step)                             
    return dps_table                                                            
                                                                                
def dlogPs(fid_ps_table, var_ps_table, step, centered=False):                   
    if centered==False:                                                         
        dlogps_table = (np.log(var_ps_table) - np.log(fid_ps_table)) / step     
    elif centered==True:                                                        
        var_high = np.log(var_ps_table)                                         
        var_low = np.log(fid_ps_table)                                          
        dlogps_table = (var_high - var_low)/(2 * step)                          
    return dlogps_table 

def derivative(function, dif_variable, relstep, **kwargs): 
    var_high = (1. + relstep) * kwargs[dif_variable]
    var_low = (1. - relstep)  * kwargs[dif_variable]

    args_high = dict(kwargs) 
    args_low = dict(kwargs)  

    args_high[dif_variable] = var_high
    args_low[dif_variable] = var_low

    func_high = function(**args_high)
    func_low = function(**args_low)

    deriv = (func_high - func_low) / (2. * relstep * kwargs[dif_variable])  
    return(deriv)  

def log_interp(x_eval, x_data, y_data):
    logx_eval = np.log10(x_eval)
    logx_data = np.log10(x_data)
    logy_data = np.log10(y_data)
    return np.power(10.0, np.interp(logx_eval, logx_data, logy_data))
