import os
import shutil

lss_survey = 'DESI' # 'BOSS' or 'DESI' or 'EUCLID' 
plot_type = 'Fine' # 'Fine' or 'Grid' 
results_dir_name = 'FINAL_RESULTS_18SEP2020'

###################################
##  USER ONLY MODIFY ABOVE HERE  ##
###################################

prepath_in= ('/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/'
    + results_dir_name
    + '/'
    + lss_survey
    + '/'
    + lss_survey
    + '_'
    + plot_type
    + 'Plot_') 

prepath_out = ('/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/'
    + results_dir_name 
    + '/Parsed/'
    + lss_survey
    + '/'
    + lss_survey
    + '_'
    + plot_type
    + 'Plot_')

if plot_type == 'Grid': 
    max_idx = 120
    mass_idx = 10
    prefix = 'gp'
elif plot_type == 'Fine': 
    max_idx = 84
    mass_idx = 21
    prefix = 'fp'  

for idx in range(max_idx): 
    tidx=idx//mass_idx
    midx=idx%mass_idx
    path=(prepath_out + str(idx) + '/')
    os.makedirs(path, exist_ok=True)
    filename= (prefix + '_' + str(tidx) + '_' + str(midx)+'.db')
    shutil.copyfile(prepath_in+str(idx)+'/'+filename, prepath_out+str(idx)+'/'+filename)
    print(path) 
