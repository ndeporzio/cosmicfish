import os
import shutil

prepath_in='/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS/EUCLID/EUCLID_GridPlot_'
prepath_out='/n/home02/ndeporzio/projects/cosmicfish/cfworkspace/results/FINAL_RESULTS/Parsed/EUCLID/EUCLID_GridPlot_'

for idx in range(120): 
    tidx=idx//10
    midx=idx%10
    path=prepath_out+str(idx)+'/'
    os.mkdir(path)
    filename= ('gp_'+str(tidx)+'_'+str(midx)+'.db')
    shutil.copyfile(prepath_in+str(idx)+'/'+filename, prepath_out+str(idx)+'/'+filename)
    print(path) 
#    os.rename(prepath_in+str(idx)+'.err', prepath_out+str(idx)+'/EUCLID_GridPlot_56234229_'+str(idx)+'.err')
