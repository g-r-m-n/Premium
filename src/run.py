# %% setup
# pip install numpy pandas doubleml datetime matplotlib xgboost


    
# load libraries
import numpy as np
import pandas as pd
from doubleml.datasets import make_iivm_data, make_pliv_CHS2015, make_plr_CCDDHNR2018, make_irm_data
import sys, os, json
from datetime import date

# set the path to repository:
root_dir = 'C:/DEV/'
#root_dir = '/home/studio-lab-user/'
#root_dir ='/mnt/batch/tasks/shared/LS_root/mounts/clusters/grmnzntt1/code/Users/grmnzntt/'

pth_to_src = root_dir+'AutoML/src/'
# data:
today = date.today().strftime('%Y%m%d')
# output folders:
path_to_data  = pth_to_src + 'data/'
output_folder = path_to_data+today+'/'
output_folder_plots  = output_folder+'plots/'
output_folder_tables = output_folder+'tables/'
# create output_folders if they do not exist:
os.makedirs(output_folder,exist_ok=True)
os.makedirs(output_folder_plots,exist_ok=True)
os.makedirs(output_folder_tables,exist_ok=True)
# load utility functions
sys.path.append(pth_to_src+'/utils/')
from utility import *
# reload functions from utility
from importlib import reload
reload(sys.modules['utility'])    

# set the run parameter configurations:  
config_params = set_configs() 
# config_params = set_configs('test_run_config.yaml')
#unpack the dictionnary to variables
locals().update(config_params)


np.random.seed(4444)



# %% run scenarios:


