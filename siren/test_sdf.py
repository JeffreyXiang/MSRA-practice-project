import os
import sys
import json
import torch
import scipy.io
import numpy as np
from tqdm import tqdm, trange
from module import sdf_model
from utils_sdf import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

"""=============== GLOBAL ARGUMENTS ==============="""
config_filepath = sys.argv[1]
with open(config_filepath, 'r') as config_file:
    config = json.load(config_file)

output_path = config['output_path']
experiment_name = config['experiment_name']

iterations = config['iterations'] if 'iterations' in config else 10000
batch_size = config['batch_size'] if 'batch_size' in config else 65536
learning_rate = config['learning_rate'] if 'learning_rate' in config else 1e-4
model_type = config['model_type'] if 'model_type' in config else 'siren'

i_print = config['i_print'] if 'i_print' in config else 100
i_save = config['i_save'] if 'i_save' in config else 10000


"""=============== START ==============="""
# Model
model = sdf_model(model_type)
for name, param in model.named_parameters():
    print(name)

# Load log directory
log_path = os.path.join(output_path, experiment_name)
os.makedirs(log_path, exist_ok=True)
check_points = [os.path.join(log_path, f) for f in sorted(os.listdir(log_path)) if 'tar' in f]
print('Found check_points', check_points)
if len(check_points) > 0:
    check_point_path = check_points[-1]
    print('Reloading from', check_point_path)
    check_point = torch.load(check_point_path)
    model.load_state_dict(check_point['model'])
else:
    global_step = 0

create_mesh(model, os.path.join(log_path, 'test'), N=1024, max_batch=65536)