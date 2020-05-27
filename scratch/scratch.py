import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import yass
from yass.visual.run import VisualizerOG
from yass import set_config
from yass import read_config
from yass.reader import READER
from yass.config import config

# Playing with Yass Output #
# ------------------------ #

# path to spike trains
p_st = '/home/jai/yass_analysis/tmp/nn_train/cluster/spike_train.npy'
p_st_post = '/home/jai/yass_analysis/tmp/nn_train/cluster_post_process/spike_train.npy'
# path to templates
p_templates = '/home/jai/yass_analysis/tmp/nn_train/cluster/templates.npy'
p_templates_post = '/home/jai/yass_analysis/tmp/nn_train/cluster_post_process/templates.npy'
# path to cluster results dir
p_cl_results = '/home/jai/yass_analysis/tmp/nn_train/cluster/cluster_result'
# path to ptp vals
ptp_vals = np.load('/home/jai/yass_analysis/tmp/nn_train/cluster/ptp_split/ptp.npy')

st = np.load(p_st_post)
templates = np.load(p_templates_post)
cl1_results = np.load(p_cl_results + '/cluster_result_1.npz')
cl1_ts = cl1_results['spiketime']
cl1_templates = cl1_results['templates']

# Playing with VisualizerOG #
# ------------------------- #

# Path to folder where all tmp data + venn data will be stored
save_dir = Path('/home/jai/yass_analysis/tmp/visualize/')
save_dir.mkdir(parents=True, exist_ok=True)
# Path to YASS config file
fname_config = Path('/home/jai/yass_analysis/config.yaml')
# Path to spike train from spike sorter 1 (blue)
fname_spiketrain1 = Path('/home/jai/yass_analysis/tmp/nn_train/cluster_post_process/spike_train.npy')
# Path to spike train from spike sorter 2 (red)
fname_spiketrain2 = Path('/home/jai/yass_analysis/tmp2/nn_train/cluster_post_process/spike_train.npy')
# Path to templates from spike sorter 1 (blue)
fname_templates1 = Path('/home/jai/yass_analysis/tmp/nn_train/cluster_post_process/templates.npy')
# Path to templates from spike sorter 2 (red)
fname_templates2 = Path('/home/jai/yass_analysis/tmp2/nn_train/cluster_post_process/templates.npy')
# Path to shifts if using YASS spiketrain
# fname_shifts1 = '/ssd/nishchal/neuropixel/tmp/final_deconv/deconv/shifts.npy'
# Path to scales if using YASS spiketrain
# fname_scales1 =  '/ssd/nishchal/neuropixel/tmp/final_deconv/deconv/scales.npy'
# Path to standardized pre processed recording
recording_path = Path('/home/jai/yass_analysis/tmp/preprocess/standardized.bin')
# Path to residuals
# residual_path = '/ssd/nishchal/neuropixel/tmp/final_deconv/residual/residual.bin'
# data type for the recording (Default coming out of YASS is float32)
recording_dtype = 'float32'
# chunks of the data (if you would like to visualize different venn plots for different 2 min chunks of the data)
chunks = 1

# make sure all paths exist
paths = [save_dir, fname_config, fname_spiketrain1, fname_spiketrain2, fname_templates1, fname_templates2,
         recording_path]
for f in paths:
    if not (f.exists()):
        raise FileNotFoundError(f)

# set `Config` and `READER` objects
CONFIG = config.Config.from_yaml(fname_config, save_dir)
reader = READER(recording_path, recording_dtype, CONFIG)
# reader_resid = READER(residual_path, recording_dtype, CONFIG)

# Create `VisualizerOG` Object
vog = VisualizerOG(fname_spiketrain1,
                  recording_path,
                  recording_dtype,
                  CONFIG,
                  save_dir,
                  fname_templates=fname_templates1,
                  # You can choose to send in pre computed file locations for each of these
                  # fname_residual=residual_path,      # When running it for YASS select the files created in the tmp folder
                  # fname_shifts=fname_shifts1,        # THe four arguments can be set to None if you'd like to calculate everything
                  # fname_scales=fname_scales1)        # from scratch.
                  )
