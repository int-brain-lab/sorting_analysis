import matplotlib
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib import gridspec

import numpy as np
import pandas as pd
import os
import shutil
import cv2
import glob2
import parmap

from matplotlib_venn import venn3, venn3_circles

from numba import jit

import tables
from scipy.io import loadmat
import scipy
import h5py
import hdf5storage
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.cm as cm
from scipy import exp
from scipy.optimize import curve_fit

from numba import jit
# functions

    
    
class WaveForms(object):

    def __init__(self, wave_forms, geometry=None):
        """Sets up and computes properties of wave forms.
        params:
        -------
        wave_forms: numpy.ndarray
            Shape of wave forms is (N, C, t). N is total number of wave forms
            C is number of channels and t is number of time points.
        geometry: numpy.ndarray
            Geometry of the probe that the wave forms belong to. Array has shape
            (N, 2) the coordinates of the probe.
        """
        self.wave_forms = wave_forms
        self.n_unit, self.n_channel, self.n_time = self.wave_forms.shape
        print ("self.n_unit, self.n_channel, self.n_time: ", 
                self.n_unit, self.n_channel, self.n_time)
        
        self.unit_overlap = None
        self.pdist = None
        self.geom = geometry
        self.main_chans = self.wave_forms.ptp(axis=2).argmax(axis=1)
        self.ptps = self.wave_forms.ptp(axis=2).max(axis=1)
        

    def generate_correlated_noise_add(self, time, n_filters=10, 
                                      min_snr=10., dtype=np.float32):
        
        """Creates correlated background noise for synthetic datasets.
        params:
        -------
        n_filters: int
            The number of filters to create background noise with.
        min_snr: float
            Minimum SNR of filter that would be convovled to create correlated noise.
        """
        background_noise = []
        allwf = self.wave_forms.reshape([-1, self.n_time])
        allwf = allwf[allwf.ptp(1) > 10.]
        allwf = allwf / allwf.std(axis=1)[:, None] / 2.

        #for it in tqdm(range(self.n_channel), "Generating correlated noise."):
        for it in range(self.n_channel):
            if it%50==0:
                print ("chan: ", it)
            # Make noise for each channel
            cor_noise = 0.
            wf_idx = np.random.choice(range(len(allwf)), n_filters, replace='False')
            for idx in wf_idx:
                noise = np.random.normal(0, 1, time)
                cor_noise += np.convolve(noise, allwf[idx][::-1], 'same')
            cor_noise += np.random.normal(0, 3, len(cor_noise))
            cor_noise = (cor_noise - cor_noise.mean()) / cor_noise.std()
            
            back_noise = cor_noise.astype(dtype)
            background_noise.append(back_noise)
   
        correlated_noise = np.array(background_noise).T

        return correlated_noise



def visualize_traces(data_synthetic, time1, time2):
    
    n_chans = data_synthetic.shape[1]
    
    for k in range(n_chans):
        plt.plot(data_synthetic[time1:time2,k]+k*20,c='black')
    plt.show()



