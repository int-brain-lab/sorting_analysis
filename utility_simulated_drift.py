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
def make_shifted_templates(ptp_fitted_centred, 
                       temps,
                       shift, resolution,xx,
                          color):
    
    ''' Rolled template in channel values (float point)
    
        resolution: upsample rate of the gaussian fit
        
    '''
    resolution = 20
    y_scale = 1.0

    roll_val = int((shift)*resolution)
    template_shifted = np.roll(ptp_fitted_centred, roll_val, axis=0)

    xx_resample = np.arange(0,resolution*temps.shape[1], resolution)
    scales = template_shifted[xx_resample]

    
    return scales
    

def make_default_template(temps, fitted, xx, 
                          resolution, chan_ids,
                          ptp_distribution):
    
    ''' 
    '''
    
    x = np.arange(temps.shape[1])
    fit_max = np.max(fitted)
    fit_argmax = np.argmax(fitted)

    # find ratios of units
    roll_val = int((xx[fit_argmax]%1)*resolution)
    fitted_roll = np.roll(fitted, roll_val, axis=0)
    
    # compute ratios for starting template
    xx_resample = np.arange(0,resolution*temps.shape[1],resolution)
   
    scales = fitted_roll[xx_resample]
    
    return (fitted_roll, scales)


def draw_template(ax, geom, neighbour_chans, idx, 
                  temps, temp_single_col, temp_scaling,col_um,
                  color, y_scale):
              
    ax.plot((geom[neighbour_chans[idx],0][:,None]+ \
              np.arange(temps.shape[1])/5.).transpose() + col_um,
              temp_single_col*temp_scaling*y_scale+geom[neighbour_chans[idx],1], 
              c=color)
   

# fit guassian to distriubtion
def gaus(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))


# get drifting templates
def make_drift_template(geom,
                       temps,
                       unit,
                       shifts,
                       radius,
                       plotting):
    
    ''' Make the drift interpolation
    '''
    resolution = 20
    y_scale = 1.0

    # find max chan of unit
    max_chan = temps[unit].ptp(0).argmax(0)
    #print ("Unit: ", unit, " , ptp: ", ptps[unit])

    # compute distsances of all electrodes to max channel
    dists=[]
    for k in range(geom.shape[0]):
        dists.append(np.linalg.norm(geom[max_chan]-geom[k]))
    dists=np.array(dists)

    neighbour_chans = np.where(dists<radius)[0]

    if plotting:
        fig=plt.figure()
        ax=plt.subplot(111)
    
    # shift each vertical column individually
    templates_out = []
    final_channels = []
    final_templates =[]

    # 
    vertical_cols_x_val = np.unique(geom[neighbour_chans][:,0])
    for vertical_col in vertical_cols_x_val:

        idx = np.where(geom[neighbour_chans,0]==vertical_col)
        #print ("local vertical geom: ", geom[neighbour_chans[idx]][:,1])
        y_vals = geom[neighbour_chans[idx]]

        ptp_local = temps[unit].ptp(0)
        #print (ptp_local[neighbour_chans[idx]])

        x = y_vals[:,1]
        data = ptp_local[neighbour_chans[idx]]

        chan_ids = np.arange(data.shape[0])
        chan_ids = np.asarray(chan_ids)
        ptp_distribution = np.asarray(data)

        n = len(ptp_distribution)  ## <---
        mean = sum(ptp_distribution*chan_ids)/n
        sigma = np.lib.scimath.sqrt(sum(ptp_distribution*(chan_ids-mean)**2)/n)

        # ***********************************
        # draw the original template
        max_chan = temps[unit].ptp(0).argmax(0)
        # select vertical 
        idx_vertical_col = np.where(geom[neighbour_chans,0]==geom[max_chan,0])
        final_channels.append(neighbour_chans[idx])

        temp_single_col = temps[unit][:,neighbour_chans[idx]]
        ptps_single_col = temp_single_col.ptp(0)
        #print ("ptps_single_col :", ptps_single_col)

        # find / interpolate channel centered template
        try:
            popt,pcov = curve_fit(gaus,chan_ids,ptp_distribution,
                             maxfev=1000)#,p0=[0.18,mean,sigma])  ## <--- leave out the first estimation of the parameters
        except:
            print ("Gaussian couldn't be fit: unit ", unit)
            
            final_templates = None
            final_channels = None
            return (final_templates, final_channels)
            
        xx = np.linspace( 0, ptp_distribution.shape[0], 
                         ptp_distribution.shape[0]*resolution )  ## <--- calculate against a continuous variable
        fitted= gaus(xx,*popt)

        (ptp_fitted_centred, ptp_default_template) = make_default_template(
                              temps[unit][:,neighbour_chans[idx]], fitted, xx, 
                              resolution, chan_ids, ptp_distribution)

        #shifts = np.array([-1])
        cmap = cm.get_cmap('viridis',shifts.shape[0])

        for ctr, shift in enumerate(shifts):
            #shift = 0.5
            temp_scaling = make_shifted_templates(ptp_fitted_centred, 
                                   temps[unit][:,neighbour_chans[idx]],
                                   shift, resolution, xx, cmap(ctr))

            # initialize the shifted template
            temp_temp = np.zeros(temp_single_col.shape)

            # shift the templates if drift is > 1 chan
            # note this should have zero filling for templates that roll too far
            col_shift = int(shift)
            temp_single_col_local = np.roll(temp_single_col,col_shift,axis=1)

            # get reisudal shift <1.0 value
            shift_local = (abs(shift)-int(abs(shift)))*np.sign(shift)
            if shift_local<0:
                temp_single_col_local = np.roll(temp_single_col,-1,axis=1)

            # blend templates from 2nd to 2nd last
            for k in range(1,temp_temp.shape[1]-1, 1):
                # blend template
                if shift_local>=0:
                    if shift_local>0.5:
                        temp_local = (temp_single_col_local[:,k-1]*(1-shift_local)+\
                                      temp_single_col_local[:,k]*(shift_local))
                    else:
                        temp_local = (temp_single_col_local[:,k-1]*(shift_local)+\
                                     temp_single_col_local[:,k]*(1-shift_local))
                else:
                    shift_local= -shift_local
                    if shift_local>0.5:
                        temp_local = (temp_single_col_local[:,k-1]*(1-shift_local)+\
                                      temp_single_col_local[:,k]*(shift_local))
                    else:
                        temp_local = (temp_single_col_local[:,k-1]*(shift_local)+\
                                      temp_single_col_local[:,k]*(1-shift_local))

                temp_local = temp_local/temp_local.ptp(0)
                #temp_local = temp_local #*temp_scaling[k] #/ptps_single_col[k]
                temp_temp[:,k] = temp_local

            #temp_scaling = np.roll(temp_scaling,col_shift,axis=0)
            if plotting:
                draw_template(ax, geom, neighbour_chans, idx, 
                              temps, temp_temp, temp_scaling,  vertical_col, cmap(ctr),
                              y_scale)

            #print (temp_temp.shape, temp_scaling.shape)
            templates_out.append(temp_temp*temp_scaling)
        
        # plot original template
        if plotting:
            temp_scaling=1.0
            ax.plot((geom[neighbour_chans[idx],0][:,None]+ \
              np.arange(temps.shape[1])/5.).transpose() + vertical_col,
              temp_single_col*y_scale+geom[neighbour_chans[idx],1], 
              c='red')
    
    
    #for k in range(0,len(templates_out)//shifts.shape[0]):
    for k in range(0,shifts.shape[0],1):
        temp2 = np.hstack((templates_out[k],
                           templates_out[k+shifts.shape[0]],
                           templates_out[k+shifts.shape[0]*2],
                           templates_out[k+shifts.shape[0]*3]))

        final_templates.append(temp2)
    
    final_templates = np.array(final_templates)
    final_channels = np.hstack(final_channels)
    
    return (final_templates, final_channels)

def shift_templates_all(units, geom, temps_local, shifts, 
                        radius, plotting):

    for ctr, unit in enumerate(units):
        (template_shifted, channels) = make_drift_template(
                           geom,
                           temps_local,
                           unit,
                           shifts,
                           radius,
                           plotting)

        if template_shifted is not None:
            # replace each template with update
            temps_local[unit,:,channels] = template_shifted.squeeze().T

    return temps_local
    
    
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

# function
def generate_poisson_uniform_firingrate2(rec_len_sec, sample_rate):
   
    # 1ms wide bins used for now;
    bin_width = 0.001 # ms precise bins;
    
    # sample uniformly to get firing rate between 1-10Hz
    f_rate = np.random.rand()*9+1
    
    # set threshold for spiking for a poisson process
    poisson_thresh = f_rate*bin_width

    # generate random spike probabilities;
    # make sure last 10 milliseconds doesn't have any spikes
    spikes = np.random.rand(int(rec_len_sec/bin_width)-10)

    # times in milliseconds
    idx=np.where(spikes<=poisson_thresh)[0]

    # find inter-time interval and obtain scaling;
    # spikes > 100ms apart are not scaled
    diffs = idx[1:]-idx[:-1]
    idx2 = np.where(diffs>=100)[0]
    diffs[idx2]=100
    
    # set scale
    scale = np.exp((diffs-100)/200)
    scale = np.hstack((1.0,scale))
      
    # convert from milliseconds to sample_rate sample-time + add random shift so 
    # not all spike trains land exactly on sample 0; 
    # Cat: TODO add individual spike-time shifts, and avoid refractory violations;
    times = idx * sample_rate//1000 +np.random.randint(sample_rate//1000,size=1)
        
    return (times, scale)



def select_ground_truth_units(root_dir,fname_templates,n_templates,
                             geom):
    try: 
        os.mkdir(root_dir+'ground_truth')
    except:
        print ("ground truth directory already made")

    # load geometry and templates
    temps = np.load(fname_templates, allow_pickle=True)
    print ("Total templates avialable; ", temps.shape)

    # get ptps and order by height
    ptps = temps.ptp(1).max(1)
    max_chans = temps.ptp(1).argmax(1)

    y = np.histogram(max_chans,bins=np.arange(385))
    plt.plot(y[1][1:], y[0])
    plt.title("Number of neurons per channel (ordered by depth)")
    plt.xlabel("Channel ID")
    plt.show()

    idx = np.argsort(ptps)[::-1]

    # Select unit
    units=idx[:n_templates]
    print ("Total units selected: ", n_templates, ', ids: ', units[:10],"...")
    # select templates to be injected based on PTP
    templates_ground_truth = temps[units]

    np.save(root_dir+'/ground_truth/templates_ground_truth.npy', templates_ground_truth)

    units_original_order = units.copy()
    
    units_reordered = np.arange(n_templates)
    
    return (units_reordered, templates_ground_truth)


def visualize_drift(shifts, geom, temps, unit,
                    radius):
    
    resolution = 20
    y_scale = 1.0
    #geom = np.loadtxt('/media/cat/1TB/data/synthetic/p1_g0_t0.imec0.ap_geom.txt')

    #for ctr, unit in enumerate(units):
    (template_shifted, channels) = make_drift_template(
                           geom,
                           temps,
                           unit,
                           shifts,
                           radius,
                           plotting=False)
   
    fig=plt.figure()
    cmap = cm.get_cmap('viridis',shifts.shape[0])
    for k in range(len(template_shifted)):
        plt.plot((geom[channels,0][:,None]+ \
                  np.arange(temps.shape[1])/5.).transpose() ,
                  template_shifted[k]*y_scale+geom[channels,1], 
                  c=cmap(k))

    plt.title("Shifted template: "+str(unit))
    plt.show()

    
def generate_synthetic_data(root_dir, 
                            rec_len_sec, sample_rate,
                            shifts, units,
                            temps,
                            radius,
                            geom):
    
    '''
    
    '''
    
    # parameter setting
    plotting = False
    n_units, n_times, n_chans = temps.shape
    
    # setup blank dataset
    data_synthetic = np.zeros((rec_len_sec*sample_rate, n_chans),'float32')
    print (data_synthetic.shape)

    # compute indexes for each shifted datachunk
    spike_chunks= np.linspace(0, rec_len_sec*sample_rate, shifts.shape[0])
    print (spike_chunks)

    n_spikes = []
    scales = []
    for k in range(units.shape[0]):
        times, scale = generate_poisson_uniform_firingrate2(rec_len_sec,
                                                           sample_rate)
        #print ("Unit: ", k, " spikes: ", times)
        n_spikes.append(times)
        scales.append(scale)

    #print (len(n_spikes))

    spike_train = np.zeros((0,2),'int32')
    # insert spikes every chunk of data
    for k in range(1,spike_chunks.shape[0]):

        window = [spike_chunks[k-1], spike_chunks[k]]

        print ("window: ", window, ", shift: ", shifts[k-1], " (inter channel units)")
        temps_insert = shift_templates_all(units, geom, temps.copy(), 
                                           np.array([shifts[k-1]]), 
                                           radius, 
                                           plotting)

        # use spike sorted spike_trains to inject (not currently used)
        if False:
            for ctr2, unit in enumerate(units):
                if (ctr2%50==0):
                    print (" inserting unit: ", ctr2, unit)
                idx = np.where(spike_train[:,1]==unit)[0]
                if idx.shape[0]<1:
                    continue
                times = np.int32(spike_train[idx,0])
                data_synthetic[times[:,None]+np.arange(101), :] += temps_insert[unit]


        # use poisson spike trains with isi-scaled spikes:
        else:
            #print (" number of scale list: ", len(scales))
            for ctr2, unit in enumerate(units):

                # load times previously generated within each window
                times = n_spikes[ctr2]
                idx = np.where(np.logical_and(times>window[0], times<=window[1]))[0]
                times = times[idx]
                
                # 
                scale1 = scales[ctr2]
                scale2 = scale1[idx]

                # generate ids and make spike_train
                idx = times*0+ctr2
                temp_train = np.vstack((times,idx)).T
                spike_train = np.vstack((spike_train, temp_train))
                if (ctr2%50==0):
                    print (" inserting unit: ", ctr2, unit)
                    #print (" times: ", times.shape, times[:10])
                    #print (" scale: ", scale2.shape, scale2[:10])
#                     print (temps_insert[unit].shape, scale.shape)
#                 print ("data synthetic: ", data_synthetic[times[:,None]+np.arange(101), :].shape)
#                 print ("first part: ", np.repeat(temps_insert[unit][None],scale.shape[0],axis=0).shape)
#                 print (" added part: ",  (np.multiply(
#                                     np.repeat(temps_insert[unit][None],scale.shape[0],axis=0).transpose(2,1,0),
#                                     scale).transpose(2,1,0).shape))
                
                #data_synthetic[times[:,None]+np.arange(101), :] += temps_insert[unit] * scale
                # add scaled data requires copies of templates and transposing the arrays x 2
                data_synthetic[times[:,None]+np.arange(101), :] += \
                        np.multiply(
                                    np.repeat(temps_insert[unit][None],scale2.shape[0],axis=0).transpose(2,1,0),
                                    scale2).transpose(2,1,0)
            print ("************")
            print ("")
            print ("")
            
    # save all data including raw binary
    np.save(root_dir + 'ground_truth/spike_train_ground_truth.npy',spike_train)
    
    return data_synthetic

def visualize_traces(data_synthetic, time1, time2):
    
    n_chans = data_synthetic.shape[1]
    
    for k in range(n_chans):
        plt.plot(data_synthetic[time1:time2,k]+k*20,c='black')
    plt.show()

    
def make_full_data(data_synthetic, rec_length_sample_times,
                  sample_rate, root_dir,
                  correlated_noise):
    
    data_sum = np.zeros(data_synthetic.shape,'int16')

    # white noise computation; add uniform noise
    white_noise = np.random.normal(0,1,size=correlated_noise.shape)
    print ("white noise: ", white_noise.shape)
    
    # 10 sec steps to match correlated noise width
    step = sample_rate*10
    for k in range(0,rec_length_sample_times, step):
        print ("time chunk: ", k, " to ", k+step)

        synthetic = data_synthetic[k:k+step]
        #print ("syntehtic chunk", synthetic.shape)
        
        data_sum[k:k+step]+= np.int16(synthetic*10.) + np.int16(correlated_noise*10.)+np.int16(white_noise)

    data_sum.tofile(root_dir+'data_int16.bin') 
    
    return data_sum
