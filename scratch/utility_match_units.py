
import matplotlib
matplotlib.use('Agg')

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

from numba import jit

from matplotlib_venn import venn2, venn2_circles

from numba import jit
# functions
def binary_reader_waveforms(filename, n_channels, n_times, spikes, data_type='float32'):
    ''' Reader for loading raw binaries
    
        standardized_filename:  name of file contianing the raw binary
        n_channels:  number of channels in the raw binary recording 
        n_times:  length of waveform 
        spikes: 1D array containing spike times in sample rate of raw data
        channels: load specific channels only
        data_type: float32 for standardized data
    
    '''

    # ***** LOAD RAW RECORDING *****
    wfs=[]
    if data_type =='float32':
        data_len = 4
    else:
        data_len = 2
    
    #filename = '/media/cat/1TB/data/synthetic/run5/data_int16.bin'
    with open(filename, "rb") as fin:
        for ctr,s in enumerate(spikes):
            # index into binary file: time steps * 4  4byte floats * n_channels
            if True:
            #try:
                fin.seek(s * data_len * n_channels, os.SEEK_SET)
                
                temp = np.fromfile(
                    fin,
                    dtype='int16',
                    count=(n_times * n_channels))
                                
                wfs.append(temp.reshape(n_times, n_channels))

            #except:
            #    print ("can't load spike: ", s)
            #    pass
    fin.close()
    return np.array(wfs)

@jit
def match_units(s1, s2):
    ctr=0
    max_diff = 15
    matched_spike_times = []
    for s in s1:
        if np.min(np.abs(s2-s))<=max_diff:
            ctr+=1
            matched_spike_times.append(s)
            
    return (ctr, matched_spike_times)


def search_spikes_parallel(units_ground_truth,
                      templates_gt,
                      max_chans_sorted,
                      spike_train_gt,
                      spike_train_sorted):
    n_spikes = []
    ids_matched = []
    ptp_gt_unit = []
    purity = []
    completeness = []
    matched_spikes_array = []
    n_spikes_array = []
    venn_array = []
    unit_ids = []
    print ("units_ground_truth: ", units_ground_truth)
    for unit in units_ground_truth:
        #ptps_unit = templates_sorted[unit].ptp(0)
        ptps_unit = templates_gt[unit].ptp(0)
        #max_chans_unit = np.argsort(ptps_unit)[::-1][:40]  #check largest nearby channels;
        max_chans_unit = np.argsort(ptps_unit)[::-1]  #check largest nearby channels;
       
        # find nearest gt templates
        ids_nearest_sorted = np.where(np.in1d(max_chans_sorted, max_chans_unit))[0]

        #print (" Matching unit: ", unit, " , with ", ids_nearest_sorted)

        # search spike time matches
        match_spikes = []
        all_spikes = []
        match_ids = []
        matched_spike_times_local = []
        idx2 = np.where(spike_train_gt[:,1]==unit)[0]
        # save n_spikes for each unit
        venn_spikes = []
        for id_ in ids_nearest_sorted:
            idx_3 = np.where(spike_train_sorted[:,1]==id_)[0]
            matches, matched_spike_times = match_units(spike_train_gt[idx2,0], 
    #                              spike_train[idx2,0])
                                  spike_train_sorted[idx_3,0])
            if unit == 0: 
                print ("searching match unit: ", id_)
                print ("# spikes in sorted unit: ", idx_3.shape[0])
                print ("# matches: ", matches)
            #if matches>match_spikes:
            match_ids.append(id_)
            match_spikes.append(matches)
            all_spikes.append(idx_3.shape[0])
            matched_spike_times_local.append(matched_spike_times)
            venn_spikes.append(matches)

        venn_array.append(venn_spikes)
        n_spikes_array.append(idx2.shape[0])
        #purity.append(match_spikes/float(idx2.shape[0]))
        #completeness.append(match_spikes/float(all_spikes))

        # this saves sample times for all the matches spikes so waveforms can be loaded later;
        matched_spikes_array.append(matched_spike_times_local)

        # save original id plus match_id
        ids_matched.append([unit,match_ids])

        # save ptp of gt unit
        ptps_unit = templates_gt[unit].ptp(0).max(0)
        ptp_gt_unit.append(ptps_unit)
        #print ("done matching: ", unit)
        unit_ids.append(unit)
        
    return (n_spikes_array, ids_matched, ptp_gt_unit, purity,
            completeness, matched_spikes_array,
            n_spikes_array, venn_array, unit_ids)


def search_spikes_parallel_single_unit(unit_ground_truth,
                                      templates_gt,
                                      max_chans_sorted,
                                      spike_train_gt,
                                      spike_train_sorted,
                                      root_dir):

                                
    # check to see if data laready saved
    fname = root_dir+'/matches/'+str(unit_ground_truth)+'.npz'
    if os.path.exists(fname)==False:
        
        n_spikes = []
        ids_matched = []
        ptp_gt_unit = []
        purity = []
        completeness = []
        matched_spikes_array = []
        n_spikes_array = []
        venn_array = []
        unit_ids = []
        
        # set unit the one loaded
        unit = unit_ground_truth

        ptps_unit = templates_gt[unit].ptp(0)
        #max_chans_unit = np.argsort(ptps_unit)[::-1][:40]  #check largest nearby channels;
        max_chans_unit = np.argsort(ptps_unit)[::-1]  # check against all units in the ground truth datasets; don't limit this any mnore
       
        # find nearest gt templates
        ids_nearest_sorted = np.where(np.in1d(max_chans_sorted, max_chans_unit))[0] # 

        #print (" Matching unit: ", unit, " , with ", ids_nearest_sorted)

        # search spike time matches
        match_spikes = []
        all_spikes = []
        match_ids = []
        matched_spike_times_local = []
        idx2 = np.where(spike_train_gt[:,1]==unit)[0]
        # save n_spikes for each unit
        venn_spikes = []
        for id_ in ids_nearest_sorted:
            idx_3 = np.where(spike_train_sorted[:,1]==id_)[0]
            matches, matched_spike_times = match_units(spike_train_gt[idx2,0], 
    #                              spike_train[idx2,0])
                                  spike_train_sorted[idx_3,0])
            if unit == 0: 
                print ("searching match unit: ", id_)
                print ("# spikes in sorted unit: ", idx_3.shape[0])
                print ("# matches: ", matches)
                
            #if matches>match_spikes:
            match_ids.append(id_)
            match_spikes.append(matches)
            all_spikes.append(idx_3.shape[0])
            matched_spike_times_local.append(matched_spike_times)
            venn_spikes.append(matches)

        venn_array.append(venn_spikes)
        n_spikes_array.append(idx2.shape[0])
        #purity.append(match_spikes/float(idx2.shape[0]))
        #completeness.append(match_spikes/float(all_spikes))

        # this saves sample times for all the matches spikes so waveforms can be loaded later;
        matched_spikes_array.append(matched_spike_times_local)

        # save original id plus match_id
        ids_matched.append([unit,match_ids])

        # save ptp of gt unit
        ptps_unit = templates_gt[unit].ptp(0).max(0)
        ptp_gt_unit.append(ptps_unit)
        #print ("done matching: ", unit)
        unit_ids.append(unit)
        
        np.savez(fname, n_spikes_array=n_spikes_array, 
                ids_matched=ids_matched, 
                ptp_gt_unit=ptp_gt_unit, 
                purity=purity,
                completeness=completeness, 
                matched_spikes_array=matched_spikes_array,
                venn_array=venn_array, 
                unit_ids=unit_ids)
            

def load_ks2_spikes(root_dir, n_channels,n_times):

    fname_out = root_dir + 'spike_train_final.npy'

    if os.path.exists(fname_out)==False:

        fname = root_dir+'/data_int16.bin'
        data_type = 'int16'

        #n_channels = 384
        #n_times = 101

        # load KS2 sorted times
        times = np.load(root_dir+'spike_times.npy')-n_times//2
        ids = np.load(root_dir +'spike_clusters.npy')

        # reorer the KS2 spike_train
        ctr=0
        for k in np.unique(ids):
            idx = np.where(ids==k)[0]
            ids[idx]=ctr
            ctr+=1

        spike_train = np.hstack((times,ids))
        print (spike_train.shape)

        units = np.unique(spike_train[:,1])
        print ("units: ", units.shape)
        ids, counts = np.unique(spike_train[:,1], return_counts=True)

        # parse KS2 units and keep only ones with minimum or max firing rates
        min_spikes = 600/4
        max_spikes = 6000*500

        good_units_ids = np.where(np.logical_and(counts>=min_spikes, counts<=max_spikes))[0]
        print ("# of good units: ", good_units_ids.shape[0], " of total KS2 units: ", ids.shape[0])

        # reorder the spikes_going forward
        ctr=0
        spike_train_final = np.zeros((0,2),'int32')
        for good_unit in good_units_ids:
            idx= np.where(spike_train[:,1]==good_unit)[0]
            spike_train[idx,1]=ctr

            spike_train_final = np.int32(np.vstack((spike_train_final,spike_train[idx])))
            ctr+=1

        print (spike_train_final)

        spike_train = spike_train_final.copy()
        print (" DONE ")

        # save spike train and templates_reloaded
        np.save(fname_out, spike_train)
        print(spike_train)

    else:
        spike_train = np.load(fname_out)
    
    return spike_train

def reload_ks2_templates(root_dir, spike_train, data_type,
                         n_channels, n_times):
    
    fname_out = root_dir + '/templates_reloaded_good.npy'
    if os.path.exists(fname_out)==False:
        
        # name of binary file
        fname = root_dir + 'data_int16.bin'
        
        templates = []
        ptps = []
        time_start = 0
        time_end = time_start+600
        wfs_array = []
        times = []
        for ctr, unit in enumerate(np.unique(spike_train[:,1])):#[:20]:
        #for ctr, unit in enumerate([248]):#[:20]:
            idx = np.where(spike_train[:,1]==unit)[0]
            spikes = np.int32(spike_train[idx,0]) #-n_times//2#-30

            # sub sample spikes to speed up loading
            idx = np.where(np.logical_and(spikes>=(time_start*30000), spikes<(time_end*30000)))[0]
            spikes = spikes[idx]
            if idx.shape[0]==0:
                ptps.append([])
                templates.append(np.zeros((n_times,n_channels)))
                times.append([])
                continue

            times.append(spikes)
            #print (spikes.shape)

            idx2 = np.where(spikes<60*30000)[0]
            spikes=spikes[idx2]

            wfs = binary_reader_waveforms(fname, n_channels, n_times, spikes, data_type)

            if wfs.shape[0]==0:
                wfs = np.zeros((10,n_times,n_channels))
                continue 
            # save template using only first 60 sec of data;
            temp = wfs.mean(0)
            templates.append(temp)

            print (ctr, '/', np.unique(spike_train[:,1]).shape[0], ' raw id: ', unit, 
                   wfs.shape, temp.shape, "oiringla spikes: ", idx.shape)

        np.save(fname_out, templates)
    else:
        templates = np.load(fname_out)

    return np.array(templates)

class Match_to_ground_truth(object):

    def __init__(self, root_dir, spike_train_sorted, templates_sorted):
        
        self.root_dir = root_dir
        self.spike_train_sorted = spike_train_sorted
        self.templates_sorted = templates_sorted
        
        self.n_times = templates_sorted.shape[1]
        self.n_channels = templates_sorted.shape[2]
        
        # load ground truth data
        self.spike_train_gt = np.load(self.root_dir + 'ground_truth/spike_train_ground_truth.npy')
        self.templates_gt = np.load(self.root_dir + 'ground_truth/templates_ground_truth.npy')
        print (" ground truth templates: ", self.templates_gt.shape)
        self.units_ground_truth = np.arange(self.templates_gt.shape[0])
        self.max_chans_gt = self.templates_gt.ptp(1).argmax(1)
        #print ("max chans gt: ", self.max_chans_gt)

        # load sorted templates
        #print ("Templates sorted: ", self.templates_sorted.shape)
        self.units_sorted = np.arange(self.templates_sorted.shape[0])
        self.max_chans_sorted = self.templates_sorted.ptp(1).argmax(1)

        #print ("calling matchi units")
        self.match_units()
        
    def match_units(self):
        fname = self.root_dir + 'matches_res.npy'
        
        try:
            os.mkdir(self.root_dir+'/matches/')
        except:
            pass
            
        #if os.path.exists(fname)==False:

            # self.units_split = np.array_split(self.units_ground_truth, 6)
            # res = parmap.map(search_spikes_parallel,
                              # self.units_split,
                              # self.templates_gt,
                              # self.max_chans_sorted,
                              # self.spike_train_gt,
                              # self.spike_train_sorted,
                              # pm_processes=6)
            
        parmap.map(search_spikes_parallel_single_unit,
                              self.units_ground_truth,
                              self.templates_gt,
                              self.max_chans_sorted,
                              self.spike_train_gt,
                              self.spike_train_sorted,
                              self.root_dir,
                              pm_processes=6)

        #else:
        #    res = np.load(fname, allow_pickle=True)

        self.n_spikes = []
        self.ids_matched = []
        self.ptp_gt_unit = []
        self.purity = []
        self.completeness = []
        self.matched_spikes_array = []
        self.n_spikes_array = []
        self.venn_array = []
        self.unit_ids = []
        #print (" # chunks: ", len(res))

        for unit in range(self.units_ground_truth.shape[0]):
                     
            fname =  self.root_dir+'/matches/'+str(unit)+'.npz'
            res = np.load(fname,allow_pickle=True)
                # n_spikes_array=n_spikes_array, 
                # ids_matched=ids_matched, 
                # ptp_gt_unit=ptp_gt_unit, 
                # purity=purity,
                # completeness=completeness, 
                # matched_spikes_array=matched_spikes_array,
                # venn_array=venn_array, 
                # unit_ids=unit_ids)
            
                # load data
            self.n_spikes.extend(res['n_spikes_array'])
            self.ids_matched.extend(res['ids_matched'])
            self.ptp_gt_unit.extend(res['ptp_gt_unit'])
            self.purity.extend(res['purity'])
            self.completeness.extend(res['completeness'])
            self.matched_spikes_array.extend(res['matched_spikes_array'])
            self.n_spikes_array.extend(res['n_spikes_array'])
            self.venn_array.extend(res['venn_array'])
            self.unit_ids.extend(res['unit_ids'])


    def make_pie_charts(self, n_matches2):
        
        print ("n matches: ", n_matches2)
        fig =plt.figure()
        colors = ['blue','red','green','magenta',
                  'cyan','yellow','pink','orange',
                  'brown','darkgreen']

        ptps = self.templates_gt.ptp(1).max(1)

        #n_spikes_array.
        for k in range(len(self.venn_array)):
            ax=plt.subplot(10,10,k+1)

            n_matches = len(self.venn_array[k])
            sizes = np.sort(self.venn_array[k])[::-1][:n_matches2]

            clrs = colors[:sizes.shape[0]]
            print ("sizes:" , sizes, " , clrs: ", clrs)
            # if the total number of spikes found is smaler than all spikes injected, add black piechart
            if sizes.sum(0)<self.n_spikes_array[k]:
                sizes = np.append(sizes, np.array(self.n_spikes_array[k]-sizes.sum(0)))
                clrs = np.append(clrs,'black')

            plt.pie(sizes, colors=clrs)
            plt.title(str(k)+" ptp: " +str(np.round(ptps[k],1)), fontsize=8)

        plt.suptitle("Drift simulation of injected neurons drift: ..."+
                     " sorted neurons (blue=best match, red=second best, green=third...)"+
                     "\n(sum matches can be > 100% depending on oversplits/duplicate units)",fontsize=12)
        plt.show()

# plot single unit max channel template and scatter plot
def plot_single_unit(selected_unit,
                     root_dir,
                     spike_train_sorted,
                     n_matches2, clrs,
                     scale_amplitude,
                     save_fig):
                         
    spike_train_gt = np.load(root_dir + 'ground_truth/spike_train_ground_truth.npy')
    templates_sorted = np.load(root_dir + 'templates_reloaded_good.npy')
    max_chans_sorted = templates_sorted.ptp(1).argmax(1)

    # fix n-chans and sample rate for computations below
    n_chans = templates_sorted.shape[2]
    sample_rate = 30000
    
    fname_int16 = root_dir + 'data_int16.bin'
    data_type = 'int16'

    # load units
    matcher = Match_to_ground_truth(root_dir, spike_train_sorted, templates_sorted)

    matched_units = np.array(matcher.ids_matched[selected_unit][1])
    print ("matched_units: ", matched_units)

    # count # of spikes in each matching unit;
    n_spk = []
    for k in range(len(matcher.matched_spikes_array[selected_unit])):
        n_spk.append(len(matcher.matched_spikes_array[selected_unit][k]))
    
    # select top 5 matching units by size and plot them;
    n_spk = np.hstack(n_spk)
    idx_sorted = np.argsort(n_spk)[::-1][:n_matches2]

    # load waveforms for sorted spikes
    ptps_sorted = []
    times_sorted = []
    ptps_sorted_all = []
    times_sorted_all = []
    for id_ in idx_sorted:
        print (" unit: ", selected_unit,", matching unit: ", id_,
              matched_units[id_])
        
        # match spikes in the sorted unit
        idxall = np.where(matcher.spike_train_sorted[:,1]==
                          matched_units[id_])[0]
        
        spk = matcher.spike_train_sorted[idxall,0]

        temp2 = binary_reader_waveforms(fname_int16, matcher.n_channels, matcher.n_times, spk, data_type)

        if temp2.shape[0]==0:
            ptps_sorted_all.append([])
            times_sorted_all.append([])
        else:
            times_sorted_all.append(spk)

            # compute ptps of matched spikes
            temp3 = temp2.mean(0)
            max_chan = temp3.ptp(0).argmax(0)

            # select fixed points of waveform max/min at which to compute PTP
            max_ = np.argmax(temp3[:,max_chan])
            min_ = np.argmin(temp3[:,max_chan])
            ptps_local =np.array(temp2[:,max_,max_chan]-
                                 temp2[:,min_,max_chan])

            ptps_local = ptps_local/scale_amplitude
            ptps_sorted_all.append(ptps_local)

        # ***************************************
        # DO THE SAME BUT ONLY FOR MATCHING SPIKE TIMES
        spk = np.array(matcher.matched_spikes_array[selected_unit][id_])
        idx = np.argsort(spk)
        spk=spk[idx]

        temp1 = binary_reader_waveforms(fname_int16, matcher.n_channels, matcher.n_times, spk, data_type)
        if temp1.shape[0]==0:
            ptps_sorted.append([])
            times_sorted.append([])
        else:

            times_sorted.append(spk)

            # compute ptps of matched spikes
            temp2 = temp1.mean(0)
            max_chan = temp2.ptp(0).argmax(0)

            # select fixed points of waveform max/min at which to compute PTP
            max_ = np.argmax(temp2[:,max_chan])
            min_ = np.argmin(temp2[:,max_chan])
            ptps_local =np.array(temp1[:,max_,max_chan]-
                                 temp1[:,min_,max_chan])

            ptps_local = ptps_local/scale_amplitude
            ptps_sorted.append(ptps_local)
        
        print ("")

    # ****************** GROUND TRUTH UNIT COMPUTATION ***************
    # load spikes for ground truth unit:
    print (" loading injected unit: ", selected_unit)
    idx = np.where(spike_train_gt[:,1]==selected_unit)[0]
    spk_gt = spike_train_gt[idx,0]
    tot_spikes_gt = spk_gt.shape[0]
    #idx = np.argsort(spk_gt)
    #spk_gt=spk_gt[idx]
    
    wfs_gt = binary_reader_waveforms(fname_int16, matcher.n_channels, matcher.n_times, spk_gt, data_type)
    print (" ground truth wfs: ", wfs_gt.shape)    

    # compute ptps for the ground truth unit
    # select only first 1 minute of data to find peak/trough to limit drift artifacts
    idxt = np.where(spk_gt<sample_rate*60)[0]    
    temp2 = wfs_gt[idxt].mean(0)
    max_chan = temp2.ptp(0).argmax(0)
    max_ = np.argmax(temp2[:,max_chan])
    min_ = np.argmin(temp2[:,max_chan])
    
    # then compute ptp for all loaded data;
    ptps_gt =np.array(wfs_gt[:,max_,max_chan]-
                         wfs_gt[:,min_,max_chan])
    ptps_gt = ptps_gt/scale_amplitude

    # *************** PLOT RESULTS ************
    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(n_matches2+1, 5)
    #gs.update(wspace=0.05, hspace=0.05)
        
    # ************ PLOT WAVEFORMS **************
    cmap = cm.get_cmap('viridis',wfs_gt.shape[0])
    ax = plt.subplot(gs[0, 0])

    # plot only 10% of spikes
    for k in range(0, wfs_gt.shape[0],wfs_gt.shape[0]//10):  
    #for k in range(0, wfs_gt.shape[0],1):        
        temp = wfs_gt[k,:,max_chan].T/10.
        plt.plot(temp, c=cmap(k),alpha=.05)

    del wfs_gt

    ax.set_title("Max chan spikes (color=time; 10%)",fontsize=14)
    plt.ylabel("Injected\ntemplate\n(max-chan; SU)", fontsize=14)
    
    # ************ SCATTER PLOT GROUND TRUTH UNIT **************
    ax = plt.subplot(gs[0, 1:])
    plt.scatter(spk_gt/30000., ptps_gt,s=150, c='black', alpha=.1)
    plt.xticks([])
    plt.ylim(bottom=0)
    plt.title("PTP of ground truth injected unit", fontsize=14)

    # ************ SCATTER PLOT BEST X MATCHES **************
    print ("len ptpssorted: ", len(ptps_sorted))
    # track TP rates; 
    TP_rate = []
    for k in range(len(ptps_sorted)):
        ax = plt.subplot(gs[k+1, 1:])
        
        # plot all PTP values for all spikes in sorted unit   
        ax.scatter(times_sorted_all[k]/30000., ptps_sorted_all[k],s=150, 
                    c=clrs[k], alpha=.1)    
            
        # plot PTP values for sorted spikes matching ground truth unit
        if len(times_sorted[k])>0:
            ax.scatter(times_sorted[k]/30000., ptps_sorted[k],s=150, c='black', alpha=.1)
        
        # label infoormation
        plt.ylim(bottom=0)
        
        size = np.sort(matcher.venn_array[selected_unit])[::-1][k]
        tot_spikes = matcher.n_spikes_array[selected_unit]  # spikes in the ground truth unit
        tp_rate = size/float(tot_spikes)   
        
        # compute purity and compleness
        id_ = idx_sorted[k]
        
        # match spikes in the sorted unit
        n_spikes_sorted_unit = np.where(matcher.spike_train_sorted[:,1]==
                          matched_units[id_])[0].shape[0]
        
        purity = size/float(n_spikes_sorted_unit)*100.
        completeness = size/float(tot_spikes)*100.
        
        plt.title("Match #"+str(k)+ ", Purity: "+str(round(purity,1))+"%"+
                  ", Completness: "+str(round(completeness,1))+"%", fontsize=14)
        if k <(len(ptps_sorted)-1):
            plt.xticks([])
        
        TP_rate.append(tp_rate)

    # ************ VENN DIAGRAMS GT vs. BEST MATCH **************
    # plot venn diagrams for best match vs. ground truth unit:
    FP_rate = []
    for k in range(n_matches2):
        ax = plt.subplot(gs[k+1, 0])
        idx_gt = np.where(matcher.spike_train_gt[:,1]==selected_unit)[0]
        times_gt = matcher.spike_train_gt[idx_gt,0]

        idx_sort = np.where(matcher.spike_train_sorted[:,1]==matched_units[idx_sorted[k]])[0]
        times_sort = matcher.spike_train_sorted[idx_sort,0]

        vintersect = n_spk[idx_sorted][k]
        v1 = times_gt.shape[0] - vintersect
        v2 = times_sort.shape[0] - vintersect

        vd = venn2(subsets = (v1, v2, vintersect), set_labels = ('ground\ntruth', 
                                                                 'sorted\nunit'))
        vd.get_patch_by_id("100").set_color("black")
        vd.get_patch_by_id("010").set_color(clrs[k])
        ax.set_title("Sorted unit: "+str(k), color='black', 
                     rotation='vertical',x=-0.3,y=0)
        
        FP_rate.append(v2/float(times_sort.shape[0]))
        
    # ************ LABELS **************
    plt.xlabel("Time (sec)", fontsize=14)
    
    temp = os.path.getsize(fname_int16)
    rec_len_sec = temp//2//n_chans//sample_rate
    print ("rec len sec: ", rec_len_sec)
    plt.suptitle("Injected unit: "+str(selected_unit)+", "+\
                 str(round(tot_spikes_gt/float(rec_len_sec),1))+"Hz"+
                 ", # spks: "+str(v1),fontsize=14)
        
    if save_fig:
        # save in good data directory, or in bad data directory
        TP_rate = np.array(TP_rate)
        FP_rate = np.array(FP_rate)
        if (TP_rate[0]<0.95) or np.any(TP_rate[1:]>0.10) or (FP_rate[0]>0.05):
            fname_out =  matcher.root_dir+"figs/bad_matches/"+str(selected_unit)+'.png'
        else:
            fname_out =  matcher.root_dir+"figs/good_matches/"+str(selected_unit)+'.png'

        #plt.savefig(fname_out,dpi=100)
        fig.savefig(fname_out,dpi=100)
        plt.close(fig)


    else:
        plt.close(fig)
        print ("can't show figure or will crash xserver...")
        #plt.show()


# plot single unit max channel template and scatter plot
def plot_single_unit_parallel(units, 
                     root_dir,
                     spike_train_sorted, 
                     n_matches2, 
                     clrs,
                     scale_amplitude,
                     save_fig,
                     n_cores):
                               
    #matcher = Match_to_ground_truth(
                               #spike_train_sorted, 
                               #templates_sorted, 

    try:
        os.mkdir(root_dir+"figs/")
    except:
        pass
    try:
        os.mkdir(root_dir+"figs/good_matches/")
    except:
        pass
    try:
        os.mkdir(root_dir+"figs/bad_matches/")
    except:
        pass
            
            
    # parallel version
    if True:
        parmap.map(plot_single_unit, units,
                 root_dir,
                 spike_train_sorted, 
                 n_matches2, 
                 clrs,
                 scale_amplitude,
                 save_fig,
                 pm_processes=n_cores)
    else:
        for unit in units:
            plot_single_unit (unit,
                 root_dir,
                 spike_train_sorted, 
                 n_matches2, 
                 clrs,
                 scale_amplitude,
                 save_fig)

    #spike_train_gt = np.load(r
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
