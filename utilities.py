# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:35:27 2021

@author: TrangLT
"""

import sys
import os.path as op

from scipy import stats
from scipy.io import loadmat, savemat

sys.path.insert(1, 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\code\\python' )
import settings as st
st.init()

import numpy as np

import mne
from mne.minimum_norm import read_inverse_operator


def __init__():
    return


def compute_globalSNR(checkingSub, sub_id, task_id, semantic_group=0):
    
    """
    - Compute global signal-to-noise ratio (globalSNR) as verall signal strength of ERPs 
    combined for all words over all sensors each participants
    
    - Algorithm:
        
        + globalSNR is computed by dividing amplitude at each time-point by the standard deviation of the baseline period 
        (-200 ms before stimulus onset) 
        and then compute the square root of the sum of squares across all sensors
        
    - Usage:
        To determine interested time window
        
        
    - Reference:
        2013, Scientific Reports, Moseley

    - Parameters:
        checkingSub: list
                    list of subjects' index in experiment, retrieved from 'settings' file.
                    
        sub_id: integer
                interation index of subject to be processed 
                
        task_id: integer
                interation index of task to be processed 
                
        semantic_group: integer; 0: do on task (default); 4: face; 7: number; 10: animal
                for later usage, currently not use.
        
    Returns
    -------
    globalSNR: numpy.array
        Global signal-to-noise ratio for all words over all sensors for each participants.

    """
    
    globalSNR = np.zeros(len(st.times))
    
    # Extract epochs for current task
    path_task_r = op.join(st.path_eeglab_epoch, st.tasks[task_id]) 
            
    # Load eeglab-typed epoch data from all sessions
    fname_epoch = "S" + str(st.checkingSub[sub_id]) + "_" + st.tasks[task_id] + ".set" 
    epochs = mne.read_epochs_eeglab(op.join(path_task_r, fname_epoch))
    epochs = epochs.set_eeg_reference('average', projection=True)
    
    epochs = epochs.apply_baseline(st.baseline) # need to compute accurate noise covariance matrix
    epochs.crop(tmin=st.baseline[0])
 
    epochs = group_word_events_semantically(epochs)
        
    # Extract epochs corresponding to each class of semantic  
    evoked = epochs.average()
    temp = evoked._data

    # Compute global SNR for the current subject
    zeroTime = np.where(st.times >= 0)[0][0]
    baselineSTD = np.ndarray.std(temp[:,0:zeroTime],1) 
    temp = np.asarray([temp[ch,:]/ baselineSTD[ch] for ch in range(64) ])
    subjectSNR = np.sqrt(sum([(temp[ch,:])**2 for ch in range(64)]))
    
    globalSNR = globalSNR + subjectSNR

    del epochs
    
    return globalSNR


def group_word_events_semantically(taskid, epochs):
        
    """
    - Merge event for each word stimuli to semantic group:
        1,2,3 -> 'face': 1
        4,5,6 -> 'number': 2
        7,8,9 -> 'animal': 3

    
    Parameters
    -------
    
        taskid: integer
                id of task to be processed
                
        epochs: mne.Epochs
    
    Returns
    -------
    epochs: mne.Epochs
            epochs with events changed
    """
    
    # Corrected stimuli events
    events = epochs.events[:,2]
    events_id = epochs.event_id
    events_keys = [int(event) for event in list(events_id.keys())]
    
    events_values = [int(event) for event in list(events_id.values())]
    
    # create new events field for epochs as event_id.keys
    new_events = [events_keys[events_values.index(event)]
                  for event in events]

    new_events = np.where(np.asarray(new_events) < 4+taskid*100, 1,new_events)    
    new_events = np.where(6+taskid*100 < np.asarray(new_events), 3, new_events)
    new_events = np.where(3+taskid*100 < np.asarray(new_events) , 2, new_events)
    
    new_keys = ['face', 'number', 'animal']
    new_dict = dict(zip(new_keys, [1,2,3]))
     
    epochs.events[:,2] = [int(event) for event in new_events]
    epochs.event_id = new_dict
    
    return epochs


    
def extract_label_activation_word_semantic(pathStc, pathinv, sub_id, labels):
    """
    - Etract activation for all ROI for each words' semantic classes in current subject
    
    Parameters
    -------
        pathStc: string
                string of directory saving source estimate
                
        pathinv: string
                string of directory saving inverse solution
                
        sub_id: integer
                subject's index to be processed
                
        labels: mne.Label
                A FreeSurfer/MNE label with vertices restricted to one hemisphere
    
    Returns
    -------
    subROIface, subROInumber, subROIanimal: numpy.array
            3 data array of ROI's activation for all 3 class of words' semantic

    """
    # Load source estimate
    fnameStc = 'S' + str(st.checkingSub[sub_id]) + '_faceWords' 
    stcSubFaceWords = mne.read_source_estimate(pathStc + '\\' + fnameStc)
    
    fnameStc = 'S' + str(st.checkingSub[sub_id]) + '_animalWords' 
    stcSubAnimalWords = mne.read_source_estimate(pathStc + '\\' + fnameStc)
        
    fnameStc = 'S' + str(st.checkingSub[sub_id]) + '_numberWords' 
    stcSubNumberWords = mne.read_source_estimate(pathStc + '\\' + fnameStc)
        
     # Load invese operator
    fnameInv = 'S' + str(st.checkingSub[sub_id]) + '_numberWords-inv.fif' 
    srcSubNumbereWords = read_inverse_operator(pathinv + '\\' + fnameInv)
    
    fnameInv = 'S' + str(st.checkingSub[sub_id]) + '_faceWords-inv.fif' 
    srcSubFaceWords = read_inverse_operator(pathinv + '\\' + fnameInv)
    
    fnameInv = 'S' + str(st.checkingSub[sub_id]) + '_animalWords-inv.fif' 
    srcSubAnimalWords = read_inverse_operator(pathinv + '\\' + fnameInv)
    
    
    # Extract mean of activation over each sub-ROI
    subROIface = np.squeeze(np.asarray([stcSubFaceWords.extract_label_time_course(
        label,srcSubFaceWords['src'], mode='mean_flip') for label in labels]))
    
    subROInumber = np.squeeze(np.asarray([stcSubNumberWords.extract_label_time_course(
        label,srcSubNumbereWords['src'], mode='mean_flip') for label in labels]))
    
    subROIanimal = np.squeeze(np.asarray([stcSubAnimalWords.extract_label_time_course(
        label,srcSubAnimalWords['src'], mode='mean_flip') for label in labels]))
    
    
    return subROIface, subROInumber, subROIanimal

    
def mask_stats_results_on_brain(statsVal, labels, mask, figtitle, time_label, path_save, showplot=0):
    """
    - Visualizing t-stats or p-values in brain with respect to split parcellation

    Parameters
    ----------
        statsVal :  numpy array,   
        
        labels :    brain labels    
        
        mask:       np.array, array of mask values for significant ROI as 1, non-significant set to 0
        
        figtitle :  string, figure's title      
        
        time_label: integer, time label to display on the figure
        
        path_save:   string, path to directory for saving figure
                                  
        showplot:   integer, integer to indicate saving data for showing brain figures with masked data.
                    default: 0

    Returns
    -------
    stc_viz:        mne.stc, source estimate with data masked by results from PeSCAR test
    
    
    """    
    
    pathStc = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\listening\\sources\\stc\\'
    fname_stc = 'S1_animalWords'
    
    stc_viz = mne.read_source_estimate(pathStc + fname_stc) # Load source to visualize t-statistic data with mask    
        
    # Extract labels in left hemisphere and right hemisphere separately   
    lh_ROIlabelsId = [lh_id for lh_id in range( len(labels) ) if '-lh' in labels[lh_id].name]
    rh_ROIlabelsId = [rh_id for rh_id in range( len(labels) ) if '-rh' in labels[rh_id].name]
        
    # Assign mask values for left-hemisphere
    lh_data = np.zeros(len(stc_viz.lh_vertno))
    for ROIid in lh_ROIlabelsId:
        
        if mask[ROIid] > 0:
            print(labels[ROIid].name + '\t' + str(statsVal[ROIid]))
          
            
        # Extract vertices' indexes of labels used in sources estimated 
        vertIdx = labels[ROIid].get_vertices_used()  # Return indexes of sampling vertex in original labels        
        lh_data[vertIdx] = [statsVal[ROIid] * mask[ROIid]] * len(vertIdx)
        del vertIdx        
    
    # Assign mask values for right-hemisphere
    rh_data = np.zeros(len(stc_viz.rh_vertno))
    for ROIid in rh_ROIlabelsId:
        
        if mask[ROIid] > 0:
            print(labels[ROIid].name+ '\t' + str(statsVal[ROIid]))
            
        # Extract vertices' indexes of labels used in sources estimated 
        vertIdx = labels[ROIid].get_vertices_used()  # Return indexes of sampling vertex in original labels
        rh_data[vertIdx] = [statsVal[ROIid] * mask[ROIid]] * len(vertIdx)   
        del vertIdx 
  
    stc_viz._data = np.hstack((np.asarray(lh_data), np.asarray(rh_data)))
    stc_viz._tmin = time_label
    stc_viz._tmax = time_label
    stc_viz._tstep = 0
    
    stc_viz._times = np.array([time_label])
    
    # # Configuration for source plot
    # if showplot==1:

    #     # Load pragmatic labels for semantic presentation
    #     subjects_dir = 'U:\\usr\\local\\freesurfer\\7.2.0\\subjects\\'
    #     path_label = subjects_dir + '\\fsaverage\\label\\pragmatic_atlas\\'

    #     lpragmatic_semantic = np.load(path_label + 'lh.pragmatic-semantic.npz')
    #     rpragmatic_semantic = np.load(path_label + 'rh.pragmatic-semantic.npz')
    #     pragmatic_info = np.load(path_label + 'pragmatic-info.npz')

    #     # Correct the order for significant semantic areas as there is only 77 areas for the left and 63 areas for the right
    #     # so other non-significant areas has the same name for the label of [l/r]'h..label'

    #     # get corresponding index of non-singificant areas which has empty name in left and right hemisphere
    #     llabels_name = [label_name.decode('UTF-8') for label_name in pragmatic_info['lnames'] ]
    #     rlabels_name = [label_name.decode('UTF-8') for label_name in pragmatic_info['rnames'] ]

    #     llabels  = [mne.read_label(path_label + 'lh.' + label_name + '.label', 'fsaverage') for label_name in  llabels_name]
    #     rlabels  = [mne.read_label(path_label + 'rh.' + label_name + '.label', 'fsaverage') for label_name in  rlabels_name]

    #     # Plot brain
    #     hemi = 'split'
    #     views = 'lateral' # 'medial'
    #     surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
    #        hemi=hemi, views=views, size=(1200,400), title = figtitle, initial_time=time_label, 
    #        time_unit='ms', background='black')
    #     brain = mne.viz.plot_source_estimates(stc_viz,**surfer_kwargs)
        
    #     [brain.add_label(llabels[i], borders=True, color = lpragmatic_semantic['rgb'][i]) for i in range(len(llabels_name)) if llabels_name[i] != '']
    #     [brain.add_label(rlabels[i], borders=True, color = rpragmatic_semantic['rgb'][i]) for i in range(len(rlabels_name)) if rlabels_name[i] != '']
    #     import time
    #     time.sleep(5)
    #     brain.add_text(0.1, 0.9, figtitle, 'title', font_size=11)
    #     fname = path_save + '\\' +views + ' ' + figtitle[2:len(figtitle)-24] + '.png'
    #     brain.save_image(fname)
    #     brain.close()
            
    return stc_viz

             

def visualize_source(taskid, stc, figtitle, zscore=True):
    """
    

    Parameters
    ----------
    taskid : integer
        task id.
    stc : source estimate object
        
    times : array
        array of time to show.
        
    figtitle : string
        figure's title.
        
    zscore : boolean, optional
        plot zscore of sources or pure source data. The default is True.

    Returns
    -------
    None.

    """
    
    
    hemi = 'split'
    if zscore==True:
        # Compute zscore
        stc_zscore = stc
        temp_zscore = stats.zscore(stc.data, axis=0, ddof=1)
        stc_zscore.data = temp_zscore
        del temp_zscore
        
        stc = stc_zscore
        
    surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
        hemi=hemi, views='lateral', size=(1000,400), smoothing_steps=5)

    brain = mne.viz.plot_source_estimates(stc,**surfer_kwargs) 
    # brain = stc.plot(**surfer_kwargs) 
    brain.add_text(0.1, 0.9,figtitle, 'title', font_size=11)



def find_ROIsubId(orglabelsname,splitlabelsname):
    """
    Create an array of ROIs'id which each subROI is belong to


    Parameters
    ----------
    orglabelsname : list of string
                    list of labels's name corresponding to original parcellation

    splitlabelsname : list of string
                    list of labels's name corresponding to splitted parcellation
                    Ref: 
        
        
    Returns
    -------
    ROIsubId: np.array 
                Array of indexes of ROI which subROI is belong to.

    """
    ROIsubId = np.zeros(len(splitlabelsname))
    
    for ROIi in range(len(orglabelsname)-1):
    
        hemi =  orglabelsname[ROIi][-2:]
        prefixROIname = orglabelsname[ROIi][:-3]
        
        for subROIi in range(len(splitlabelsname)):
            
            subROIname = splitlabelsname[subROIi]
            if subROIname[-2:] == hemi and subROIname[:len(prefixROIname)] == prefixROIname :
                ROIsubId[subROIi] = int(ROIi)     
    
        
        # "unknown" region is only located in left hemisphere in the original labels, so both left and right division of "unknown" is numbered with same value
        ROIsubId[-2:] = int(68)
            
        del hemi
        del prefixROIname
    return ROIsubId
       


def extract_mean_activation_masked_sources(labels, pathMain, task_id, timewins, saveOption):
    
    mask = loadmat(pathMain + 'semantic_processing\\' + st.tasks[task_id] + '\\fdrMaskSubROIallGroup.mat')
    mask = mask['fdrMaskSubROIallGroup']
    
    ntimewins = len(timewins)
    
    for si, sub_id in enumerate(st.checkingSub):
        
        fname = 'S' + str(sub_id)
        stc = mne.read_source_estimate(pathMain + 'sources\\'  + st.tasks[task_id] + '\\stc\\' + fname)
        
        #####################################################################################
        
        # mask source data with 0 if no significant and 1 if significant after PeSCAR
        
        #####################################################################################


        # Extract labels in left hemisphere and right hemisphere separately   
        lh_ROIlabelsId = [lh_id for lh_id in range( len(labels) ) if '-lh' in labels[lh_id].name]
        rh_ROIlabelsId = [rh_id for rh_id in range( len(labels) ) if '-rh' in labels[rh_id].name]
            
        # Assign mask values for left-hemisphere
        lh_data = np.zeros(len(stc.lh_vertno))
        for ROIid in lh_ROIlabelsId:           
                            
            # Extract vertices' indexes of labels used in sources estimated 
            vertIdx = labels[ROIid].get_vertices_used()  # Return indexes of sampling vertex in original labels        
            lh_data[vertIdx] = stc._data[vertIdx] * mask[ROIid]
            del vertIdx        
        
        # Assign mask values for right-hemisphere
        rh_data = np.zeros(len(stc.rh_vertno))
        for ROIid in rh_ROIlabelsId:
                            
            # Extract vertices' indexes of labels used in sources estimated 
            vertIdx = labels[ROIid].get_vertices_used()  # Return indexes of sampling vertex in original labels
            rh_data[vertIdx] = stc._data[vertIdx] * mask[ROIid]   
            del vertIdx 
        
        stc._data = np.hstack((np.asarray(lh_data), np.asarray(rh_data)))
        
        
        meanActivation = np.zeros((len(stc._data), ntimewins))

        for timei in range(len(timewins)):
            timeIdx = [np.argwhere(st.times>timewins[timei][0])[0,0],np.argwhere(st.times>timewins[timei][1])[0,0] ]
            meanActivation[:,timei] =  np.mean( stc._data[:,timeIdx[0] : timeIdx[1]] )               
            
   
        if saveOption == 1:
            savemat( pathMain + 'sources\\'  + st.task[task_id] +'\\meanActivation\\S' +str(sub_id) +'.mat', {'activation':meanActivation})
        

    
    

 
