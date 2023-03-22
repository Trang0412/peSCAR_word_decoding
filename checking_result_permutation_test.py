# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:24:03 2021

1. Visualizing permutation t-test result with mask on
2. Extracting indexes of unique subROIs showing difference in one of 2 permutation t-tests

@author: TrangLT
"""
#%%
import os
import sys
from scipy.io import loadmat, savemat
from statsmodels.stats.multitest import fdrcorrection

import numpy as np
import seaborn as sns
import pandas as pd
import mne


sys.path.insert(1, 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\code\\python\\semantic_processing')
import utilities as ut

sys.path.insert(1, 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\code\\python' )
import settings as st
st.init()

orglabelsname = [label.name for label in st.orglabels]
splitlabelsname = [label.name for label in st.splitlabels]
ROIsubId = ut.find_ROIsubId(orglabelsname,splitlabelsname)

comparisons = st.comparisons
#%% Run PeSCAR for each time window

def create_fdr_mask(criticalTval, alpha, taskid, compi, l):
    """
    - Create a mask on source estimate with 1 represents significant difference and ) represents non-significant

    Inputs:
        criticalTval: float, t-value for defining critical interval
        alpha: float, p-value for defining confidence interval
        taskid: int, task id
        compi: int, index of comparison
        l: int, index of interested time window

    Outputs:
        fdrMaskSubROIs: np.array, (1,450) shaped array of 1 or 0 indicating fdr mask for each sub ROI corresponding to current time window
                                keys of ccurrent omparisos, e.g., 'FaceNum', 'AnimalNum','AnimalFace'.
                                
    """

    if taskid == 0:
        latencies = [[0.106, 0.133],[0.196, 0.231],[0.352, 0.407],[0.458, 0.485]] # listening
    elif taskid == 1:
        latencies = [[0.063, 0.134],[0.173, 0.243],[0.274, 0.325],[0.446, 0.477],[0.580, 0.657]] # imagined
    else:
        latencies =[[-0.2, 0], [0.134, 0.177], [0.189, 0.216], [0.240, 0.302], [0.384, 0.497], [0.560, 0.689] ]# overt task

    # criticalTval = 2.021 # 2 tail t-test for 40 sample with critical value is set to 0.01
    # alpha = 0.05 # for FDR correction
    path_pescar = st.pathDatSem +  st.tasks[taskid]  + "\\peSCAR\\"
    latency = latencies[l-1]
    
    # Load observed t- values and permutation t-values data  
    obsTvalsubROIs = loadmat(path_pescar + "obsTvalsubROIs_" + str(int(latency[0]*1000)) + "_" + str(int(latency[1]*1000)) + ".mat") # t-statistic from osberved data
    permsTvalsubROIs = loadmat(path_pescar + "permsTvalsubROIs_" + str(int(latency[0]*1000)) + "_" + str(int(latency[1]*1000)) + ".mat") # p-values from permutation t-test
          
    #% Find p-value for each ROI 
    # create a mask matrix which subROI show significant p-value will be masked with value of 1 otherwise 0
    maskSubROIs = np.where(np.abs(obsTvalsubROIs[comparisons[compi]])>criticalTval,1,0) 
    maskPerms= np.where(np.abs(permsTvalsubROIs[comparisons[compi]])>criticalTval,1,0) 

    # Variable to store number of subROI with significant observed t-value for each ROI
    ROIsigCountObs = np.zeros(len(st.checkingROIid))

    # Variable to store number of subROI with significant permutation t-value for each ROI
    ROIsigCountPerms = np.zeros((len(st.checkingROIid), st.nPerms))
    pVal = np.zeros(len(st.checkingROIid))

    for ROIi in range(len(st.checkingROIid)):
        
        # extract list of indexes of subROIs from ROI    
        subROIsId = [int(subROIi) for subROIi in range(len(splitlabelsname)) if ROIsubId[subROIi] == ROIi] 
        subROIsId = np.asarray(subROIsId)
        
        # Extract number of subROI with significant observed t-value for each ROI
        ROIsigCountObs[ROIi] = np.sum(maskSubROIs[:,subROIsId])

        # Extract number of subROI with significant permutation t-value for each ROI
        ROIsigCountPerms[ROIi,:] = np.sum( maskPerms[subROIsId,:], 0)
        pVal[ROIi] = np.sum(ROIsigCountPerms[ROIi,:] > ROIsigCountObs[ROIi])/st.nPerms

    #% FDR correction for multiple tests over all ROIs
    FDRh, FDRpVal = fdrcorrection(pVal, alpha=alpha)

    # Extract ROI which show significant different after FDR correction
    fdrROI = np.where(FDRh==True)[0] # np.where return a tuple
    fdrROInames = [st.orglabels[idx].name for idx in fdrROI]

    #%  Modify mask matrices due to applying FDR
    fdrMaskROIs = [maskVal in fdrROIfor maskVal in ROIsubId]
    fdrMaskSubROIs = np.multiply(fdrMaskROIs, maskSubROIs)

    return fdrMaskSubROIs

def plot_brain_with_mask(taskid, l, plot_brain=0):
    if taskid == 0:
        latencies = [[0.106, 0.133],[0.196, 0.231],[0.352, 0.407],[0.458, 0.485]] # listening
    elif taskid == 1:
        latencies = [[0.063, 0.134],[0.173, 0.243],[0.274, 0.325],[0.446, 0.477],[0.580, 0.657]] # imagined
    else:
        latencies =[[-0.2, 0], [0.134, 0.177], [0.189, 0.216], [0.240, 0.302], [0.384, 0.497], [0.560, 0.689] ]# overt task

    criticalTval = 2.021 # 2 tail t-test for 40 sample with critical value is set to 0.01
    alpha = 0.05 # for FDR correction
    latency = latencies[l]

    # Load observed t- values and permutation t-values data  
    path_pescar = st.pathDatSem +  st.tasks[taskid]  + "\\peSCAR\\"
    obsTvalsubROIs = loadmat(path_pescar + "obsTvalsubROIs_" + str(int(latency[0]*1000)) + "_" + str(int(latency[1]*1000)) + ".mat") # t-statistic from osberved data

    fdrMaskSubROIallPairs = dict()
    for compi in range(len(comparisons)):   

        fdrMaskSubROIallPairs[comparisons[compi]] = create_fdr_mask(criticalTval, alpha, taskid, l)

        # %  Visualizing permutation t-test result with mask on pragmatic atlas
        '''
        time_label = latency[0]*1000 # in ms
        path_save = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\' + st.tasks[taskid] + '\\figure\\brain'

        # Load pragmatic labels for semantic presentation
        subjects_dir = 'U:\\usr\\local\\freesurfer\\7.2.0\\subjects\\'
        path_label = subjects_dir + '\\fsaverage\\label\\pragmatic_atlas\\'

        lpragmatic_semantic = np.load(path_label + 'lh.pragmatic-semantic.npz')
        rpragmatic_semantic = np.load(path_label + 'rh.pragmatic-semantic.npz')
        pragmatic_info = np.load(path_label + 'pragmatic-info.npz')

        # Correct the order for significant semantic areas as there is only 77 areas for the left and 63 areas for the right
        # so other non-significant areas has the same name for the label of [l/r]'h..label'

        # get corresponding index of non-singificant areas which has empty name in left and right hemisphere
        llabels_name = [label_name.decode('UTF-8') for label_name in pragmatic_info['lnames'] ]
        rlabels_name = [label_name.decode('UTF-8') for label_name in pragmatic_info['rnames'] ]

        llabels  = [mne.read_label(path_label + 'lh.' + label_name + '.label', 'fsaverage') for label_name in  llabels_name]
        rlabels  = [mne.read_label(path_label + 'rh.' + label_name + '.label', 'fsaverage') for label_name in  rlabels_name]

        # Plot brain
        hemi = 'split'
        views = 'lateral' #'medial' 
        # create saving folder if not existing
        if os.path.exists(path_save + '\\' +views)==False:
            os.mkdir(path_save + '\\' +views)

        # Face-Number
        figtitle = comparisons[compi] + str(time_label) + " ms, FDR correction, p<"  + str(alpha)      
        stc_viz = ut.mask_stats_results_on_brain( np.reshape(obsTvalsubROIs[comparisons[compi]], -1) , 
                                st.splitlabels, np.reshape(fdrMaskSubROIs,-1), st.tasks[taskid] + figtitle, time_label, path_save,1)     

        # parameter to for brain plotting
        surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
            hemi=hemi, views=views, size=(1200,400), title =  st.tasks[taskid] + figtitle, initial_time=time_label, 
            time_unit='ms', background='white', clim={ 'kind':'value', 'pos_lims':[criticalTval, criticalTval, max(np.max(abs(stc_viz.data)), criticalTval)] }  )


        brain = mne.viz.plot_source_estimates(stc_viz,**surfer_kwargs)

        [brain.add_label(llabels[i], borders=True, color = lpragmatic_semantic['rgb'][i]) for i in range(len(llabels_name)) if llabels_name[i] != '']
        [brain.add_label(rlabels[i], borders=True, color = rpragmatic_semantic['rgb'][i]) for i in range(len(rlabels_name)) if rlabels_name[i] != '']

        brain.add_text(0.1, 0.9, figtitle, 'title', font_size=11)
        fname = path_save + '\\' +views + '\\' +  figtitle[2:len(figtitle)-24] + '.png'
        brain.save_image(fname)
        brain.close()
        del stc_viz

        '''
    #%  Visualizing permutation t-test result with mask on, no plotting atlas

        if plot_brain == 1:
            time_label = latency[0]*1000 # in ms
            # path_save = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\' + st.tasks[taskid] + '\\figure\\brain\\without_pragmatic_atlas\\v2\\'
            path_save = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\' + st.tasks[taskid] + '\\figure\\brain\\p_0.01\\without_pragmatic_atlas\\'

            # Plot brain
            hemi = 'split'
            views = 'lateral'  #'lateral' 
            # create saving folder if not existing
            if os.path.exists(path_save + views)==False:
                os.mkdir(path_save + views)

            # Face-Number
            figtitle = comparisons[compi] + str(time_label) + " ms, FDR correction, p<"  + str(alpha)      
            stc_viz = ut.mask_stats_results_on_brain( np.reshape(obsTvalsubROIs[comparisons[compi]], -1) , 
                                    st.splitlabels, np.reshape(fdrMaskSubROIallPairs[comparisons[compi]],-1), st.tasks[taskid] + figtitle, time_label, path_save,1)     

            # parameter to for brain plotting
            surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
                hemi=hemi, views=views, size=(1200,400), title =  st.tasks[taskid] + figtitle, initial_time=time_label, 
                time_unit='ms', background='white', clim={ 'kind':'value', 'pos_lims':[criticalTval, criticalTval, max(np.max(abs(stc_viz.data)), criticalTval)] }  )


            brain = mne.viz.plot_source_estimates(stc_viz,**surfer_kwargs)
            brain.add_text(0.1, 0.9, figtitle, 'title', font_size=11)
            fname = path_save + '\\' +views + '\\' +  figtitle[2:len(figtitle)-24] + '.png'
            brain.save_image(fname)
            brain.close()

            del stc_viz


def plot_brain_all_time(taskid, plot_brain=1): 
    """
    Plot brain for differences in 4 time windows into 1, only shows t values of regions that shows consistent activation for either semantic category.
    Regions that changes their activation for either semantic category over time are shown in color of parcellation with opacity
    Regions showing no significant difference are plotted in default color.

        - Regions that shows consistently strong activation for first semantic category over 4 time windows are shown in red
        - Regions that shows consistently strong activation for first semantic category over 4 time windows are shown in blue

    Inputs:
        taskid: int, task index
        plot_brain: int, option whether to plot brain of difference 

    Outputs:
        overlappedIndex: dict, dictionary storing indices of regions that change sign of fdrMask over time
        redIndex: dict, dictionary storing indices of regions that show consitent significant for the 1st category in comparison 
        blueIndex: dict, dictionary storing indices of regions that show consitent significant for the 2nd category in comparison 

    """

    if taskid == 0:
        latencies = [[0.106, 0.133],[0.196, 0.231],[0.352, 0.407],[0.458, 0.485]] # listening
    elif taskid == 1:
        latencies = [[0.063, 0.134],[0.173, 0.243],[0.274, 0.325],[0.446, 0.477],[0.580, 0.657]] # imagined
    else:
        latencies =[[-0.2, 0], [0.134, 0.177], [0.189, 0.216], [0.240, 0.302], [0.384, 0.497], [0.560, 0.689] ]# overt task

    criticalTval = 2.021 # 2 tail t-test for 40 sample with critical value is set to 0.01
    alpha = 0.05 # for FDR correction
    path_pescar = st.pathDatSem + st.tasks[taskid]  + "\\peSCAR\\"

    obsTvalWithMask = dict()
    overlappedIndex = dict()
    redIndex = dict() # dictionary storing all subRoi indice, which is repeatable, for latter category in comparison
    blueIndex = dict()

    for compi in range(len(comparisons)):
        obsTvalWithMask[comparisons[compi]] = np.zeros((len(latencies),(len(st.splitlabels))))
        redIndex_temp = []
        blueIndex_temp  = []

        for l in range(len(latencies)):
            latency = latencies[l]

            # Load observed t- values and permutation t-values data  
            obsTvalsubROIs = loadmat(path_pescar + "obsTvalsubROIs_" + str(int(latency[0]*1000)) + "_" + str(int(latency[1]*1000)) + ".mat") # t-statistic from osberved data
            fdrMask = loadmat(path_pescar +"fdrMaskSubROIallPairs_" + str(l+1) + ".mat" )
            obsTvalWithMask[comparisons[compi]][l] = np.multiply(fdrMask[comparisons[compi]],obsTvalsubROIs[comparisons[compi]])

            # Find regions that shows significant difference for different semantic categories over time 
            # defined as overlapped regions during different semantic categories processing

            mask = np.sign( obsTvalWithMask[comparisons[compi]][l])
            redIndex_temp.extend([i for i in  range(len(mask)) if mask[i] == 1])
            blueIndex_temp.extend([i for i in  range(len(mask)) if mask[i] == -1])

        overlappedIndex[comparisons[compi]] = list(set(redIndex_temp).intersection(blueIndex_temp))
        redIndex[comparisons[compi]] = list(set(redIndex_temp) - set(overlappedIndex[comparisons[compi]]))
        blueIndex[comparisons[compi]] = list(set(blueIndex_temp) - set(overlappedIndex[comparisons[compi]]))

        # Visualize on the brain. NOT FINISHED YET.
        if plot_brain == 1:
            path_save = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\' + st.tasks[taskid] + '\\figure\\brain\\p_0.05\\without_pragmatic_atlas\\'

            # Plot brain
            hemi = 'split'
            views = 'lateral'  #'lateral' 
            # create saving folder if not existing
            if os.path.exists(path_save + views)==False:
                os.mkdir(path_save + views)

            # Face-Number
            figtitle = comparisons[compi] + " all time, FDR correction, p<"  + str(alpha)      
            stc_viz = ut.mask_stats_results_on_brain( np.reshape(obsTvalWithMask[comparisons[compi]], -1) , 
                                    st.splitlabels, np.reshape(fdrMaskSubROIallPairs[comparisons[compi]],-1), st.tasks[taskid] + figtitle, time_label, path_save,1)     

            # parameter to for brain plotting
            surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
                hemi=hemi, views=views, size=(1200,400), title =  st.tasks[taskid] + figtitle, initial_time=time_label, 
                time_unit='ms', background='white', clim={ 'kind':'value', 'pos_lims':[criticalTval, criticalTval, max(np.max(abs(stc_viz.data)), criticalTval)] }  )


            brain = mne.viz.plot_source_estimates(stc_viz,**surfer_kwargs)
            brain.add_text(0.1, 0.9, figtitle, 'title', font_size=11)
            fname = path_save + '\\' +views + '\\' +  figtitle[2:len(figtitle)-24] + '.png'
            brain.save_image(fname)
            brain.close()


        return overlappedIndex, blueIndex, redIndex





