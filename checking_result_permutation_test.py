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

     #%% Variabels need to change corresponding to task 
"""
    taskid = int(input( ' 1: listening,   2: imagined,     3: overt \t')) -1
    
    # Chang latencies corresponding to checking task
    if taskid == 0:
        latencies = [[0.106, 0.133],[0.196, 0.231],[0.352, 0.407],[0.458, 0.485]] # listening
        l = int(input('Latency to check: \n1: [0.106, 0.133],   2:[0.196, 0.231],    3:[0.352, 0.407],    4:[0.458, 0.485]? \t'))
    elif taskid == 1:
        latencies = [[0.063, 0.134],[0.173, 0.243],[0.274, 0.325],[0.446, 0.477],[0.580, 0.657]] # imagined
        l = int(input('Latency to check: \n1: [0.063, 0.134],   2:[0.173, 0.243],   3:[0.274, 0.325],   4:[0.446, 0.477],   5:[0.580, 0.657]? \t'))
    else:
        # latencies =[[-0.2, 0], [0.134, 0.177], [0.189, 0.216], [0.240, 0.302], [0.384, 0.497], [0.560, 0.689] ]# overt task
        # l = int(input('Latency to check: \n1:[0.134, 0.177],   2:[0.189, 0.216],   3:[0.240, 0.302],   4:[0.384, 0.497],   5:[0.560, 0.689] ? \t'))

        latencies =[[-0.2, 0], [0.134, 0.177], [0.189, 0.216], [0.240, 0.302], [0.384, 0.497], [0.560, 0.689] ]# overt task
        # latencies =[[-0.2, 0]]
        l = int(input('Latency to check: \n 1:[-0.2, 0]  2:[0.134, 0.177],   3:[0.189, 0.216],  4:[0.240, 0.302],   5:[0.384, 0.497],   6:[0.560, 0.689] ? \t'))
 """
    #%% Run PeSCAR for each time window
def plot_brain_with_mask(taskid, l, plot_brain=0):
    if taskid == 0:
        latencies = [[0.106, 0.133],[0.196, 0.231],[0.352, 0.407],[0.458, 0.485]] # listening
    elif taskid == 1:
        latencies = [[0.063, 0.134],[0.173, 0.243],[0.274, 0.325],[0.446, 0.477],[0.580, 0.657]] # imagined
    else:
        latencies =[[-0.2, 0], [0.134, 0.177], [0.189, 0.216], [0.240, 0.302], [0.384, 0.497], [0.560, 0.689] ]# overt task

    criticalTval = 2.021 # 2 tail t-test for 40 sample with critical value is set to 0.01
    alpha = 0.05 # for FDR correction
    path_pescar = st.pathDatSem + "\\" + st.tasks[taskid]  + "\\peSCAR"
    latency = latencies[l-1]
    
    # Load observed t- values and permutation t-values data  
    obsTvalsubROIs = loadmat(path_pescar + "\\obsTvalsubROIs_" + str(int(latency[0]*1000)) + "_" + str(int(latency[1]*1000)) + ".mat") # t-statistic from osberved data
    permsTvalsubROIs = loadmat(path_pescar + "\\permsTvalsubROIs_" + str(int(latency[0]*1000)) + "_" + str(int(latency[1]*1000)) + ".mat") # p-values from permutation t-test


    pValFaceNum = np.zeros(len(st.checkingROIid))
    pValAnimalNum = np.zeros(len(st.checkingROIid))
    pValAnimalFace = np.zeros(len(st.checkingROIid))
             
    #% Find p-value for each ROI 


    # create a mask matrix which subROI show significant p-value will be masked with value of 1 otherwise 0
    maskSubROIsFaceNum = np.where(np.abs(obsTvalsubROIs['FaceNum'])>criticalTval,1,0) 
    maskSubROIsAnimalNum = np.where(np.abs(obsTvalsubROIs['AnimalNum'])>criticalTval,1,0) 
    maskSubROIsAnimalFace = np.where(np.abs(obsTvalsubROIs['AnimalFace'])>criticalTval,1,0) 


    maskPermsFaceNum = np.where(np.abs(permsTvalsubROIs['FaceNum'])>criticalTval,1,0) 
    maskPermsAnimalNum = np.where(np.abs(permsTvalsubROIs['AnimalNum'])>criticalTval,1,0) 
    maskPermsAnimalFace = np.where(np.abs(permsTvalsubROIs['AnimalFace'])>criticalTval,1,0) 


    # Variable to store number of subROI with significant observed t-value for each ROI
    ROIsigCountObsFaceNum = np.zeros(len(st.checkingROIid))
    ROIsigCountObsAnimalNum = np.zeros(len(st.checkingROIid))
    ROIsigCountObsAnimalFace = np.zeros(len(st.checkingROIid))

    # Variable to store number of subROI with significant permutation t-value for each ROI
    ROIsigCountPermsFaceNum = np.zeros((len(st.checkingROIid), st.nPerms))
    ROIsigCountPermsAnimalNum = np.zeros((len(st.checkingROIid), st.nPerms))
    ROIsigCountPermsAnimalFace = np.zeros((len(st.checkingROIid), st.nPerms))

    pValFaceNum = np.zeros(len(st.checkingROIid))
    pValAnimalNum = np.zeros(len(st.checkingROIid))
    pValAnimalFace = np.zeros(len(st.checkingROIid))


    for ROIi in range(len(st.checkingROIid)):
        
        # extract list of indexes of subROIs from ROI    
        subROIsId = [int(subROIi) for subROIi in range(len(splitlabelsname)) if ROIsubId[subROIi] == ROIi] 
        subROIsId = np.asarray(subROIsId)
        
        # Extract number of subROI with significant observed t-value for each ROI
        ROIsigCountObsFaceNum[ROIi] = np.sum(maskSubROIsFaceNum[:,subROIsId])
        ROIsigCountObsAnimalNum[ROIi] = np.sum(maskSubROIsAnimalNum[:,subROIsId])
        ROIsigCountObsAnimalFace[ROIi] = np.sum(maskSubROIsAnimalFace[:,subROIsId])

        
        # Extract number of subROI with significant permutation t-value for each ROI
        ROIsigCountPermsFaceNum[ROIi,:] = np.sum( maskPermsFaceNum[subROIsId,:], 0)
        ROIsigCountPermsAnimalNum[ROIi,:] = np.sum( maskPermsAnimalNum[subROIsId,:], 0)
        ROIsigCountPermsAnimalFace[ROIi,:] = np.sum( maskPermsAnimalFace[subROIsId,:], 0)
        
        pValFaceNum[ROIi] = np.sum(ROIsigCountPermsFaceNum[ROIi,:] > ROIsigCountObsFaceNum[ROIi])/st.nPerms
        pValAnimalNum[ROIi]  = np.sum(ROIsigCountPermsAnimalNum[ROIi,:] > ROIsigCountObsAnimalNum[ROIi])/st.nPerms
        pValAnimalFace[ROIi]  = np.sum(ROIsigCountPermsAnimalFace[ROIi,:] > ROIsigCountObsAnimalFace[ROIi])/st.nPerms
            
        
    compares = []
    compares = ['FaceNum']*len(st.orglabels)
    compares.extend(['AnimalNum']*len(st.orglabels))
    compares.extend(['AnimalFace']*len(st.orglabels))


    #% FDR correction for multiple tests over all ROIs
    FDRhFaceNum, FDRpValFaceNum = fdrcorrection(pValFaceNum, alpha=alpha)
    FDRhAnimalNum, FDRpValAnimalNum = fdrcorrection(pValAnimalNum,alpha=alpha )
    FDRhAnimalFace, FDRpValAnimalFace = fdrcorrection(pValAnimalFace,alpha=alpha)

    # Extract ROI which show significant different after FDR correction
    fdrROIfaceNum = np.where(FDRhFaceNum==True)[0] # np.where return a tuple
    fdrROIanimalNum = np.where(FDRhAnimalNum==True)[0]
    fdrROIanimalFace = np.where(FDRhAnimalFace==True)[0]

    fdrROInamesFaceNum = [st.orglabels[idx].name for idx in fdrROIfaceNum]
    fdrROInamesAnimalNum = [st.orglabels[idx].name for idx in fdrROIanimalNum]
    fdrROInamesAnimalFace = [st.orglabels[idx].name for idx in fdrROIanimalFace]


    fdrROInamesAllPairs = dict()
    fdrROInamesAllPairs['FaceNum'] = fdrROInamesFaceNum.copy()
    fdrROInamesAllPairs['AnimalNum'] = fdrROInamesAnimalNum.copy()
    fdrROInamesAllPairs['AnimalFace'] = fdrROInamesAnimalFace.copy()


    # Plot FDR-corrected p-value
    FDRpVal = np.hstack((FDRpValFaceNum, FDRpValAnimalNum,FDRpValAnimalFace))
    FDRpVal = pd.DataFrame(FDRpVal, columns=['FDR p-value'])
    FDRpVal['comparison'] = compares


    g = sns.FacetGrid(FDRpVal, col='comparison')
    g.map(sns.histplot, 'FDR p-value')

    #%  Modify mask matrices due to applying FDR
    fdrMaskROIs = [maskVal in fdrROIfaceNum for maskVal in ROIsubId]
    fdrMaskSubROIsFaceNum = np.multiply(fdrMaskROIs, maskSubROIsFaceNum)

    fdrMaskROIs = [maskVal in fdrROIanimalNum for maskVal in ROIsubId]
    fdrMaskSubROIsAnimalNum = np.multiply(fdrMaskROIs, maskSubROIsAnimalNum)

    fdrMaskROIs = [maskVal in fdrROIanimalFace for maskVal in ROIsubId]
    fdrMaskSubROIsAnimalFace = np.multiply(fdrMaskROIs, maskSubROIsAnimalFace)

    fdrMaskSubROIallPairs =  fdrMaskSubROIsFaceNum + fdrMaskSubROIsAnimalNum + fdrMaskSubROIsAnimalFace
    fdrMaskSubROIallPairs= np.where(fdrMaskSubROIallPairs>0,1,0)


    # save array of masked subROI for later use
    # path_save = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\' + st.tasks[taskid] + '\\'   
    # savemat(path_pescar + '\\fdrMaskSubROI_'  + str(l) + '.mat', {'fdrMaskSubROI': fdrMaskSubROIallGroup}) # str(l) indicates the time window when applied statistical test



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
    figtitle = ": Face-Number " + str(time_label) + " ms, FDR correction, p<"  + str(alpha)      
    stc_viz = ut.mask_stats_results_on_brain( np.reshape(obsTvalsubROIs['FaceNum'], -1) , 
                            st.splitlabels, np.reshape(fdrMaskSubROIsFaceNum,-1), st.tasks[taskid] + figtitle, time_label, path_save,1)     

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

    # Animal-Number
    figtitle = ": Animal-Number " + str(time_label) + " ms, p<" + str(alpha) + ', FDR-corrected'
    stc_viz = ut.mask_stats_results_on_brain(np.reshape(obsTvalsubROIs['AnimalNum'], -1) , 
                            st.splitlabels, np.reshape(fdrMaskSubROIsAnimalNum,-1), st.tasks[taskid] + figtitle, time_label, path_save,1)
    # parameter to for brain plotting
    surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
        hemi=hemi, views=views, size=(1200,400), title =  st.tasks[taskid] + figtitle, initial_time=time_label, 
        time_unit='ms', background='white', clim={ 'kind':'value', 'pos_lims':[criticalTval, criticalTval, max(np.max(abs(stc_viz.data)), criticalTval)] }  )


    brain = mne.viz.plot_source_estimates(stc_viz,**surfer_kwargs)

    [brain.add_label(llabels[i], borders=True, color = lpragmatic_semantic['rgb'][i]) for i in range(len(llabels_name)) if llabels_name[i] != '']
    [brain.add_label(rlabels[i], borders=True, color = rpragmatic_semantic['rgb'][i]) for i in range(len(rlabels_name)) if rlabels_name[i] != '']

    brain.add_text(0.1, 0.9, figtitle, 'title', font_size=11)
    fname = path_save + '\\' +views + '\\' + figtitle[2:len(figtitle)-24] + '.png'
    brain.save_image(fname)
    brain.close()
    del stc_viz

    # Animal-Face
    figtitle = ": Animal-Face " + str(time_label) + " ms, p<"   + str(alpha) + ', FDR-corrected'
    stc_viz = ut.mask_stats_results_on_brain( np.reshape(obsTvalsubROIs['AnimalFace'], -1) ,
                            st.splitlabels, np.reshape(fdrMaskSubROIsAnimalFace,-1), st.tasks[taskid] + figtitle, time_label, path_save,1)

    # parameter to for brain plotting
    surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
        hemi=hemi, views=views, size=(1200,400), title =  st.tasks[taskid] + figtitle, initial_time=time_label, 
        time_unit='ms', background='white', clim={ 'kind':'value', 'pos_lims':[criticalTval, criticalTval, max(np.max(abs(stc_viz.data)), criticalTval)] }  )

    brain = mne.viz.plot_source_estimates(stc_viz,**surfer_kwargs)

    [brain.add_label(llabels[i], borders=True, color = lpragmatic_semantic['rgb'][i]) for i in range(len(llabels_name)) if llabels_name[i] != '']
    [brain.add_label(rlabels[i], borders=True, color = rpragmatic_semantic['rgb'][i]) for i in range(len(rlabels_name)) if rlabels_name[i] != '']

    brain.add_text(0.1, 0.9, figtitle, 'title', font_size=11)
    fname = path_save + '\\' +views + '\\' + figtitle[2:len(figtitle)-24] + '.png'
    brain.save_image(fname)
    brain.close()

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
        figtitle = ": Face-Number " + str(time_label) + " ms, FDR correction, p<"  + str(alpha)      
        stc_viz = ut.mask_stats_results_on_brain( np.reshape(obsTvalsubROIs['FaceNum'], -1) , 
                                st.splitlabels, np.reshape(fdrMaskSubROIsFaceNum,-1), st.tasks[taskid] + figtitle, time_label, path_save,1)     

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

        # Animal-Number
        figtitle = ": Animal-Number " + str(time_label) + " ms, p<" + str(alpha) + ', FDR-corrected'
        stc_viz = ut.mask_stats_results_on_brain(np.reshape(obsTvalsubROIs['AnimalNum'], -1) , 
                                st.splitlabels, np.reshape(fdrMaskSubROIsAnimalNum,-1), st.tasks[taskid] + figtitle, time_label, path_save,1)
        # parameter to for brain plotting
        surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
            hemi=hemi, views=views, size=(1200,400), title =  st.tasks[taskid] + figtitle, initial_time=time_label, 
            time_unit='ms', background='white', clim={ 'kind':'value', 'pos_lims':[criticalTval, criticalTval, max(np.max(abs(stc_viz.data)), criticalTval)] }  )


        brain = mne.viz.plot_source_estimates(stc_viz,**surfer_kwargs)
        brain.add_text(0.1, 0.9, figtitle, 'title', font_size=11)
        fname = path_save + '\\' +views + '\\' + figtitle[2:len(figtitle)-24] + '.png'
        brain.save_image(fname)
        brain.close()
        del stc_viz

        # Animal-Face
        figtitle = ": Animal-Face " + str(time_label) + " ms, p<"   + str(alpha) + ', FDR-corrected'
        stc_viz = ut.mask_stats_results_on_brain( np.reshape(obsTvalsubROIs['AnimalFace'], -1) ,
                                st.splitlabels, np.reshape(fdrMaskSubROIsAnimalFace,-1), st.tasks[taskid] + figtitle, time_label, path_save,1)

        # parameter to for brain plotting
        surfer_kwargs = dict(surface='inflated', subject=st.subject, subjects_dir=st.subjects_dir,
            hemi=hemi, views=views, size=(1200,400), title =  st.tasks[taskid] + figtitle, initial_time=time_label, 
            time_unit='ms', background='white', clim={ 'kind':'value', 'pos_lims':[criticalTval, criticalTval, max(np.max(abs(stc_viz.data)), criticalTval)] }  )

        brain = mne.viz.plot_source_estimates(stc_viz,**surfer_kwargs)
        brain.add_text(0.1, 0.9, figtitle, 'title', font_size=11)
        fname = path_save + '\\' +views + '\\' + figtitle[2:len(figtitle)-24] + '.png'
        brain.save_image(fname)
        brain.close()


    return fdrROInamesAllPairs, fdrMaskSubROIallPairs
