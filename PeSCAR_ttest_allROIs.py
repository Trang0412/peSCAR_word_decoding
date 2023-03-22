# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:42:32 2021

Extract source activation data for all subROIs
    - Use customized splitted parcellation (PeSCAR)
    - Ref: 2019, Scientific Reports, Mamashli
    
            
Ref: 2007, Maris, Journal of Neuroscience Methos, "Nonparametric statistical testing of EEG- and MEG-data"

Permutation testing over all sub-ROI in a ROI for  source acitvation 
between 2 group of sematic class (i.e., face vs number and animal vs number)


Permutation procedure (within-subject study):

    1. Collect mean ampltiude data of all sub-ROIs of all subjects in a single set

    2. Randomly draw from this combined data set  as there were data in condition 1 and place those data into subset 1.
       Place the remaining data in subset 2

    3. Calculate the test statistic on this random partition

    4. Repeat steps 2 and 3 10000 times and construct a histogram of the test statistic

    5. From the test statistic that was actually observed and the histogram in step 4, calculate p-value for observed one
    
    6. If the p-value is smaller than the critical alpha-level, then conclude that the data in the two experiment conditions are significant different

    
@author: TrangLT
"""


import sys
from scipy.io import savemat, loadmat

import numpy as np
import matplotlib.pyplot as plt
import mne
from tqdm import tqdm
import time

from scipy.stats import ttest_rel

sys.path.insert(1, 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\code\\python\\semantic_processing')
from utilities import compute_globalSNR, extract_label_activation_word_semantic,find_ROIsubId

sys.path.insert(1, 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\code\\python' )
import settings as st
st.init()

def __init__():
    return


task_id = 2 #   0: listening,   1: imagined,     2: overt

# subjects_dir = "U:\\usr\\local\\freesurfer\\7.2.0\\subjects"
path_task = st.pathDatSem + st.tasks[task_id] 
path_pescar = path_task + "\\peSCAR"
subjects_dir =st.subjects_dir
subject = 'fsaverage'

fNameAnnot = "aparc_sub"
splitlabels =  mne.read_labels_from_annot(subject, fNameAnnot, subjects_dir=subjects_dir)       
fNameAnnot = "aparc"
orglabels =  mne.read_labels_from_annot(subject, fNameAnnot, subjects_dir=subjects_dir)
    
    
orglabelsname = [label.name for label in st.orglabels]
splitlabelsname = [label.name for label in st.splitlabels]

if task_id == 0:
    latencies = [[0.106, 0.133],[0.196, 0.231],[0.352, 0.407],[0.458, 0.485]] # listening
elif task_id == 1:
    latencies = [[0.063, 0.134],[0.173, 0.243],[0.274, 0.325],[0.446, 0.477],[0.580, 0.657]] # imagined task
else:
    latencies =[[0.134, 0.177], [0.189, 0.216], [0.240, 0.302], [0.384, 0.497], [0.560, 0.689] ]# overt task
    # latencies =[[-0.2, 0]]# overt task

nlatencies = len(latencies)
    
# finding ROI's id which a subROi belongs to
ROIsubId = find_ROIsubId(orglabelsname,splitlabelsname)
     
#%% Extract data for each ROI in split parcellation
   
def extract_label_activation_all_word_semantic(task_id):
    """
    Extract activation for all regions in split parcellaation for each of semantic group

    Parameters
    ----------
    task_id : integer
                integer to indicate which task are processed

    Returns
    -------
    None.

    """
    # initiate empty list to store mean amplitude of subROI and ROIsfrom all subjects      
    subROIdataFaceWords = []  
    subROIdataNumberWords = [] 
    subROIdataAnimalWords = []
    

    pathStc = st.pathDatSem + st.tasks[task_id] + "\\sources\\stc"
    pathinv= st.pathDatSem + st.tasks[task_id] + "\\sources\\inv"
    
    for sub_id in range(len(st.checkingSub)):
        
        # Extract mean of activation over each sub-ROI
        [tempFace, tempNumber, tempAnimal] = extract_label_activation_word_semantic(pathStc,pathinv, sub_id, splitlabels)
        
        subROIdataFaceWords.append(tempFace)        
        subROIdataNumberWords.append(tempNumber)        
        subROIdataAnimalWords.append(tempAnimal)
        
        del tempFace
        del tempNumber
        del tempAnimal
          
    
    # Convert to array, [nsub, labels, timepnts]
    subROIdataFaceWords = np.asarray(subROIdataFaceWords)
    subROIdataNumberWords = np.asarray(subROIdataNumberWords)
    subROIdataAnimalWords = np.asarray(subROIdataAnimalWords)
    
    
    savemat(path_pescar +"\\subROIdataFaceWords.mat", {'subROIdataFaceWords': subROIdataFaceWords})
    savemat(path_pescar+"\\subROIdataNumberWords.mat", {'subROIdataNumberWords': subROIdataNumberWords})
    savemat(path_pescar +"\\subROIdataAnimalWords.mat", {'subROIdataAnimalWords': subROIdataAnimalWords})
        
        
extract_label_activation_all_word_semantic(task_id)


#%% Load data for running PeSCAR 
# Load all  subROIs activation data
subROIdataFaceWords = loadmat(path_pescar + "\\subROIdataFaceWords.mat")
subROIdataNumberWords = loadmat(path_pescar +"\\subROIdataNumberWords.mat")
subROIdataAnimalWords = loadmat(path_pescar + "\\subROIdataAnimalWords.mat")

subROIdataFaceWords = subROIdataFaceWords['subROIdataFaceWords']
subROIdataAnimalWords = subROIdataAnimalWords['subROIdataAnimalWords']
subROIdataNumberWords = subROIdataNumberWords['subROIdataNumberWords']

for li in range(nlatencies):

                
    timewin = latencies[li]
    timewinidx = [np.where(st.times >= timewin[tidx])[0][0] for tidx in range(len(timewin))]
                        
    # Getting mean value of all subROI's activation over pre-defined time window
    meanSubROIface      = np.squeeze(np.ndarray.mean(subROIdataFaceWords[:,:,timewinidx[0]:timewinidx[1]],2))
    meanSubROInumber    = np.squeeze(np.ndarray.mean(subROIdataNumberWords[:,:,timewinidx[0]:timewinidx[1]],2))
    meanSubROIanimal    = np.squeeze(np.ndarray.mean(subROIdataAnimalWords[:,:,timewinidx[0]:timewinidx[1]],2))
    
    
    # Create dummy variables to store values of permutation test   
            
    # Initilize zeros arrays to store observed t-value  for all sub-ROIs
    obsTvalsubROIsFaceNum = np.zeros(len(splitlabelsname)) 
    obsTvalsubROIsAnimalNum = np.zeros(len(splitlabelsname)) 
    obsTvalsubROIsAnimalFace = np.zeros(len(splitlabelsname)) 
    
    
    # Initilize zeros arrays to store permutation t-value sum for all ROIs
    permsTvalsubROIsFaceNum =  np.zeros( (len(splitlabelsname), st.nPerms) ) 
    permsTvalsubROIsAnimalNum = np.zeros( (len(splitlabelsname), st.nPerms) ) 
    permsTvalsubROIsAnimalFace = np.zeros( (len(splitlabelsname), st.nPerms) ) 
    

    #%% Run PeSCAR ttest for all ROIs
     
    print('\n \n------------------------------ Running permutation testing -------------------------------')
    
    for i in tqdm(range(10)): # display progress bar
        time.sleep(3)

    for ROIi in range(len(st.checkingROIid)):
        
        # extract list of indexes of subROIs from ROI    
        subROIsId = [int(subROIi) for subROIi in range(len(splitlabelsname)) if ROIsubId[subROIi] == ROIi] 
        subROIsId = np.asarray(subROIsId)
           
    
        # Compute osbverved t-value for all subROIs in the current ROI
        for subROIi in subROIsId:
            
            obsTvalsubROIsFaceNum[subROIi] = ttest_rel( meanSubROIface[:,subROIi], meanSubROInumber[:,subROIi] ).statistic 
            obsTvalsubROIsAnimalNum[subROIi] =  ttest_rel( meanSubROIanimal[:,subROIi], meanSubROInumber[:,subROIi]).statistic 
            obsTvalsubROIsAnimalFace[subROIi] = ttest_rel(meanSubROIanimal[:,subROIi], meanSubROIface[:,subROIi]).statistic 
                        
            
       # Permutation between each pair of sub-ROIs in a ROI   
        for pi in range(st.nPerms):
     
            # Face words vs. Number words
            # stack data for t-test into one dimension array, first half is Face words data, second half is Number words data
            permData = np.reshape(np.hstack(( meanSubROIface[:,subROIsId], meanSubROInumber[:,subROIsId]  )),-1)
            permData = np.random.permutation(permData)              
            permData = np.reshape(permData,(st.nsub,-1))
                
            for (i, subROIi) in enumerate(subROIsId):     
                permsTvalsubROIsFaceNum[subROIi, pi] = ttest_rel(permData[:,i], permData[:, i+len(subROIsId)]).statistic

            del permData
            
            # Animal words vs. Number words
            permData = np.reshape(np.hstack(( meanSubROIanimal[:,subROIsId], meanSubROInumber[:,subROIsId] )),-1)
            permData = np.random.permutation(permData)              
            permData = np.reshape(permData,(st.nsub,-1))
                
            for (i, subROIi) in enumerate(subROIsId):     
                permsTvalsubROIsAnimalNum[subROIi, pi] = ttest_rel(permData[:,i], permData[:, i+len(subROIsId)]).statistic
                
            del permData     
            
            # Animal words vs. Face words
            permData = np.reshape(np.hstack((meanSubROIanimal[:,subROIsId], meanSubROIface[:,subROIsId])),-1)
            permData = np.random.permutation(permData)        
            permData = np.reshape(permData,(st.nsub,-1))
                
            for (i, subROIi) in enumerate(subROIsId):     
                permsTvalsubROIsAnimalFace[subROIi, pi] = ttest_rel(permData[:,i], permData[:, i+len(subROIsId)]).statistic
                
            del permData
            
            
            
    # Save data  

    obsTvalsubROIs = {"FaceNum":np.asarray(obsTvalsubROIsFaceNum), "AnimalNum":np.asarray(obsTvalsubROIsAnimalNum),
            "AnimalFace":np.asarray(obsTvalsubROIsAnimalFace)}
                   
    permsTvalsubROIs = {"FaceNum":np.asarray(permsTvalsubROIsFaceNum), "AnimalNum":np.asarray(permsTvalsubROIsAnimalNum),
                    "AnimalFace":np.asarray(permsTvalsubROIsAnimalFace)}
    
    
    savemat(path_pescar +"\\obsTvalsubROIs_" + str(int(timewin[0]*1000)) + "_" + str(int(timewin[1]*1000)) + ".mat", obsTvalsubROIs)
    savemat(path_pescar +"\\permsTvalsubROIs_"+ str(int(timewin[0]*1000)) + "_" + str(int(timewin[1]*1000)) + ".mat", permsTvalsubROIs)
 
