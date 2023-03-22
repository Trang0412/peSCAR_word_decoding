
#%%
import checking_result_permutation_test
def main():

    ROInames = []
    ROImasks = []
    for taskid in range(2):

        for l in range(4):
            roiNames, roiMasks = checking_result_permutation_test.plot_brain_with_mask(taskid, l+1,1)
            ROInames.append(roiNames)
            ROImasks.append(roiMasks)

    return ROInames, ROImasks

# main()

#%% get names of ROI showing difference in semantic processing for each task
import pandas as pd
from collections import defaultdict
import checking_result_permutation_test
ROInames = list()
path_save = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\listening\\peSCAR\\'
fname_excel = 'signifcantROIs.xlsx'

tasks = ['listening', 'imagined', 'overt']
with pd.ExcelWriter(path_save + fname_excel) as writer:
    for taskid in range(2):
        ROInames.append(defaultdict(list))
        for l in range(4):
            
            temp, _ = checking_result_permutation_test.plot_brain_with_mask(taskid, l+1,0)
            for k in temp.keys():
                ROInames[taskid][k].extend(temp[k])
        
        # keep only unique value for each ROI
        for k in ROInames[taskid].keys():
            ROInames[taskid][k] = list(set(ROInames[taskid][k]))
        saved_df = pd.DataFrame.from_dict(ROInames[taskid], orient = 'index')
        saved_df = saved_df.transpose()
        # with pd.ExcelWriter(path_save + fname_excel) as writer:
        saved_df.to_excel(writer, sheet_name=tasks[taskid])
        del saved_df


        





# # %% plot ROIs with significant difference for semantic on brain
# import sys
# import mne
# import numpy as np


# sys.path.insert(1, 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\code\\python' )
# import settings as st
# st.init()

# orglabelsname = [label.name for label in st.orglabels]
# splitlabelsname = [label.name for label in st.splitlabels]

# roi_listening = []
# roi_imagined = []
# for i in range(3):
#     roi_listening = np.hstack((roi_listening, ROInames[i]['FaceNum'], ROInames[i]['AnimalNum'], ROInames[i]['AnimalFace']))
#     roi_imagined = np.hstack((roi_imagined, ROInames[i+4]['FaceNum'], ROInames[i+4]['AnimalNum'], ROInames[i+4]['AnimalFace']))
    

# Brain = mne.viz.get_brain_class()

# subjects_dir = mne.datasets.sample.data_path() / 'subjects'
# mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
#                                         verbose=True)

# mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
#                                           verbose=True)

# labels = mne.read_labels_from_annot(
#     'fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)

# brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=subjects_dir,
#               cortex='low_contrast', background='white', size=(800, 600))
# brain.add_annotation('HCPMMP1')
# aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
# brain.add_label(aud_label, borders=False)
    
# # %%

# %%
