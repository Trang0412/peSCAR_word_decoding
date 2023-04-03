
#%%

import sys
import pandas as pd
import checking_result_permutation_test

sys.path.insert(1, 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\code\\python' )
import settings as st
st.init()


maplimits = [7.11, 3.53] # map limits for plotting brain corresponding to listening and imagined task
contrasts = ['FaceNum', 'AnimalNum', 'AnimalFace']
tasks = ['listening', 'imagined', 'overt']
def main():

    # path_save = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\listening\\peSCAR\\'
    # fname_excel = 'semanticROIs.xlsx'
    
    # # making summary for significant brain regions over 4 time windows for 3 contrasts
    # with pd.ExcelWriter(path_save + fname_excel) as writer:
    #     for taskid in range(2):
    #         semanticROIname = dict()
    #         overlappedIndex, blueIndex, redIndex, semanticROIindex = checking_result_permutation_test.plot_brain_all_time(taskid, 0, 'lateral')
            
    #         semanticROIname['Animal'] = [label.name for (i, label) in enumerate(st.splitlabels[:-2]) if i in semanticROIindex['Animal']]
    #         semanticROIname['Number'] = [label.name for (i, label) in enumerate(st.splitlabels[:-2]) if i in semanticROIindex['Number']]
    #         semanticROIname['Face'] = [label.name for (i, label) in enumerate(st.splitlabels[:-2]) if i in semanticROIindex['Face']]
            
    #         saved_df = pd.DataFrame.from_dict(semanticROIname, orient = 'index')
    #         saved_df = saved_df.transpose()
    #         saved_df.to_excel(writer, sheet_name=tasks[taskid])
    #         del saved_df

    # Plot bain
    for taskid in range(2):
        for l in range(4):
            checking_result_permutation_test.plot_brain_with_mask(taskid, l, plot_brain=1)

main()

# #%% get names of ROI showing difference in semantic processing for each task
# import pandas as pd
# from collections import defaultdict
# import checking_result_permutation_test
# ROInames = list()
# path_save = 'X:\\EEG_BCI\\2. Word decoding\\1. Temporal dynamic decoding\\analysis\\semantic_processing\\listening\\peSCAR\\'
# fname_excel = 'signifcantROIs.xlsx'

# tasks = ['listening', 'imagined', 'overt']
# with pd.ExcelWriter(path_save + fname_excel) as writer:
#     for taskid in range(2):
#         ROInames.append(defaultdict(list))
#         for l in range(4):
            
#             temp, _ = checking_result_permutation_test.plot_brain_with_mask(taskid, l+1,0)
#             for k in temp.keys():
#                 ROInames[taskid][k].extend(temp[k])
        
#         # keep only unique value for each ROI
#         for k in ROInames[taskid].keys():
#             ROInames[taskid][k] = list(set(ROInames[taskid][k]))
#         saved_df = pd.DataFrame.from_dict(ROInames[taskid], orient = 'index')
#         saved_df = saved_df.transpose()
#         # with pd.ExcelWriter(path_save + fname_excel) as writer:
#         saved_df.to_excel(writer, sheet_name=tasks[taskid])
#         del saved_df


        

# %%
