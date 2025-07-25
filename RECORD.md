# Data

CT path:

* Training: Data/data_nifty/Training_603/Name_ID
* Internal Validation: Data/data_nifty/Internal_Test_259/Name_ID
* External Test: Data/data_nifty/External_Test_308/Name_ID
  example: "Data/data_nifty/1.Training_603/AI_JIU_GEN_P1548715/CT/A_8/A_8_02_shenAgioRoutine_20170728165801_6.nii.gz"

CT Type: A, D, N, V
example: in "Data/data_nifty/1.Training_603/AI_JIU_GEN_P1548715/CT/A_8/A_8_02_shenAgioRoutine_20170728165801_6.nii.gz". "A_8" can be "D_8", "N_8", "V_8".

VOI path:

* Training: "Data/ROI/1.Training_ROI_603/Name_ID/ROI/XXX.nrrd" (XXX include A_8, D_8, N_8, V_8)
* Internal: "Data/ROI/2.Internal Test_ROI_259/Name_ID/ROI/XXX.nrrd" (XXX include A_8, D_8, N_8, V_8)"
* External: "Data/ROI/3.External Test_ROI_308/Name_ID/ROI/XXX.nrrd" (XXX include A_8, D_8, N_8, V_8)"

Label (aggressive) Path:

* Training: Data/ccRCC_Survival_Analysis_Dataset_english/training_set_603_cases.csv
* Internal: Data/ccRCC_Survival_Analysis_Dataset_english/internal_test_set_259_cases.csv
* External: Data/ccRCC_Survival_Analysis_Dataset_english/external_verification_set_308_cases.csv

ID column: 'serial_number'
label column: 'aggressive_pathology_1_indolent_2_aggressive'

# Prompt

## V1

Help me think about how to build deep learning (from most commonly used/accepted by reviewer ones to Sota/Fancy ones) for the following task, then think about for high-impact jounal, what should I do like what figures I need to do (it is complimentary, we first have target figure, then do coding/modeling):

task: based on CT image and VOI segmentation to predict: patient kidney cancer aggressive or indolent, overall survival

data description:

1. CT image with CT type A, D, N, V, along with their VOI segmentation labels.
2. Label: for each patient, we have label aggressive or not, survival
3. Training set, internal test, external test already split (600, 250, 300 patients respectly)

# MISC

serial_number unique?
only pick VOI existed area