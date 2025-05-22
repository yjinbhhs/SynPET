# Databricks notebook source
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import openpyxl
import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def SUVR_real_ROI_calculation():

    root_dir = '/mnt/depts/dept01/mdsa/MLAI/PETSyn/ADNI'
    pet_dir = 'pet'
    sub_file = 'PET_18F-AV45_AD.csv'

    braak12_suffix = 'PET_to_oasis_SUVR_cropped_braak12.nii.gz'
    braak34_suffix = 'PET_to_oasis_SUVR_cropped_braak34.nii.gz'
    braak56_suffix = 'PET_to_oasis_SUVR_cropped_braak56.nii.gz'
    meta_temp_suffix = 'PET_to_oasis_SUVR_cropped_meta-temporal.nii.gz'
    frontal_suffix = 'PET_to_oasis_SUVR_cropped_frontal.nii.gz'
    parietal_suffix = 'PET_to_oasis_SUVR_cropped_parietal.nii.gz'
    lat_temp_suffix = 'PET_to_oasis_SUVR_cropped_lateral_temporal.nii.gz'
    med_temp_suffix = 'PET_to_oasis_SUVR_cropped_medial_temporal.nii.gz'
    ant_cingulate_suffix = 'PET_to_oasis_SUVR_cropped_ant_cingulate.nii.gz'
    pos_cingulate_suffix = 'PET_to_oasis_SUVR_cropped_pos_cingulate.nii.gz'
    composite_suffix = 'PET_to_oasis_SUVR_cropped_composite.nii.gz'
    
    pet_names = []
    mri_names = []
    group_ids = []
    SUVRs_braak12 = []
    SUVRs_braak34 = []
    SUVRs_braak56 = []
    SUVRs_meta_temp = []
    SUVRs_frontal = []
    SUVRs_parietal = []
    SUVRs_lat_temp = []
    SUVRs_med_temp = []
    SUVRs_ant_cingulate = []
    SUVRs_pos_cingulate = []
    SUVRs_composite = []

    sub_df = pd.read_csv(os.path.join(root_dir, sub_file))
    pet_list = sub_df['PET_ID']
    mri_list = sub_df['MRI_ID']
    group_list = sub_df['Group_ID']
    
    for pet_name, mri_name, group_id in zip(pet_list, mri_list, group_list):
        
        print(pet_name)
        pet_names.append(pet_name)
        mri_names.append(mri_name)
        group_ids.append(group_id)
        
        image_name = pet_name + '_' + braak12_suffix
        braak12_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        braak12_image_sitk = sitk.ReadImage(braak12_image_name)
        braak12_image_np = sitk.GetArrayFromImage(braak12_image_sitk).astype(np.float32)
        SUVR_braak12 = np.sum(braak12_image_np) / np.count_nonzero(braak12_image_np)
        SUVRs_braak12.append(SUVR_braak12)
        
        image_name = pet_name + '_' + braak34_suffix
        braak34_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        braak34_image_sitk = sitk.ReadImage(braak34_image_name)
        braak34_image_np = sitk.GetArrayFromImage(braak34_image_sitk).astype(np.float32)
        SUVR_braak34 = np.sum(braak34_image_np) / np.count_nonzero(braak34_image_np)
        SUVRs_braak34.append(SUVR_braak34)
        
        image_name = pet_name + '_' + braak56_suffix
        braak56_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        braak56_image_sitk = sitk.ReadImage(braak56_image_name)
        braak56_image_np = sitk.GetArrayFromImage(braak56_image_sitk).astype(np.float32)
        SUVR_braak56 = np.sum(braak56_image_np) / np.count_nonzero(braak56_image_np)
        SUVRs_braak56.append(SUVR_braak56)
        
        image_name = pet_name + '_' + meta_temp_suffix
        meta_temp_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        meta_temp_image_sitk = sitk.ReadImage(meta_temp_image_name)
        meta_temp_image_np = sitk.GetArrayFromImage(meta_temp_image_sitk).astype(np.float32)
        SUVR_meta_temp = np.sum(meta_temp_image_np) / np.count_nonzero(meta_temp_image_np)
        SUVRs_meta_temp.append(SUVR_meta_temp)
        
        image_name = pet_name + '_' + frontal_suffix
        frontal_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        frontal_image_sitk = sitk.ReadImage(frontal_image_name)
        frontal_image_np = sitk.GetArrayFromImage(frontal_image_sitk).astype(np.float32)
        SUVR_frontal = np.sum(frontal_image_np) / np.count_nonzero(frontal_image_np)
        SUVRs_frontal.append(SUVR_frontal)
        
        image_name = pet_name + '_' + parietal_suffix
        parietal_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        parietal_image_sitk = sitk.ReadImage(parietal_image_name)
        parietal_image_np = sitk.GetArrayFromImage(parietal_image_sitk).astype(np.float32)
        SUVR_parietal = np.sum(parietal_image_np) / np.count_nonzero(parietal_image_np)
        SUVRs_parietal.append(SUVR_parietal)
        
        image_name = pet_name + '_' + lat_temp_suffix
        lat_temp_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        lat_temp_image_sitk = sitk.ReadImage(lat_temp_image_name)
        lat_temp_image_np = sitk.GetArrayFromImage(lat_temp_image_sitk).astype(np.float32)
        SUVR_lat_temp = np.sum(lat_temp_image_np) / np.count_nonzero(lat_temp_image_np)
        SUVRs_lat_temp.append(SUVR_lat_temp)
        
        image_name = pet_name + '_' + med_temp_suffix
        med_temp_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        med_temp_image_sitk = sitk.ReadImage(med_temp_image_name)
        med_temp_image_np = sitk.GetArrayFromImage(med_temp_image_sitk).astype(np.float32)
        SUVR_med_temp = np.sum(med_temp_image_np) / np.count_nonzero(med_temp_image_np)
        SUVRs_med_temp.append(SUVR_med_temp)
        
        image_name = pet_name + '_' + ant_cingulate_suffix
        ant_cingulate_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        ant_cingulate_image_sitk = sitk.ReadImage(ant_cingulate_image_name)
        ant_cingulate_image_np = sitk.GetArrayFromImage(ant_cingulate_image_sitk).astype(np.float32)
        SUVR_ant_cingulate = np.sum(ant_cingulate_image_np) / np.count_nonzero(ant_cingulate_image_np)
        SUVRs_ant_cingulate.append(SUVR_ant_cingulate)
        
        image_name = pet_name + '_' + pos_cingulate_suffix
        pos_cingulate_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        pos_cingulate_image_sitk = sitk.ReadImage(pos_cingulate_image_name)
        pos_cingulate_image_np = sitk.GetArrayFromImage(pos_cingulate_image_sitk).astype(np.float32)
        SUVR_pos_cingulate = np.sum(pos_cingulate_image_np) / np.count_nonzero(pos_cingulate_image_np)
        SUVRs_pos_cingulate.append(SUVR_pos_cingulate)

        image_name = pet_name + '_' + composite_suffix
        composite_image_name = os.path.join(root_dir, pet_dir, pet_name, image_name)
        composite_image_sitk = sitk.ReadImage(composite_image_name)
        composite_image_np = sitk.GetArrayFromImage(composite_image_sitk).astype(np.float32)
        SUVR_composite = np.sum(composite_image_np) / np.count_nonzero(composite_image_np)
        SUVRs_composite.append(SUVR_composite)
  
    SUVR_list = {
        'PET_ID': pet_names,
        'MRI_ID': mri_names,
        'Group_ID': group_ids,
        'SUVR_braak12': np.round(SUVRs_braak12, 4),
        'SUVR_braak34': np.round(SUVRs_braak34, 4),
        'SUVR_braak56': np.round(SUVRs_braak56, 4),
        'SUVR_meta_temporal': np.round(SUVRs_meta_temp, 4),
        'SUVR_frontal': np.round(SUVRs_frontal, 4),
        'SUVR_parietal': np.round(SUVRs_parietal, 4),
        'SUVR_lat_temp': np.round(SUVRs_lat_temp, 4),
        'SUVR_med_temp': np.round(SUVRs_med_temp, 4),
        'SUVR_ant_cing': np.round(SUVRs_ant_cingulate, 4),
        'SUVR_pos_cing': np.round(SUVRs_pos_cingulate, 4),
        'SUVR_composite': np.round(SUVRs_composite, 4)
    }

    keys = SUVR_list.keys()
    csv_path = os.path.join(root_dir, 'SUVR_real.csv')
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(keys)
        writer.writerows(zip(*[SUVR_list[key] for key in keys]))

    ind_AD = [i for i, x in enumerate(group_ids) if x=='AD']
    ind_MCI = [i for i, x in enumerate(group_ids) if x =='MCI']
    ind_NC = [i for i, x in enumerate(group_ids) if x =='CN']

    SUVR_braak12_AD = [SUVRs_braak12[i] for i in ind_AD]
    SUVR_braak12_MCI = [SUVRs_braak12[i] for i in ind_MCI]
    SUVR_braak12_NC = [SUVRs_braak12[i] for i in ind_NC]
    SUVR_braak12_mean_AD = np.mean(SUVR_braak12_AD)
    SUVR_braak12_std_AD = np.std(SUVR_braak12_AD)
    SUVR_braak12_mean_MCI = np.mean(SUVR_braak12_MCI)
    SUVR_braak12_std_MCI = np.std(SUVR_braak12_MCI)
    SUVR_braak12_mean_NC = np.mean(SUVR_braak12_NC)
    SUVR_braak12_std_NC = np.std(SUVR_braak12_NC)
    
    SUVR_braak34_AD = [SUVRs_braak34[i] for i in ind_AD]
    SUVR_braak34_MCI = [SUVRs_braak34[i] for i in ind_MCI]
    SUVR_braak34_NC = [SUVRs_braak34[i] for i in ind_NC]
    SUVR_braak34_mean_AD = np.mean(SUVR_braak34_AD)
    SUVR_braak34_std_AD = np.std(SUVR_braak34_AD)
    SUVR_braak34_mean_MCI = np.mean(SUVR_braak34_MCI)
    SUVR_braak34_std_MCI = np.std(SUVR_braak34_MCI)
    SUVR_braak34_mean_NC = np.mean(SUVR_braak34_NC)
    SUVR_braak34_std_NC = np.std(SUVR_braak34_NC)
    
    SUVR_braak56_AD = [SUVRs_braak56[i] for i in ind_AD]
    SUVR_braak56_MCI = [SUVRs_braak56[i] for i in ind_MCI]
    SUVR_braak56_NC = [SUVRs_braak56[i] for i in ind_NC]
    SUVR_braak56_mean_AD = np.mean(SUVR_braak56_AD)
    SUVR_braak56_std_AD = np.std(SUVR_braak56_AD)
    SUVR_braak56_mean_MCI = np.mean(SUVR_braak56_MCI)
    SUVR_braak56_std_MCI = np.std(SUVR_braak56_MCI)
    SUVR_braak56_mean_NC = np.mean(SUVR_braak56_NC)
    SUVR_braak56_std_NC = np.std(SUVR_braak56_NC)
    
    SUVR_meta_temp_AD = [SUVRs_meta_temp[i] for i in ind_AD]
    SUVR_meta_temp_MCI = [SUVRs_meta_temp[i] for i in ind_MCI]
    SUVR_meta_temp_NC = [SUVRs_meta_temp[i] for i in ind_NC]
    SUVR_meta_temp_mean_AD = np.mean(SUVR_meta_temp_AD)
    SUVR_meta_temp_std_AD = np.std(SUVR_meta_temp_AD)
    SUVR_meta_temp_mean_MCI = np.mean(SUVR_meta_temp_MCI)
    SUVR_meta_temp_std_MCI = np.std(SUVR_meta_temp_MCI)
    SUVR_meta_temp_mean_NC = np.mean(SUVR_meta_temp_NC)
    SUVR_meta_temp_std_NC = np.std(SUVR_meta_temp_NC)
    
    SUVR_frontal_AD = [SUVRs_frontal[i] for i in ind_AD]
    SUVR_frontal_MCI = [SUVRs_frontal[i] for i in ind_MCI]
    SUVR_frontal_NC = [SUVRs_frontal[i] for i in ind_NC]
    SUVR_frontal_mean_AD = np.mean(SUVR_frontal_AD)
    SUVR_frontal_std_AD = np.std(SUVR_frontal_AD)
    SUVR_frontal_mean_MCI = np.mean(SUVR_frontal_MCI)
    SUVR_frontal_std_MCI = np.std(SUVR_frontal_MCI)
    SUVR_frontal_mean_NC = np.mean(SUVR_frontal_NC)
    SUVR_frontal_std_NC = np.std(SUVR_frontal_NC)
    
    SUVR_parietal_AD = [SUVRs_parietal[i] for i in ind_AD]
    SUVR_parietal_MCI = [SUVRs_parietal[i] for i in ind_MCI]
    SUVR_parietal_NC = [SUVRs_parietal[i] for i in ind_NC]
    SUVR_parietal_mean_AD = np.mean(SUVR_parietal_AD)
    SUVR_parietal_std_AD = np.std(SUVR_parietal_AD)
    SUVR_parietal_mean_MCI = np.mean(SUVR_parietal_MCI)
    SUVR_parietal_std_MCI = np.std(SUVR_parietal_MCI)
    SUVR_parietal_mean_NC = np.mean(SUVR_parietal_NC)
    SUVR_parietal_std_NC = np.std(SUVR_parietal_NC)
    
    SUVR_lat_temp_AD = [SUVRs_lat_temp[i] for i in ind_AD]
    SUVR_lat_temp_MCI = [SUVRs_lat_temp[i] for i in ind_MCI]
    SUVR_lat_temp_NC = [SUVRs_lat_temp[i] for i in ind_NC]
    SUVR_lat_temp_mean_AD = np.mean(SUVR_lat_temp_AD)
    SUVR_lat_temp_std_AD = np.std(SUVR_lat_temp_AD)
    SUVR_lat_temp_mean_MCI = np.mean(SUVR_lat_temp_MCI)
    SUVR_lat_temp_std_MCI = np.std(SUVR_lat_temp_MCI)
    SUVR_lat_temp_mean_NC = np.mean(SUVR_lat_temp_NC)
    SUVR_lat_temp_std_NC = np.std(SUVR_lat_temp_NC)
    
    SUVR_med_temp_AD = [SUVRs_med_temp[i] for i in ind_AD]
    SUVR_med_temp_MCI = [SUVRs_med_temp[i] for i in ind_MCI]
    SUVR_med_temp_NC = [SUVRs_med_temp[i] for i in ind_NC]
    SUVR_med_temp_mean_AD = np.mean(SUVR_med_temp_AD)
    SUVR_med_temp_std_AD = np.std(SUVR_med_temp_AD)
    SUVR_med_temp_mean_MCI = np.mean(SUVR_med_temp_MCI)
    SUVR_med_temp_std_MCI = np.std(SUVR_med_temp_MCI)
    SUVR_med_temp_mean_NC = np.mean(SUVR_med_temp_NC)
    SUVR_med_temp_std_NC = np.std(SUVR_med_temp_NC)
    
    SUVR_ant_cingulate_AD = [SUVRs_ant_cingulate[i] for i in ind_AD]
    SUVR_ant_cingulate_MCI = [SUVRs_ant_cingulate[i] for i in ind_MCI]
    SUVR_ant_cingulate_NC = [SUVRs_ant_cingulate[i] for i in ind_NC]
    SUVR_ant_cingulate_mean_AD = np.mean(SUVR_ant_cingulate_AD)
    SUVR_ant_cingulate_std_AD = np.std(SUVR_ant_cingulate_AD)
    SUVR_ant_cingulate_mean_MCI = np.mean(SUVR_ant_cingulate_MCI)
    SUVR_ant_cingulate_std_MCI = np.std(SUVR_ant_cingulate_MCI)
    SUVR_ant_cingulate_mean_NC = np.mean(SUVR_ant_cingulate_NC)
    SUVR_ant_cingulate_std_NC = np.std(SUVR_ant_cingulate_NC)
    
    SUVR_pos_cingulate_AD = [SUVRs_pos_cingulate[i] for i in ind_AD]
    SUVR_pos_cingulate_MCI = [SUVRs_pos_cingulate[i] for i in ind_MCI]
    SUVR_pos_cingulate_NC = [SUVRs_pos_cingulate[i] for i in ind_NC]
    SUVR_pos_cingulate_mean_AD = np.mean(SUVR_pos_cingulate_AD)
    SUVR_pos_cingulate_std_AD = np.std(SUVR_pos_cingulate_AD)
    SUVR_pos_cingulate_mean_MCI = np.mean(SUVR_pos_cingulate_MCI)
    SUVR_pos_cingulate_std_MCI = np.std(SUVR_pos_cingulate_MCI)
    SUVR_pos_cingulate_mean_NC = np.mean(SUVR_pos_cingulate_NC)
    SUVR_pos_cingulate_std_NC = np.std(SUVR_pos_cingulate_NC)

    SUVR_composite_AD = [SUVRs_composite[i] for i in ind_AD]
    SUVR_composite_MCI = [SUVRs_composite[i] for i in ind_MCI]
    SUVR_composite_NC = [SUVRs_composite[i] for i in ind_NC]
    SUVR_composite_mean_AD = np.mean(SUVR_composite_AD)
    SUVR_composite_std_AD = np.std(SUVR_composite_AD)
    SUVR_composite_mean_MCI = np.mean(SUVR_composite_MCI)
    SUVR_composite_std_MCI = np.std(SUVR_composite_MCI)
    SUVR_composite_mean_NC = np.mean(SUVR_composite_NC)
    SUVR_composite_std_NC = np.std(SUVR_composite_NC)

    # Create lists for the plot
    clinical_cat = ['AD', 'MCI', 'NC']
    x_pos = np.arange(len(clinical_cat))
    SUVR_braak12_mean = [SUVR_braak12_mean_AD, SUVR_braak12_mean_MCI, SUVR_braak12_mean_NC]
    SUVR_braak12_std = [SUVR_braak12_std_AD, SUVR_braak12_std_MCI, SUVR_braak12_std_NC]
    SUVR_braak34_mean = [SUVR_braak34_mean_AD, SUVR_braak34_mean_MCI, SUVR_braak34_mean_NC]
    SUVR_braak34_std = [SUVR_braak34_std_AD, SUVR_braak34_std_MCI, SUVR_braak34_std_NC]
    SUVR_braak56_mean = [SUVR_braak56_mean_AD, SUVR_braak56_mean_MCI, SUVR_braak56_mean_NC]
    SUVR_braak56_std = [SUVR_braak56_std_AD, SUVR_braak56_std_MCI, SUVR_braak56_std_NC]
    SUVR_meta_temp_mean = [SUVR_meta_temp_mean_AD, SUVR_meta_temp_mean_MCI, SUVR_meta_temp_mean_NC]
    SUVR_meta_temp_std = [SUVR_meta_temp_std_AD, SUVR_meta_temp_std_MCI, SUVR_meta_temp_std_NC]
    SUVR_frontal_mean = [SUVR_frontal_mean_AD, SUVR_frontal_mean_MCI, SUVR_frontal_mean_NC]
    SUVR_frontal_std = [SUVR_frontal_std_AD, SUVR_frontal_std_MCI, SUVR_frontal_std_NC]
    SUVR_parietal_mean = [SUVR_parietal_mean_AD, SUVR_parietal_mean_MCI, SUVR_parietal_mean_NC]
    SUVR_parietal_std = [SUVR_parietal_std_AD, SUVR_parietal_std_MCI, SUVR_parietal_std_NC]
    SUVR_lat_temp_mean = [SUVR_lat_temp_mean_AD, SUVR_lat_temp_mean_MCI, SUVR_lat_temp_mean_NC]
    SUVR_lat_temp_std = [SUVR_lat_temp_std_AD, SUVR_lat_temp_std_MCI, SUVR_lat_temp_std_NC]
    SUVR_med_temp_mean = [SUVR_med_temp_mean_AD, SUVR_med_temp_mean_MCI, SUVR_med_temp_mean_NC]
    SUVR_med_temp_std = [SUVR_med_temp_std_AD, SUVR_med_temp_std_MCI, SUVR_med_temp_std_NC]
    SUVR_ant_cingulate_mean = [SUVR_ant_cingulate_mean_AD, SUVR_ant_cingulate_mean_MCI, SUVR_ant_cingulate_mean_NC]
    SUVR_ant_cingulate_std = [SUVR_ant_cingulate_std_AD, SUVR_ant_cingulate_std_MCI, SUVR_ant_cingulate_std_NC]
    SUVR_pos_cingulate_mean = [SUVR_pos_cingulate_mean_AD, SUVR_pos_cingulate_mean_MCI, SUVR_pos_cingulate_mean_NC]
    SUVR_pos_cingulate_std = [SUVR_pos_cingulate_std_AD, SUVR_pos_cingulate_std_MCI, SUVR_pos_cingulate_std_NC]
    SUVR_composite_mean = [SUVR_composite_mean_AD, SUVR_composite_mean_MCI, SUVR_composite_mean_NC]
    SUVR_composite_std = [SUVR_composite_std_AD, SUVR_composite_std_MCI, SUVR_composite_std_NC]

    width = 0.2
    fig1, ax1 = plt.subplots(2, 2, figsize=(15,10))
    ax1[0, 0].bar(x_pos, height=SUVR_braak12_mean, width=width, yerr=SUVR_braak12_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[0, 0].set_title('Braak12', fontsize=16)
    ax1[0, 0].set_ylabel('SUVR',fontsize=12)
    ax1[0, 0].set_xticks(x_pos)
    ax1[0, 0].set_xticklabels(clinical_cat)
    ax1[0, 0].yaxis.grid(True)

    ax1[0, 1].bar(x_pos, height=SUVR_braak34_mean, width=width, yerr=SUVR_braak34_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[0, 1].set_title('Braak34', fontsize=16)
    ax1[0, 1].set_ylabel('SUVR',fontsize=12)
    ax1[0, 1].set_xticks(x_pos)
    ax1[0, 1].set_xticklabels(clinical_cat)
    ax1[0, 1].yaxis.grid(True)

    ax1[1, 0].bar(x_pos, height=SUVR_braak56_mean, width=width, yerr=SUVR_braak56_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[1, 0].set_title('Braak56', fontsize=16)
    ax1[1, 0].set_ylabel('SUVR',fontsize=12)
    ax1[1, 0].set_xticks(x_pos)
    ax1[1, 0].set_xticklabels(clinical_cat)
    ax1[1, 0].yaxis.grid(True)

    ax1[1, 1].bar(x_pos, height=SUVR_meta_temp_mean, width=width, yerr=SUVR_meta_temp_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[1, 1].set_title('Meta Temporal', fontsize=16)
    ax1[1, 1].set_ylabel('SUVR',fontsize=12)
    ax1[1, 1].set_xticks(x_pos)
    ax1[1, 1].set_xticklabels(clinical_cat)
    ax1[1, 1].yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    fig_path = os.path.join(root_dir, 'SUVR_real_tau_group_comparison.png')
    plt.savefig(fig_path)
    #plt.show()
    
    
 #   width = 0.2
    fig2, ax2 = plt.subplots(4, 2, figsize=(15,10))
    ax2[0, 0].bar(x_pos, height=SUVR_frontal_mean, width=width, yerr=SUVR_frontal_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax2[0, 0].set_title('Frontal Cortex', fontsize=16)
    ax2[0, 0].set_ylabel('SUVR',fontsize=12)
    ax2[0, 0].set_xticks(x_pos)
    ax2[0, 0].set_xticklabels(clinical_cat)
    ax2[0, 0].yaxis.grid(True)

    ax2[0, 1].bar(x_pos, height=SUVR_parietal_mean, width=width, yerr=SUVR_parietal_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax2[0, 1].set_title('Parietal Cortex', fontsize=16)
    ax2[0, 1].set_ylabel('SUVR',fontsize=12)
    ax2[0, 1].set_xticks(x_pos)
    ax2[0, 1].set_xticklabels(clinical_cat)
    ax2[0, 1].yaxis.grid(True)

    ax2[1, 0].bar(x_pos, height=SUVR_lat_temp_mean, width=width, yerr=SUVR_lat_temp_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax2[1, 0].set_title('Lateral Temporal Cortex', fontsize=16)
    ax2[1, 0].set_ylabel('SUVR',fontsize=12)
    ax2[1, 0].set_xticks(x_pos)
    ax2[1, 0].set_xticklabels(clinical_cat)
    ax2[1, 0].yaxis.grid(True)

    ax2[1, 1].bar(x_pos, height=SUVR_med_temp_mean, width=width, yerr=SUVR_med_temp_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax2[1, 1].set_title('Medial Temporal Cortex', fontsize=16)
    ax2[1, 1].set_ylabel('SUVR',fontsize=12)
    ax2[1, 1].set_xticks(x_pos)
    ax2[1, 1].set_xticklabels(clinical_cat)
    ax2[1, 1].yaxis.grid(True)

    ax2[2, 0].bar(x_pos, height=SUVR_ant_cingulate_mean, width=width, yerr=SUVR_ant_cingulate_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax2[2, 0].set_title('Anterior Cingulate Cortex', fontsize=16)
    ax2[2, 0].set_ylabel('SUVR',fontsize=12)
    ax2[2, 0].set_xticks(x_pos)
    ax2[2, 0].set_xticklabels(clinical_cat)
    ax2[2, 0].yaxis.grid(True)

    ax2[2, 1].bar(x_pos, height=SUVR_pos_cingulate_mean, width=width, yerr=SUVR_pos_cingulate_std, color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax2[2, 1].set_title('Posterior Cingulate Cortex', fontsize=16)
    ax2[2, 1].set_ylabel('SUVR',fontsize=12)
    ax2[2, 1].set_xticks(x_pos)
    ax2[2, 1].set_xticklabels(clinical_cat)
    ax2[2, 1].yaxis.grid(True)

    ax2[3, 0].bar(x_pos, height=SUVR_composite_mean, width=width, yerr=SUVR_composite_std, color='r', align='center',
                  alpha=0.5, ecolor='black', capsize=5)
    ax2[3, 0].set_title('Composite Cortex', fontsize=16)
    ax2[3, 0].set_ylabel('SUVR', fontsize=12)
    ax2[3, 0].set_xticks(x_pos)
    ax2[3, 0].set_xticklabels(clinical_cat)
    ax2[3, 0].yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    fig_path = os.path.join(root_dir, 'SUVR_real_amyloid_group_comparison.png')
    plt.savefig(fig_path)
    plt.show()

# COMMAND ----------

SUVR_real_ROI_calculation()