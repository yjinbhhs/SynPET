# Databricks notebook source
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import openpyxl
import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.pyplot import figure
import matplotlib

def SUVR_validation():
    root_dir = '/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn'
    syn_image_dir = 'gan_data/outputs/PETSyn-3DWGAN-GP/results_contra_gpu/images'
    pet_dir = 'pet'
    sub_file = 'PETSyn_Pilot_Pair.csv'
    real_image_suffix = '_PET_to_oasis_SUVR_norm_cropped.nii.gz'
    real_orig_image_suffix = '_PET_to_oasis_SUVR_cropped.nii.gz'
    SUVR_front_suffix = '_roi_frontal_to_oasis_cropped.nii.gz'
    SUVR_lat_temp_suffix = '_roi_lateral_temporal_to_oasis_cropped.nii.gz'
    SUVR_med_temp_suffix = '_roi_medial_temporal_to_oasis_cropped.nii.gz'
    SUVR_pari_suffix = '_roi_parietal_to_oasis_cropped.nii.gz'
    SUVR_ant_cing_suffix = '_roi_ant_cingulate_to_oasis_cropped.nii.gz'
    SUVR_pos_cing_suffix = '_roi_pos_cingulate_to_oasis_cropped.nii.gz'
    SUVR_composite_suffix = '_roi_composite_to_oasis_cropped.nii.gz'

    sub_df = pd.read_csv(os.path.join(root_dir, sub_file))

    syn_image_path = os.path.join(root_dir, syn_image_dir)
    syn_image_list = os.listdir(syn_image_path)
    syn_image_list.sort()
    syn_image_names = []
    for f in syn_image_list:
        syn_image_names.append(os.path.join(syn_image_path, f))

    sub_names = []
    group_ids = []
    SUVRs_front_real = []
    SUVRs_front_syn = []
    SUVRs_front_error = []
    SUVRs_lat_temp_real = []
    SUVRs_lat_temp_syn = []
    SUVRs_lat_temp_error = []
    SUVRs_med_temp_real = []
    SUVRs_med_temp_syn = []
    SUVRs_med_temp_error = []
    SUVRs_pari_real = []
    SUVRs_pari_syn = []
    SUVRs_pari_error = []
    SUVRs_ant_cing_real = []
    SUVRs_ant_cing_syn = []
    SUVRs_ant_cing_error = []
    SUVRs_pos_cing_real = []
    SUVRs_pos_cing_syn = []
    SUVRs_pos_cing_error = []
    SUVRs_composite_real = []
    SUVRs_composite_orig_real = []
    SUVRs_composite_syn = []
    SUVRs_composite_error = []

    for syn_image in syn_image_names:
        syn_image_sitk = sitk.ReadImage(syn_image)
        syn_image_np = sitk.GetArrayFromImage(syn_image_sitk).astype(np.float32)

        # Locate group category
        sub_name = syn_image.split("/")
        sub_name = sub_name[-1]
        sub_name = sub_name.split(".")
        sub_name = sub_name[0]
        sub_name = sub_name.split("_")
        sub_name = sub_name[0] + '_' + sub_name[1] + '_' + sub_name[2] + '_' + sub_name[3]
        print(sub_name)
        sub_names.append(sub_name)
        group_id = sub_df.loc[sub_df['PET_ID'] == sub_name, 'Group_ID'].iloc[0]
        group_ids.append(group_id)

        real_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + real_image_suffix)
        real_image_sitk = sitk.ReadImage(real_image_name)
        real_image_np = sitk.GetArrayFromImage(real_image_sitk).astype(np.float32)
        
        real_orig_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + real_orig_image_suffix)
        real_orig_image_sitk = sitk.ReadImage(real_orig_image_name)
        real_orig_image_np = sitk.GetArrayFromImage(real_orig_image_sitk).astype(np.float32)

        # Calculate SUVR in ROIs

        SUVR_front_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + SUVR_front_suffix)
        SUVR_front_sitk = sitk.ReadImage(SUVR_front_image_name)
        SUVR_front_image_np = sitk.GetArrayFromImage(SUVR_front_sitk).astype(np.float32)
        SUVR_front_real = np.sum(SUVR_front_image_np * real_image_np) / np.count_nonzero(SUVR_front_image_np)
        SUVRs_front_real.append(SUVR_front_real)
        SUVR_front_syn = np.sum(SUVR_front_image_np * syn_image_np) / np.count_nonzero(SUVR_front_image_np)
        SUVRs_front_syn.append(SUVR_front_syn)
        SUVR_front_error = 100 * abs(SUVR_front_syn - SUVR_front_real) / SUVR_front_real
        SUVRs_front_error.append(SUVR_front_error)

        SUVR_lat_temp_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + SUVR_lat_temp_suffix)
        SUVR_lat_temp_sitk = sitk.ReadImage(SUVR_lat_temp_image_name)
        SUVR_lat_temp_image_np = sitk.GetArrayFromImage(SUVR_lat_temp_sitk).astype(np.float32)
        SUVR_lat_temp_real = np.sum(SUVR_lat_temp_image_np * real_image_np) / np.count_nonzero(SUVR_lat_temp_image_np)
        SUVRs_lat_temp_real.append(SUVR_lat_temp_real)
        SUVR_lat_temp_syn = np.sum(SUVR_lat_temp_image_np * syn_image_np) / np.count_nonzero(SUVR_lat_temp_image_np)
        SUVRs_lat_temp_syn.append(SUVR_lat_temp_syn)
        SUVR_lat_temp_error = 100 * abs(SUVR_lat_temp_syn - SUVR_lat_temp_real) / SUVR_lat_temp_real
        SUVRs_lat_temp_error.append(SUVR_lat_temp_error)

        SUVR_med_temp_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + SUVR_med_temp_suffix)
        SUVR_med_temp_sitk = sitk.ReadImage(SUVR_med_temp_image_name)
        SUVR_med_temp_image_np = sitk.GetArrayFromImage(SUVR_med_temp_sitk).astype(np.float32)
        SUVR_med_temp_real = np.sum(SUVR_med_temp_image_np * real_image_np) / np.count_nonzero(SUVR_med_temp_image_np)
        SUVRs_med_temp_real.append(SUVR_med_temp_real)
        SUVR_med_temp_syn = np.sum(SUVR_med_temp_image_np * syn_image_np) / np.count_nonzero(SUVR_med_temp_image_np)
        SUVRs_med_temp_syn.append(SUVR_med_temp_syn)
        SUVR_med_temp_error = 100 * abs(SUVR_med_temp_syn - SUVR_med_temp_real) / SUVR_med_temp_real
        SUVRs_med_temp_error.append(SUVR_med_temp_error)

        SUVR_pari_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + SUVR_pari_suffix)
        SUVR_pari_sitk = sitk.ReadImage(SUVR_pari_image_name)
        SUVR_pari_image_np = sitk.GetArrayFromImage(SUVR_pari_sitk).astype(np.float32)
        SUVR_pari_real = np.sum(SUVR_pari_image_np * real_image_np) / np.count_nonzero(SUVR_pari_image_np)
        SUVRs_pari_real.append(SUVR_pari_real)
        SUVR_pari_syn = np.sum(SUVR_pari_image_np * syn_image_np) / np.count_nonzero(SUVR_pari_image_np)
        SUVRs_pari_syn.append(SUVR_pari_syn)
        SUVR_pari_error = 100 * abs(SUVR_pari_syn - SUVR_pari_real) / SUVR_pari_real
        SUVRs_pari_error.append(SUVR_pari_error)

        SUVR_ant_cing_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + SUVR_ant_cing_suffix)
        SUVR_ant_cing_sitk = sitk.ReadImage(SUVR_ant_cing_image_name)
        SUVR_ant_cing_image_np = sitk.GetArrayFromImage(SUVR_ant_cing_sitk).astype(np.float32)
        SUVR_ant_cing_real = np.sum(SUVR_ant_cing_image_np * real_image_np) / np.count_nonzero(SUVR_ant_cing_image_np)
        SUVRs_ant_cing_real.append(SUVR_ant_cing_real)
        SUVR_ant_cing_syn = np.sum(SUVR_ant_cing_image_np * syn_image_np) / np.count_nonzero(SUVR_ant_cing_image_np)
        SUVRs_ant_cing_syn.append(SUVR_ant_cing_syn)
        SUVR_ant_cing_error = 100 * abs(SUVR_ant_cing_syn - SUVR_ant_cing_real) / SUVR_ant_cing_real
        SUVRs_ant_cing_error.append(SUVR_ant_cing_error)

        SUVR_pos_cing_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + SUVR_pos_cing_suffix)
        SUVR_pos_cing_sitk = sitk.ReadImage(SUVR_pos_cing_image_name)
        SUVR_pos_cing_image_np = sitk.GetArrayFromImage(SUVR_pos_cing_sitk).astype(np.float32)
        SUVR_pos_cing_real = np.sum(SUVR_pos_cing_image_np * real_image_np) / np.count_nonzero(SUVR_pos_cing_image_np)
        SUVRs_pos_cing_real.append(SUVR_pos_cing_real)
        SUVR_pos_cing_syn = np.sum(SUVR_pos_cing_image_np * syn_image_np) / np.count_nonzero(SUVR_pos_cing_image_np)
        SUVRs_pos_cing_syn.append(SUVR_pos_cing_syn)
        SUVR_pos_cing_error = 100 * abs(SUVR_pos_cing_syn - SUVR_pos_cing_real) / SUVR_pos_cing_real
        SUVRs_pos_cing_error.append(SUVR_pos_cing_error)

        SUVR_composite_image_name = os.path.join(root_dir, pet_dir, sub_name, sub_name + SUVR_composite_suffix)
        SUVR_composite_sitk = sitk.ReadImage(SUVR_composite_image_name)
        SUVR_composite_image_np = sitk.GetArrayFromImage(SUVR_composite_sitk).astype(np.float32)
        SUVR_composite_real = np.sum(SUVR_composite_image_np * real_image_np) / np.count_nonzero(SUVR_composite_image_np)
        SUVRs_composite_real.append(SUVR_composite_real)
        SUVR_composite_orig_real = np.sum(SUVR_composite_image_np * real_orig_image_np) / np.count_nonzero(SUVR_composite_image_np)
        SUVRs_composite_orig_real.append(SUVR_composite_orig_real)
        SUVR_composite_syn = np.sum(SUVR_composite_image_np * syn_image_np) / np.count_nonzero(SUVR_composite_image_np)
        SUVRs_composite_syn.append(SUVR_composite_syn)
        SUVR_composite_error = 100 * abs(SUVR_composite_syn - SUVR_composite_real) / SUVR_composite_real
        SUVRs_composite_error.append(SUVR_composite_error)

    SUVR_list = {
        'PET_ID': sub_names,
        'Group_ID': group_ids,
        'SUVR_front_real': np.round(SUVRs_front_real, 4),
        'SUVR_front_syn': np.round(SUVRs_front_syn, 4),
        'SUVR_front_error': np.round(SUVRs_front_error, 4),
        'SUVR_lat_temp_real': np.round(SUVRs_lat_temp_real, 4),
        'SUVR_lat_temp_syn': np.round(SUVRs_lat_temp_syn, 4),
        'SUVR_lat_temp_error': np.round(SUVRs_lat_temp_error, 4),
        'SUVR_med_temp_real': np.round(SUVRs_med_temp_real, 4),
        'SUVR_med_temp_syn': np.round(SUVRs_med_temp_syn, 4),
        'SUVR_med_temp_error': np.round(SUVRs_med_temp_error, 4),
        'SUVR_pari_real': np.round(SUVRs_pari_real, 4),
        'SUVR_pari_syn': np.round(SUVRs_pari_syn, 4),
        'SUVR_pari_error': np.round(SUVRs_pari_error, 4),
        'SUVR_ant_cing_real': np.round(SUVRs_ant_cing_real, 4),
        'SUVR_ant_cing_syn': np.round(SUVRs_ant_cing_syn, 4),
        'SUVR_ant_cing_error': np.round(SUVRs_ant_cing_error, 4),
        'SUVR_pos_cing_real': np.round(SUVRs_pos_cing_real, 4),
        'SUVR_pos_cing_syn': np.round(SUVRs_pos_cing_syn, 4),
        'SUVR_pos_cing_error': np.round(SUVRs_pos_cing_error, 4),
        'SUVR_composite_real': np.round(SUVRs_composite_real, 4),
        'SUVR_composit_orig_real': np.round(SUVRs_composite_orig_real, 4),
        'SUVR_composite_syn': np.round(SUVRs_composite_syn, 4),
        'SUVR_composite_error': np.round(SUVRs_composite_error, 4)
    }

    keys = SUVR_list.keys()
    csv_path = os.path.join(root_dir, 'SUVR.csv')
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(keys)
        writer.writerows(zip(*[SUVR_list[key] for key in keys]))

    ind_AD = [i for i, x in enumerate(group_ids) if x == 'AD']
    ind_MCI = [i for i, x in enumerate(group_ids) if x == 'MCI']
    ind_NC = [i for i, x in enumerate(group_ids) if x == 'CN']

    SUVR_front_real_AD = [SUVRs_front_real[i] for i in ind_AD]
    SUVR_front_real_MCI = [SUVRs_front_real[i] for i in ind_MCI]
    SUVR_front_real_NC = [SUVRs_front_real[i] for i in ind_NC]
    SUVR_front_real_mean_AD = np.mean(SUVR_front_real_AD)
    SUVR_front_real_mean_MCI = np.mean(SUVR_front_real_MCI)
    SUVR_front_real_mean_NC = np.mean(SUVR_front_real_NC)
    SUVR_front_syn_AD = [SUVRs_front_syn[i] for i in ind_AD]
    SUVR_front_syn_MCI = [SUVRs_front_syn[i] for i in ind_MCI]
    SUVR_front_syn_NC = [SUVRs_front_syn[i] for i in ind_NC]
    SUVR_front_syn_mean_AD = np.mean(SUVR_front_syn_AD)
    SUVR_front_syn_mean_MCI = np.mean(SUVR_front_syn_MCI)
    SUVR_front_syn_mean_NC = np.mean(SUVR_front_syn_NC)
    SUVR_front_error_AD = [SUVRs_front_error[i] for i in ind_AD]
    SUVR_front_error_MCI = [SUVRs_front_error[i] for i in ind_MCI]
    SUVR_front_error_NC = [SUVRs_front_error[i] for i in ind_NC]
    SUVR_front_error_mean_AD = np.mean(SUVR_front_error_AD)
    SUVR_front_error_mean_MCI = np.mean(SUVR_front_error_MCI)
    SUVR_front_error_mean_NC = np.mean(SUVR_front_error_NC)
    SUVR_front_real_std_AD = np.std(SUVR_front_real_AD)
    SUVR_front_real_std_MCI = np.std(SUVR_front_real_MCI)
    SUVR_front_real_std_NC = np.std(SUVR_front_real_NC)
    SUVR_front_syn_std_AD = np.std(SUVR_front_syn_AD)
    SUVR_front_syn_std_MCI = np.std(SUVR_front_syn_MCI)
    SUVR_front_syn_std_NC = np.std(SUVR_front_syn_NC)
    SUVR_front_error_std_AD = np.std(SUVR_front_error_AD)
    SUVR_front_error_std_MCI = np.std(SUVR_front_error_MCI)
    SUVR_front_error_std_NC = np.std(SUVR_front_error_NC)

    SUVR_lat_temp_real_AD = [SUVRs_lat_temp_real[i] for i in ind_AD]
    SUVR_lat_temp_real_MCI = [SUVRs_lat_temp_real[i] for i in ind_MCI]
    SUVR_lat_temp_real_NC = [SUVRs_lat_temp_real[i] for i in ind_NC]
    SUVR_lat_temp_real_mean_AD = np.mean(SUVR_lat_temp_real_AD)
    SUVR_lat_temp_real_mean_MCI = np.mean(SUVR_lat_temp_real_MCI)
    SUVR_lat_temp_real_mean_NC = np.mean(SUVR_lat_temp_real_NC)
    SUVR_lat_temp_syn_AD = [SUVRs_lat_temp_syn[i] for i in ind_AD]
    SUVR_lat_temp_syn_MCI = [SUVRs_lat_temp_syn[i] for i in ind_MCI]
    SUVR_lat_temp_syn_NC = [SUVRs_lat_temp_syn[i] for i in ind_NC]
    SUVR_lat_temp_syn_mean_AD = np.mean(SUVR_lat_temp_syn_AD)
    SUVR_lat_temp_syn_mean_MCI = np.mean(SUVR_lat_temp_syn_MCI)
    SUVR_lat_temp_syn_mean_NC = np.mean(SUVR_lat_temp_syn_NC)
    SUVR_lat_temp_error_AD = [SUVRs_lat_temp_error[i] for i in ind_AD]
    SUVR_lat_temp_error_MCI = [SUVRs_lat_temp_error[i] for i in ind_MCI]
    SUVR_lat_temp_error_NC = [SUVRs_lat_temp_error[i] for i in ind_NC]
    SUVR_lat_temp_error_mean_AD = np.mean(SUVR_lat_temp_error_AD)
    SUVR_lat_temp_error_mean_MCI = np.mean(SUVR_lat_temp_error_MCI)
    SUVR_lat_temp_error_mean_NC = np.mean(SUVR_lat_temp_error_NC)
    SUVR_lat_temp_real_std_AD = np.std(SUVR_lat_temp_real_AD)
    SUVR_lat_temp_real_std_MCI = np.std(SUVR_lat_temp_real_MCI)
    SUVR_lat_temp_real_std_NC = np.std(SUVR_lat_temp_real_NC)
    SUVR_lat_temp_syn_std_AD = np.std(SUVR_lat_temp_syn_AD)
    SUVR_lat_temp_syn_std_MCI = np.std(SUVR_lat_temp_syn_MCI)
    SUVR_lat_temp_syn_std_NC = np.std(SUVR_lat_temp_syn_NC)
    SUVR_lat_temp_error_std_AD = np.std(SUVR_lat_temp_error_AD)
    SUVR_lat_temp_error_std_MCI = np.std(SUVR_lat_temp_error_MCI)
    SUVR_lat_temp_error_std_NC = np.std(SUVR_lat_temp_error_NC)

    SUVR_med_temp_real_AD = [SUVRs_med_temp_real[i] for i in ind_AD]
    SUVR_med_temp_real_MCI = [SUVRs_med_temp_real[i] for i in ind_MCI]
    SUVR_med_temp_real_NC = [SUVRs_med_temp_real[i] for i in ind_NC]
    SUVR_med_temp_real_mean_AD = np.mean(SUVR_med_temp_real_AD)
    SUVR_med_temp_real_mean_MCI = np.mean(SUVR_med_temp_real_MCI)
    SUVR_med_temp_real_mean_NC = np.mean(SUVR_med_temp_real_NC)
    SUVR_med_temp_syn_AD = [SUVRs_med_temp_syn[i] for i in ind_AD]
    SUVR_med_temp_syn_MCI = [SUVRs_med_temp_syn[i] for i in ind_MCI]
    SUVR_med_temp_syn_NC = [SUVRs_med_temp_syn[i] for i in ind_NC]
    SUVR_med_temp_syn_mean_AD = np.mean(SUVR_med_temp_syn_AD)
    SUVR_med_temp_syn_mean_MCI = np.mean(SUVR_med_temp_syn_MCI)
    SUVR_med_temp_syn_mean_NC = np.mean(SUVR_med_temp_syn_NC)
    SUVR_med_temp_error_AD = [SUVRs_med_temp_error[i] for i in ind_AD]
    SUVR_med_temp_error_MCI = [SUVRs_med_temp_error[i] for i in ind_MCI]
    SUVR_med_temp_error_NC = [SUVRs_med_temp_error[i] for i in ind_NC]
    SUVR_med_temp_error_mean_AD = np.mean(SUVR_med_temp_error_AD)
    SUVR_med_temp_error_mean_MCI = np.mean(SUVR_med_temp_error_MCI)
    SUVR_med_temp_error_mean_NC = np.mean(SUVR_med_temp_error_NC)
    SUVR_med_temp_real_std_AD = np.std(SUVR_med_temp_real_AD)
    SUVR_med_temp_real_std_MCI = np.std(SUVR_med_temp_real_MCI)
    SUVR_med_temp_real_std_NC = np.std(SUVR_med_temp_real_NC)
    SUVR_med_temp_syn_std_AD = np.std(SUVR_med_temp_syn_AD)
    SUVR_med_temp_syn_std_MCI = np.std(SUVR_med_temp_syn_MCI)
    SUVR_med_temp_syn_std_NC = np.std(SUVR_med_temp_syn_NC)
    SUVR_med_temp_error_std_AD = np.std(SUVR_med_temp_error_AD)
    SUVR_med_temp_error_std_MCI = np.std(SUVR_med_temp_error_MCI)
    SUVR_med_temp_error_std_NC = np.std(SUVR_med_temp_error_NC)

    SUVR_pari_real_AD = [SUVRs_pari_real[i] for i in ind_AD]
    SUVR_pari_real_MCI = [SUVRs_pari_real[i] for i in ind_MCI]
    SUVR_pari_real_NC = [SUVRs_pari_real[i] for i in ind_NC]
    SUVR_pari_real_mean_AD = np.mean(SUVR_pari_real_AD)
    SUVR_pari_real_mean_MCI = np.mean(SUVR_pari_real_MCI)
    SUVR_pari_real_mean_NC = np.mean(SUVR_pari_real_NC)
    SUVR_pari_syn_AD = [SUVRs_pari_syn[i] for i in ind_AD]
    SUVR_pari_syn_MCI = [SUVRs_pari_syn[i] for i in ind_MCI]
    SUVR_pari_syn_NC = [SUVRs_pari_syn[i] for i in ind_NC]
    SUVR_pari_syn_mean_AD = np.mean(SUVR_pari_syn_AD)
    SUVR_pari_syn_mean_MCI = np.mean(SUVR_pari_syn_MCI)
    SUVR_pari_syn_mean_NC = np.mean(SUVR_pari_syn_NC)
    SUVR_pari_error_AD = [SUVRs_pari_error[i] for i in ind_AD]
    SUVR_pari_error_MCI = [SUVRs_pari_error[i] for i in ind_MCI]
    SUVR_pari_error_NC = [SUVRs_pari_error[i] for i in ind_NC]
    SUVR_pari_error_mean_AD = np.mean(SUVR_pari_error_AD)
    SUVR_pari_error_mean_MCI = np.mean(SUVR_pari_error_MCI)
    SUVR_pari_error_mean_NC = np.mean(SUVR_pari_error_NC)
    SUVR_pari_real_std_AD = np.std(SUVR_pari_real_AD)
    SUVR_pari_real_std_MCI = np.std(SUVR_pari_real_MCI)
    SUVR_pari_real_std_NC = np.std(SUVR_pari_real_NC)
    SUVR_pari_syn_std_AD = np.std(SUVR_pari_syn_AD)
    SUVR_pari_syn_std_MCI = np.std(SUVR_pari_syn_MCI)
    SUVR_pari_syn_std_NC = np.std(SUVR_pari_syn_NC)
    SUVR_pari_error_std_AD = np.std(SUVR_pari_error_AD)
    SUVR_pari_error_std_MCI = np.std(SUVR_pari_error_MCI)
    SUVR_pari_error_std_NC = np.std(SUVR_pari_error_NC)

    SUVR_ant_cing_real_AD = [SUVRs_ant_cing_real[i] for i in ind_AD]
    SUVR_ant_cing_real_MCI = [SUVRs_ant_cing_real[i] for i in ind_MCI]
    SUVR_ant_cing_real_NC = [SUVRs_ant_cing_real[i] for i in ind_NC]
    SUVR_ant_cing_real_mean_AD = np.mean(SUVR_ant_cing_real_AD)
    SUVR_ant_cing_real_mean_MCI = np.mean(SUVR_ant_cing_real_MCI)
    SUVR_ant_cing_real_mean_NC = np.mean(SUVR_ant_cing_real_NC)
    SUVR_ant_cing_syn_AD = [SUVRs_ant_cing_syn[i] for i in ind_AD]
    SUVR_ant_cing_syn_MCI = [SUVRs_ant_cing_syn[i] for i in ind_MCI]
    SUVR_ant_cing_syn_NC = [SUVRs_ant_cing_syn[i] for i in ind_NC]
    SUVR_ant_cing_syn_mean_AD = np.mean(SUVR_ant_cing_syn_AD)
    SUVR_ant_cing_syn_mean_MCI = np.mean(SUVR_ant_cing_syn_MCI)
    SUVR_ant_cing_syn_mean_NC = np.mean(SUVR_ant_cing_syn_NC)
    SUVR_ant_cing_error_AD = [SUVRs_ant_cing_error[i] for i in ind_AD]
    SUVR_ant_cing_error_MCI = [SUVRs_ant_cing_error[i] for i in ind_MCI]
    SUVR_ant_cing_error_NC = [SUVRs_ant_cing_error[i] for i in ind_NC]
    SUVR_ant_cing_error_mean_AD = np.mean(SUVR_ant_cing_error_AD)
    SUVR_ant_cing_error_mean_MCI = np.mean(SUVR_ant_cing_error_MCI)
    SUVR_ant_cing_error_mean_NC = np.mean(SUVR_ant_cing_error_NC)
    SUVR_ant_cing_real_std_AD = np.std(SUVR_ant_cing_real_AD)
    SUVR_ant_cing_real_std_MCI = np.std(SUVR_ant_cing_real_MCI)
    SUVR_ant_cing_real_std_NC = np.std(SUVR_ant_cing_real_NC)
    SUVR_ant_cing_syn_std_AD = np.std(SUVR_ant_cing_syn_AD)
    SUVR_ant_cing_syn_std_MCI = np.std(SUVR_ant_cing_syn_MCI)
    SUVR_ant_cing_syn_std_NC = np.std(SUVR_ant_cing_syn_NC)
    SUVR_ant_cing_error_std_AD = np.std(SUVR_ant_cing_error_AD)
    SUVR_ant_cing_error_std_MCI = np.std(SUVR_ant_cing_error_MCI)
    SUVR_ant_cing_error_std_NC = np.std(SUVR_ant_cing_error_NC)

    SUVR_pos_cing_real_AD = [SUVRs_pos_cing_real[i] for i in ind_AD]
    SUVR_pos_cing_real_MCI = [SUVRs_pos_cing_real[i] for i in ind_MCI]
    SUVR_pos_cing_real_NC = [SUVRs_pos_cing_real[i] for i in ind_NC]
    SUVR_pos_cing_real_mean_AD = np.mean(SUVR_pos_cing_real_AD)
    SUVR_pos_cing_real_mean_MCI = np.mean(SUVR_pos_cing_real_MCI)
    SUVR_pos_cing_real_mean_NC = np.mean(SUVR_pos_cing_real_NC)
    SUVR_pos_cing_syn_AD = [SUVRs_pos_cing_syn[i] for i in ind_AD]
    SUVR_pos_cing_syn_MCI = [SUVRs_pos_cing_syn[i] for i in ind_MCI]
    SUVR_pos_cing_syn_NC = [SUVRs_pos_cing_syn[i] for i in ind_NC]
    SUVR_pos_cing_syn_mean_AD = np.mean(SUVR_pos_cing_syn_AD)
    SUVR_pos_cing_syn_mean_MCI = np.mean(SUVR_pos_cing_syn_MCI)
    SUVR_pos_cing_syn_mean_NC = np.mean(SUVR_pos_cing_syn_NC)
    SUVR_pos_cing_error_AD = [SUVRs_pos_cing_error[i] for i in ind_AD]
    SUVR_pos_cing_error_MCI = [SUVRs_pos_cing_error[i] for i in ind_MCI]
    SUVR_pos_cing_error_NC = [SUVRs_pos_cing_error[i] for i in ind_NC]
    SUVR_pos_cing_error_mean_AD = np.mean(SUVR_pos_cing_error_AD)
    SUVR_pos_cing_error_mean_MCI = np.mean(SUVR_pos_cing_error_MCI)
    SUVR_pos_cing_error_mean_NC = np.mean(SUVR_pos_cing_error_NC)
    SUVR_pos_cing_real_std_AD = np.std(SUVR_pos_cing_real_AD)
    SUVR_pos_cing_real_std_MCI = np.std(SUVR_pos_cing_real_MCI)
    SUVR_pos_cing_real_std_NC = np.std(SUVR_pos_cing_real_NC)
    SUVR_pos_cing_syn_std_AD = np.std(SUVR_pos_cing_syn_AD)
    SUVR_pos_cing_syn_std_MCI = np.std(SUVR_pos_cing_syn_MCI)
    SUVR_pos_cing_syn_std_NC = np.std(SUVR_pos_cing_syn_NC)
    SUVR_pos_cing_error_std_AD = np.std(SUVR_pos_cing_error_AD)
    SUVR_pos_cing_error_std_MCI = np.std(SUVR_pos_cing_error_MCI)
    SUVR_pos_cing_error_std_NC = np.std(SUVR_pos_cing_error_NC)

    SUVR_composite_real_AD = [SUVRs_composite_real[i] for i in ind_AD]
    SUVR_composite_real_MCI = [SUVRs_composite_real[i] for i in ind_MCI]
    SUVR_composite_real_NC = [SUVRs_composite_real[i] for i in ind_NC]
    SUVR_composite_real_mean_AD = np.mean(SUVR_composite_real_AD)
    SUVR_composite_real_mean_MCI = np.mean(SUVR_composite_real_MCI)
    SUVR_composite_real_mean_NC = np.mean(SUVR_composite_real_NC)
    SUVR_composite_syn_AD = [SUVRs_composite_syn[i] for i in ind_AD]
    SUVR_composite_syn_MCI = [SUVRs_composite_syn[i] for i in ind_MCI]
    SUVR_composite_syn_NC = [SUVRs_composite_syn[i] for i in ind_NC]
    SUVR_composite_syn_mean_AD = np.mean(SUVR_composite_syn_AD)
    SUVR_composite_syn_mean_MCI = np.mean(SUVR_composite_syn_MCI)
    SUVR_composite_syn_mean_NC = np.mean(SUVR_composite_syn_NC)
    SUVR_composite_error_AD = [SUVRs_composite_error[i] for i in ind_AD]
    SUVR_composite_error_MCI = [SUVRs_composite_error[i] for i in ind_MCI]
    SUVR_composite_error_NC = [SUVRs_composite_error[i] for i in ind_NC]
    SUVR_composite_error_mean_AD = np.mean(SUVR_composite_error_AD)
    SUVR_composite_error_mean_MCI = np.mean(SUVR_composite_error_MCI)
    SUVR_composite_error_mean_NC = np.mean(SUVR_composite_error_NC)
    SUVR_composite_real_std_AD = np.std(SUVR_composite_real_AD)
    SUVR_composite_real_std_MCI = np.std(SUVR_composite_real_MCI)
    SUVR_composite_real_std_NC = np.std(SUVR_composite_real_NC)
    SUVR_composite_syn_std_AD = np.std(SUVR_composite_syn_AD)
    SUVR_composite_syn_std_MCI = np.std(SUVR_composite_syn_MCI)
    SUVR_composite_syn_std_NC = np.std(SUVR_composite_syn_NC)
    SUVR_composite_error_std_AD = np.std(SUVR_composite_error_AD)
    SUVR_composite_error_std_MCI = np.std(SUVR_composite_error_MCI)
    SUVR_composite_error_std_NC = np.std(SUVR_composite_error_NC)

    # Create lists for the plot
    clinical_cat = ['AD', 'MCI', 'NC']
    x_pos = np.arange(len(clinical_cat))
    SUVR_front_real_mean = [SUVR_front_real_mean_AD, SUVR_front_real_mean_MCI, SUVR_front_real_mean_NC]
    SUVR_front_syn_mean = [SUVR_front_syn_mean_AD, SUVR_front_syn_mean_MCI, SUVR_front_syn_mean_NC]
    SUVR_front_error_mean = [SUVR_front_error_mean_AD, SUVR_front_error_mean_MCI, SUVR_front_error_mean_NC]
    SUVR_front_real_std = [SUVR_front_real_std_AD, SUVR_front_real_std_MCI, SUVR_front_real_std_NC]
    SUVR_front_syn_std = [SUVR_front_syn_std_AD, SUVR_front_syn_std_MCI, SUVR_front_syn_std_NC]
    SUVR_front_error_std = [SUVR_front_error_std_AD, SUVR_front_error_std_MCI, SUVR_front_error_std_NC]
    SUVR_lat_temp_real_mean = [SUVR_lat_temp_real_mean_AD, SUVR_lat_temp_real_mean_MCI, SUVR_lat_temp_real_mean_NC]
    SUVR_lat_temp_syn_mean = [SUVR_lat_temp_syn_mean_AD, SUVR_lat_temp_syn_mean_MCI, SUVR_lat_temp_syn_mean_NC]
    SUVR_lat_temp_error_mean = [SUVR_lat_temp_error_mean_AD, SUVR_lat_temp_error_mean_MCI, SUVR_lat_temp_error_mean_NC]
    SUVR_lat_temp_real_std = [SUVR_lat_temp_real_std_AD, SUVR_lat_temp_real_std_MCI, SUVR_lat_temp_real_std_NC]
    SUVR_lat_temp_syn_std = [SUVR_lat_temp_syn_std_AD, SUVR_lat_temp_syn_std_MCI, SUVR_lat_temp_syn_std_NC]
    SUVR_lat_temp_error_std = [SUVR_lat_temp_error_std_AD, SUVR_lat_temp_error_std_MCI, SUVR_lat_temp_error_std_NC]
    SUVR_med_temp_real_mean = [SUVR_med_temp_real_mean_AD, SUVR_med_temp_real_mean_MCI, SUVR_med_temp_real_mean_NC]
    SUVR_med_temp_syn_mean = [SUVR_med_temp_syn_mean_AD, SUVR_med_temp_syn_mean_MCI, SUVR_med_temp_syn_mean_NC]
    SUVR_med_temp_error_mean = [SUVR_med_temp_error_mean_AD, SUVR_med_temp_error_mean_MCI, SUVR_med_temp_error_mean_NC]
    SUVR_med_temp_real_std = [SUVR_med_temp_real_std_AD, SUVR_med_temp_real_std_MCI, SUVR_med_temp_real_std_NC]
    SUVR_med_temp_syn_std = [SUVR_med_temp_syn_std_AD, SUVR_med_temp_syn_std_MCI, SUVR_med_temp_syn_std_NC]
    SUVR_med_temp_error_std = [SUVR_med_temp_error_std_AD, SUVR_med_temp_error_std_MCI, SUVR_med_temp_error_std_NC]
    SUVR_pari_real_mean = [SUVR_pari_real_mean_AD, SUVR_pari_real_mean_MCI, SUVR_pari_real_mean_NC]
    SUVR_pari_syn_mean = [SUVR_pari_syn_mean_AD, SUVR_pari_syn_mean_MCI, SUVR_pari_syn_mean_NC]
    SUVR_pari_error_mean = [SUVR_pari_error_mean_AD, SUVR_pari_error_mean_MCI, SUVR_pari_error_mean_NC]
    SUVR_pari_real_std = [SUVR_pari_real_std_AD, SUVR_pari_real_std_MCI, SUVR_pari_real_std_NC]
    SUVR_pari_syn_std = [SUVR_pari_syn_std_AD, SUVR_pari_syn_std_MCI, SUVR_pari_syn_std_NC]
    SUVR_pari_error_std = [SUVR_pari_error_std_AD, SUVR_pari_error_std_MCI, SUVR_pari_error_std_NC]
    SUVR_ant_cing_real_mean = [SUVR_ant_cing_real_mean_AD, SUVR_ant_cing_real_mean_MCI, SUVR_ant_cing_real_mean_NC]
    SUVR_ant_cing_syn_mean = [SUVR_ant_cing_syn_mean_AD, SUVR_ant_cing_syn_mean_MCI, SUVR_ant_cing_syn_mean_NC]
    SUVR_ant_cing_error_mean = [SUVR_ant_cing_error_mean_AD, SUVR_ant_cing_error_mean_MCI, SUVR_ant_cing_error_mean_NC]
    SUVR_ant_cing_real_std = [SUVR_ant_cing_real_std_AD, SUVR_ant_cing_real_std_MCI, SUVR_ant_cing_real_std_NC]
    SUVR_ant_cing_syn_std = [SUVR_ant_cing_syn_std_AD, SUVR_ant_cing_syn_std_MCI, SUVR_ant_cing_syn_std_NC]
    SUVR_ant_cing_error_std = [SUVR_ant_cing_error_std_AD, SUVR_ant_cing_error_std_MCI, SUVR_ant_cing_error_std_NC]
    SUVR_pos_cing_real_mean = [SUVR_pos_cing_real_mean_AD, SUVR_pos_cing_real_mean_MCI, SUVR_pos_cing_real_mean_NC]
    SUVR_pos_cing_syn_mean = [SUVR_pos_cing_syn_mean_AD, SUVR_pos_cing_syn_mean_MCI, SUVR_pos_cing_syn_mean_NC]
    SUVR_pos_cing_error_mean = [SUVR_pos_cing_error_mean_AD, SUVR_pos_cing_error_mean_MCI, SUVR_pos_cing_error_mean_NC]
    SUVR_pos_cing_real_std = [SUVR_pos_cing_real_std_AD, SUVR_pos_cing_real_std_MCI, SUVR_pos_cing_real_std_NC]
    SUVR_pos_cing_syn_std = [SUVR_pos_cing_syn_std_AD, SUVR_pos_cing_syn_std_MCI, SUVR_pos_cing_syn_std_NC]
    SUVR_pos_cing_error_std = [SUVR_pos_cing_error_std_AD, SUVR_pos_cing_error_std_MCI, SUVR_pos_cing_error_std_NC]
    SUVR_composite_real_mean = [SUVR_composite_real_mean_AD, SUVR_composite_real_mean_MCI, SUVR_composite_real_mean_NC]
    SUVR_composite_syn_mean = [SUVR_composite_syn_mean_AD, SUVR_composite_syn_mean_MCI, SUVR_composite_syn_mean_NC]
    SUVR_composite_error_mean = [SUVR_composite_error_mean_AD, SUVR_composite_error_mean_MCI, SUVR_composite_error_mean_NC]
    SUVR_composite_real_std = [SUVR_composite_real_std_AD, SUVR_composite_real_std_MCI, SUVR_composite_real_std_NC]
    SUVR_composite_syn_std = [SUVR_composite_syn_std_AD, SUVR_composite_syn_std_MCI, SUVR_composite_syn_std_NC]
    SUVR_composite_error_std = [SUVR_composite_error_std_AD, SUVR_composite_error_std_MCI, SUVR_composite_error_std_NC]
    #SUVR_composite_real_mean = [SUVR_composite_real_mean_AD, SUVR_composite_real_mean_NC]
    #SUVR_composite_syn_mean = [SUVR_composite_syn_mean_AD, SUVR_composite_syn_mean_NC]
    #SUVR_composite_error_mean = [SUVR_composite_error_mean_AD, SUVR_composite_error_mean_NC]
    #SUVR_composite_real_std = [SUVR_composite_real_std_AD, SUVR_composite_real_std_NC]
    #SUVR_composite_syn_std = [SUVR_composite_syn_std_AD, SUVR_composite_syn_std_NC]
    #SUVR_composite_error_std = [SUVR_composite_error_std_AD, SUVR_composite_error_std_NC]
    fig1, ax1 = plt.subplots(3, 2, figsize=(15, 10))
    width = 0.1
    ax1[0, 0].bar(x_pos - width, height=SUVR_front_real_mean, width=2 * width, yerr=SUVR_front_real_std, color='r',
                  label='Real', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[0, 0].bar(x_pos + width, height=SUVR_front_syn_mean, width=2 * width, yerr=SUVR_front_syn_std, color='g',
                  label='Synthetic', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[0, 0].set_title('Frontal Cortex', fontsize=16)
    ax1[0, 0].set_ylabel('Normalized SUVR', fontsize=12)
    ax1[0, 0].set_xticks(x_pos)
    ax1[0, 0].set_xticklabels(clinical_cat)
    ax1[0, 0].yaxis.grid(True)
    ax1[0, 0].legend()
    ax1[0, 1].bar(x_pos - width, height=SUVR_pari_real_mean, width=2 * width, yerr=SUVR_pari_real_std, color='r',
                  label='Real', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[0, 1].bar(x_pos + width, height=SUVR_pari_syn_mean, width=2 * width, yerr=SUVR_pari_syn_std, color='g',
                  label='Synthetic', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[0, 1].set_title('Parietal Cortex', fontsize=16)
    ax1[0, 1].set_ylabel('Normalized SUVR', fontsize=12)
    ax1[0, 1].set_xticks(x_pos)
    ax1[0, 1].set_xticklabels(clinical_cat)
    ax1[0, 1].yaxis.grid(True)
    ax1[0, 1].legend()
    ax1[1, 0].bar(x_pos - width, height=SUVR_lat_temp_real_mean, width=2 * width, yerr=SUVR_lat_temp_real_std,
                  label='Real', color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[1, 0].bar(x_pos + width, height=SUVR_lat_temp_syn_mean, width=2 * width, yerr=SUVR_lat_temp_syn_std,
                  label='Synthetic', color='g', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[1, 0].set_title('Lateral Temporal Cortex', fontsize=16)
    ax1[1, 0].set_ylabel('Normalized SUVR', fontsize=12)
    ax1[1, 0].set_xticks(x_pos)
    ax1[1, 0].set_xticklabels(clinical_cat)
    ax1[1, 0].yaxis.grid(True)
    ax1[1, 0].legend()
    ax1[1, 1].bar(x_pos - width, height=SUVR_med_temp_real_mean, width=2 * width, yerr=SUVR_med_temp_real_std,
                  label='Real', color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[1, 1].bar(x_pos + width, height=SUVR_med_temp_syn_mean, width=2 * width, yerr=SUVR_med_temp_syn_std,
                  label='Synthetic', color='g', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[1, 1].set_title('Medial Temporal Cortex', fontsize=16)
    ax1[1, 1].set_ylabel('Normalized SUVR', fontsize=12)
    ax1[1, 1].set_xticks(x_pos)
    ax1[1, 1].set_xticklabels(clinical_cat)
    ax1[1, 1].yaxis.grid(True)
    ax1[1, 1].legend()

    ax1[2, 0].bar(x_pos - width, height=SUVR_ant_cing_real_mean, width=2 * width, yerr=SUVR_ant_cing_real_std,
                  label='Real', color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[2, 0].bar(x_pos + width, height=SUVR_ant_cing_syn_mean, width=2 * width, yerr=SUVR_ant_cing_syn_std,
                  label='Synthetic', color='g', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[2, 0].set_title('Anterior Cingulate Cortex', fontsize=16)
    ax1[2, 0].set_ylabel('Normalized SUVR', fontsize=12)
    ax1[2, 0].set_xticks(x_pos)
    ax1[2, 0].set_xticklabels(clinical_cat)
    ax1[2, 0].yaxis.grid(True)
    ax1[2, 0].legend()

    ax1[2, 1].bar(x_pos - width, height=SUVR_pos_cing_real_mean, width=2 * width, yerr=SUVR_pos_cing_real_std,
                  label='Real', color='r', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[2, 1].bar(x_pos + width, height=SUVR_pos_cing_syn_mean, width=2 * width, yerr=SUVR_pos_cing_syn_std,
                  label='Synthetic', color='g', align='center', alpha=0.5, ecolor='black', capsize=5)
    ax1[2, 1].set_title('Posterior Cingulate Cortex', fontsize=16)
    ax1[2, 1].set_ylabel('Normalized SUVR', fontsize=12)
    ax1[2, 1].set_xticks(x_pos)
    ax1[2, 1].set_xticklabels(clinical_cat)
    ax1[2, 1].yaxis.grid(True)
    ax1[2, 1].legend()

    # Save the figure and show
    plt.tight_layout()
    fig_path = os.path.join(root_dir, 'SUVR_roi_group_comparison.png')
    plt.savefig(fig_path)
    # plt.show()

    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    figure(figsize=(15, 10))
    ax2 = plt.gca()
    ax2.bar(x_pos - width, height=SUVR_composite_real_mean, width=2 * width, yerr=SUVR_composite_real_std, label='Real',
            color='r', align='center', alpha=0.5, ecolor='black', capsize=12)
    ax2.bar(x_pos + width, height=SUVR_composite_syn_mean, width=2 * width, yerr=SUVR_composite_syn_std, label='Synthetic',
            color='g', align='center', alpha=0.5, ecolor='black', capsize=20)
    ax2.set_title('Composite Cortex', fontsize=40, weight='bold')
    ax2.set_ylabel('Normalized SUVR', fontsize=36, weight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(clinical_cat, fontsize=36, weight='bold')
    ax2.tick_params(axis='y', labelsize=32)
    ax2.yaxis.grid(True)
    ax2.legend(fontsize=30)

    # Save the figure and show
    plt.tight_layout()
    fig_path = os.path.join(root_dir, 'SUVR_comp_group_comparison.png')
    plt.savefig(fig_path)
    # plt.show()

    fig3, ax3 = plt.subplots(3, 2, figsize=(15, 10))
    width = 0.2
    ax3[0, 0].bar(x_pos, height=SUVR_front_error_mean, width=2 * width, yerr=SUVR_front_error_std, color='b',
                  align='center', alpha=0.5, ecolor='black', capsize=5)
    ax3[0, 0].set_title('Frontal Cortex', fontsize=16)
    ax3[0, 0].set_ylabel('Error Percentage', fontsize=12)
    ax3[0, 0].set_xticks(x_pos)
    ax3[0, 0].set_xticklabels(clinical_cat)
    ax3[0, 0].yaxis.grid(True)
    ax3[0, 0].set_ylim([0, 30])
    ax3[0, 1].bar(x_pos, height=SUVR_pari_error_mean, width=2 * width, yerr=SUVR_pari_error_std, color='b',
                  align='center', alpha=0.5,
                  ecolor='black', capsize=5)
    ax3[0, 1].set_title('Parietal Cortex', fontsize=16)
    ax3[0, 1].set_ylabel('Error Percentage', fontsize=12)
    ax3[0, 1].set_xticks(x_pos)
    ax3[0, 1].set_xticklabels(clinical_cat)
    ax3[0, 1].set_ylim([0, 30])
    ax3[0, 1].yaxis.grid(True)
    ax3[1, 0].bar(x_pos, height=SUVR_lat_temp_error_mean, width=2 * width, yerr=SUVR_lat_temp_error_std, color='b',
                  align='center',
                  alpha=0.5, ecolor='black', capsize=5)
    ax3[1, 0].set_title('Lateral Temporal Cortex', fontsize=16)
    ax3[1, 0].set_ylabel('Error Percentage', fontsize=12)
    ax3[1, 0].set_xticks(x_pos)
    ax3[1, 0].set_xticklabels(clinical_cat)
    ax3[1, 0].set_ylim([0, 30])
    ax3[1, 0].yaxis.grid(True)
    ax3[1, 1].bar(x_pos, height=SUVR_med_temp_error_mean, width=2 * width, yerr=SUVR_med_temp_error_std, color='b',
                  align='center',
                  alpha=0.5, ecolor='black', capsize=5)
    ax3[1, 1].set_title('Medial Temporal Cortex', fontsize=16)
    ax3[1, 1].set_ylabel('Error Percentage', fontsize=12)
    ax3[1, 1].set_xticks(x_pos)
    ax3[1, 1].set_xticklabels(clinical_cat)
    ax3[1, 1].set_ylim([0, 30])
    ax3[1, 1].yaxis.grid(True)

    ax3[2, 0].bar(x_pos, height=SUVR_ant_cing_error_mean, width=2 * width, yerr=SUVR_ant_cing_error_std, color='b',
                  align='center',
                  alpha=0.5, ecolor='black', capsize=5)
    ax3[2, 0].set_title('Anterior Cingulate Cortex', fontsize=16)
    ax3[2, 0].set_ylabel('Error Percentage', fontsize=12)
    ax3[2, 0].set_xticks(x_pos)
    ax3[2, 0].set_xticklabels(clinical_cat)
    ax3[2, 0].set_ylim([0, 30])
    ax3[2, 0].yaxis.grid(True)

    ax3[2, 1].bar(x_pos, height=SUVR_pos_cing_error_mean, width=2 * width, yerr=SUVR_pos_cing_error_std, color='b',
                  align='center',
                  alpha=0.5, ecolor='black', capsize=5)
    ax3[2, 1].set_title('Posterior Cingulate Cortex', fontsize=16)
    ax3[2, 1].set_ylabel('Error Percentage', fontsize=12)
    ax3[2, 1].set_xticks(x_pos)
    ax3[2, 1].set_xticklabels(clinical_cat)
    ax3[2, 1].set_ylim([0, 30])
    ax3[2, 1].yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    fig_path = os.path.join(root_dir, 'SUVR_roi_error.png')
    plt.savefig(fig_path)
    # plt.show()

    figure(figsize=(15, 10))
    ax4 = plt.gca()
    ax4.bar(x_pos, height=SUVR_composite_error_mean, width=2 * width, yerr=SUVR_composite_error_std, color='0.5',
            align='center', alpha=0.5, ecolor='black', capsize=20)
    ax4.set_title('Composite Cortex', fontsize=40, weight='bold')
    ax4.set_ylabel('% Error', fontsize=36, weight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(clinical_cat, fontsize=36, weight='bold')
    ax4.tick_params(axis='y', labelsize=32)
    ax4.set_ylim([0, 50])
    ax4.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    fig_path = os.path.join(root_dir, 'SUVR_comp_error.png')
    plt.savefig(fig_path)
    # plt.show()

    fig5, ax5 = plt.subplots(3, 2, figsize=(10, 10))
    ax5[0, 0].plot(SUVR_front_real_AD, SUVR_front_syn_AD, marker='o', color='r', linestyle='', markersize=4, label='AD')
    ax5[0, 0].plot(SUVR_front_real_MCI, SUVR_front_syn_MCI, marker='o', color='y', linestyle='', markersize=4,
                   label='MCI')
    ax5[0, 0].plot(SUVR_front_real_NC, SUVR_front_syn_NC, marker='o', color='g', linestyle='', markersize=4, label='NC')
    ax5[0, 0].axline((0, 0), (1, 1), linestyle='--', linewidth=2, color='k')
    ax5[0, 0].set_title('Frontal Cortex', fontsize=16)
    ax5[0, 0].set_xlabel('Real Normalized SUVR', fontsize=12)
    ax5[0, 0].set_ylabel('Synthetic Normalized SUVR', fontsize=12)
    ax5[0, 0].set_xlim([0.4, 0.8])
    ax5[0, 0].set_ylim([0.4, 0.8])
    corr, p = pearsonr(SUVRs_front_real, SUVRs_front_syn)
    ax5[0, 0].annotate("r = {:.2f}, p = {:.2f}".format(corr, p), (0.55, 0.75))
    ax5[0, 0].legend()

    ax5[0, 1].plot(SUVR_pari_real_AD, SUVR_pari_syn_AD, marker='o', color='r', linestyle='', markersize=4, label='AD')
    ax5[0, 1].plot(SUVR_pari_real_MCI, SUVR_pari_syn_MCI, marker='o', color='y', linestyle='', markersize=4,
                   label='MCI')
    ax5[0, 1].plot(SUVR_pari_real_NC, SUVR_pari_syn_NC, marker='o', color='g', linestyle='', markersize=4, label='NC')
    ax5[0, 1].axline((0, 0), (1, 1), linestyle='--', linewidth=2, color='k')
    ax5[0, 1].set_title('Parietal Cortex', fontsize=16)
    ax5[0, 1].set_xlabel('Real Normalized SUVR', fontsize=12)
    ax5[0, 1].set_ylabel('Synthetic Normalized SUVR', fontsize=12)
    ax5[0, 1].set_xlim([0.4, 0.8])
    ax5[0, 1].set_ylim([0.4, 0.8])
    corr, p = pearsonr(SUVRs_pari_real, SUVRs_pari_syn)
    ax5[0, 1].annotate("r = {:.2f}, p = {:.2f}".format(corr, p), (0.55, 0.75))
    ax5[0, 1].legend()

    ax5[1, 0].plot(SUVR_lat_temp_real_AD, SUVR_lat_temp_syn_AD, marker='o', color='r', linestyle='', markersize=4,
                   label='AD')
    ax5[1, 0].plot(SUVR_lat_temp_real_MCI, SUVR_lat_temp_syn_MCI, marker='o', color='y', linestyle='', markersize=4,
                   label='MCI')
    ax5[1, 0].plot(SUVR_lat_temp_real_NC, SUVR_lat_temp_syn_NC, marker='o', color='g', linestyle='', markersize=4,
                   label='NC')
    ax5[1, 0].axline((0, 0), (1, 1), linestyle='--', linewidth=2, color='k')
    ax5[1, 0].set_title('Lateral Temporal Cortex', fontsize=16)
    ax5[1, 0].set_xlabel('Real Normalized SUVR', fontsize=12)
    ax5[1, 0].set_ylabel('Synthetic Normalized SUVR', fontsize=12)
    ax5[1, 0].set_xlim([0.4, 0.8])
    ax5[1, 0].set_ylim([0.4, 0.8])
    corr, p = pearsonr(SUVRs_lat_temp_real, SUVRs_lat_temp_syn)
    ax5[1, 0].annotate("r = {:.2f}, p = {:.2f}".format(corr, p), (0.55, 0.75))
    ax5[1, 0].legend()

    ax5[1, 1].plot(SUVR_med_temp_real_AD, SUVR_med_temp_syn_AD, marker='o', color='r', linestyle='', markersize=4,
                   label='AD')
    ax5[1, 1].plot(SUVR_med_temp_real_MCI, SUVR_med_temp_syn_MCI, marker='o', color='y', linestyle='', markersize=4,
                   label='MCI')
    ax5[1, 1].plot(SUVR_med_temp_real_NC, SUVR_med_temp_syn_NC, marker='o', color='g', linestyle='', markersize=4,
                   label='NC')
    ax5[1, 1].axline((0, 0), (1, 1), linestyle='--', linewidth=2, color='k')
    ax5[1, 1].set_title('Medial Temporal Cortex', fontsize=16)
    ax5[1, 1].set_xlabel('Real Normalized SUVR', fontsize=12)
    ax5[1, 1].set_ylabel('Synthetic Normalized SUVR', fontsize=12)
    ax5[1, 1].set_xlim([0.4, 0.8])
    ax5[1, 1].set_ylim([0.4, 0.8])
    corr, p = pearsonr(SUVRs_med_temp_real, SUVRs_med_temp_syn)
    ax5[1, 1].annotate("r = {:.2f}, p = {:.2f}".format(corr, p), (0.5, 0.75))
    ax5[1, 1].legend()

    ax5[2, 0].plot(SUVR_ant_cing_real_AD, SUVR_ant_cing_syn_AD, marker='o', color='r', linestyle='', markersize=4,
                   label='AD')
    ax5[2, 0].plot(SUVR_ant_cing_real_MCI, SUVR_ant_cing_syn_MCI, marker='o', color='y', linestyle='', markersize=4,
                   label='MCI')
    ax5[2, 0].plot(SUVR_ant_cing_real_NC, SUVR_ant_cing_syn_NC, marker='o', color='g', linestyle='', markersize=4,
                   label='NC')
    ax5[2, 0].axline((0, 0), (1, 1), linestyle='--', linewidth=2, color='k')
    ax5[2, 0].set_title('Anterior Cingulate Cortex', fontsize=16)
    ax5[2, 0].set_xlabel('Real Normalized SUVR', fontsize=12)
    ax5[2, 0].set_ylabel('Synthetic Normalized SUVR', fontsize=12)
    ax5[2, 0].set_xlim([0.4, 0.8])
    ax5[2, 0].set_ylim([0.4, 0.8])
    corr, p = pearsonr(SUVRs_ant_cing_real, SUVRs_ant_cing_syn)
    ax5[2, 0].annotate("r = {:.2f}, p = {:.2f}".format(corr, p), (0.5, 0.75))
    ax5[2, 0].legend()

    ax5[2, 1].plot(SUVR_pos_cing_real_AD, SUVR_pos_cing_syn_AD, marker='o', color='r', linestyle='', markersize=4,
                   label='AD')
    ax5[2, 1].plot(SUVR_pos_cing_real_MCI, SUVR_pos_cing_syn_MCI, marker='o', color='y', linestyle='', markersize=4,
                   label='MCI')
    ax5[2, 1].plot(SUVR_pos_cing_real_NC, SUVR_pos_cing_syn_NC, marker='o', color='g', linestyle='', markersize=4,
                   label='NC')
    ax5[2, 1].axline((0, 0), (1, 1), linestyle='--', linewidth=2, color='k')
    ax5[2, 1].set_title('Posterior Cingulate Cortex', fontsize=16)
    ax5[2, 1].set_xlabel('Real Normalized SUVR', fontsize=12)
    ax5[2, 1].set_ylabel('Synthetic Normalized SUVR', fontsize=12)
    ax5[2, 1].set_xlim([0.4, 0.8])
    ax5[2, 1].set_ylim([0.4, 0.8])
    corr, p = pearsonr(SUVRs_pos_cing_real, SUVRs_pos_cing_syn)
    ax5[2, 1].annotate("r = {:.2f}, p = {:.2f}".format(corr, p), (0.5, 0.75))
    ax5[2, 1].legend()

    # Save the figure and show
    plt.tight_layout()
    fig_path = os.path.join(root_dir, 'SUVR_roi_scatter.png')
    plt.savefig(fig_path)
    # plt.show()

    figure(figsize=(15, 10))
    ax6 = plt.gca()
    ax6.plot(SUVR_composite_real_AD, SUVR_composite_syn_AD, marker='o', color='r', linestyle='', markersize=4,
             label='AD')
    ax6.plot(SUVR_composite_real_MCI, SUVR_composite_syn_MCI, marker='o', color='y', linestyle='', markersize=4,
             label='MCI')
    ax6.plot(SUVR_composite_real_NC, SUVR_composite_syn_NC, marker='o', color='g', linestyle='', markersize=4,
             label='NC')
    ax6.axline((0, 0), (1, 1), linestyle='--', linewidth=2, color='k')
    ax6.set_title('Composite Cortex', fontsize=16)
    ax6.set_xlabel('Real Normalized SUVR', fontsize=12)
    ax6.set_ylabel('Synthetic Normalized SUVR', fontsize=12)
    ax6.set_xlim([0.4, 0.8])
    ax6.set_ylim([0.4, 0.8])
    SUVRs_composite_real = SUVR_composite_real_AD + SUVR_composite_real_MCI + SUVR_composite_real_NC
    SUVRs_composite_syn = SUVR_composite_syn_AD + SUVR_composite_syn_MCI + SUVR_composite_syn_NC
    corr, p = pearsonr(SUVRs_composite_real, SUVRs_composite_syn)
    ax6.annotate("r = {:.2f}, p = {:.2f}".format(corr, p), (0.5, 0.75))
    ax6.legend()

    # Save the figure and show
    plt.tight_layout()
    fig_path = os.path.join(root_dir, 'SUVR_comp_scatter.png')
    plt.savefig(fig_path)
    plt.show()

# COMMAND ----------

SUVR_validation()