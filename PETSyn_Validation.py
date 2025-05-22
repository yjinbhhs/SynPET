# Databricks notebook source
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.image as tfimg
import tensorflow_datasets as tfds
import SimpleITK as sitk

img_num = 10

def PETSyn_validation():

    root_dir = '/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn'
    real_image_dir = 'gan_data/testPET'
    syn_image_dir = 'gan_data/outputs/PETSyn-3DWGAN-GP/results/images'

    real_image_path = os.path.join(root_dir, real_image_dir)
    real_image_list = os.listdir(real_image_path)
    real_image_list.sort()
    real_image_names = []
    for f in real_image_list[:img_num]:
        real_image_names.append(os.path.join(real_image_path, f))
    syn_image_path = os.path.join(root_dir, syn_image_dir)
    syn_image_list = os.listdir(syn_image_path)
    syn_image_list.sort()
    syn_image_names = []
    for f in syn_image_list[:img_num]:
        syn_image_names.append(os.path.join(syn_image_path, f))
   
    real_images_np = []
    syn_images_np = []
    mae_np = []
    mre_np = []
    threshold = 0.01
    for real_image, syn_image in zip(real_image_names, syn_image_names):
        real_image_sitk = sitk.ReadImage(real_image)
        real_image_np = sitk.GetArrayFromImage(real_image_sitk).astype(np.float32)
        real_images_np.append(real_image_np)
        syn_image_sitk = sitk.ReadImage(syn_image)
        syn_image_np = sitk.GetArrayFromImage(syn_image_sitk).astype(np.float32)
        syn_images_np.append(syn_image_np)
        real_image_roi = real_image_np[real_image_np > threshold]
        syn_image_roi = syn_image_np[real_image_np > threshold]
        mae = np.mean(np.absolute(np.subtract(real_image_roi, syn_image_roi)))
        mae_np.append(mae)
        mre = np.mean(np.absolute(np.subtract(real_image_roi, syn_image_roi))/(real_image_roi))
        mre_np.append(mre)


    real_images_np = np.array(real_images_np, dtype=np.float32)
    syn_images_np = np.array(syn_images_np, dtype=np.float32)

    mae_ds = pd.Series(mae_np)
    print('MAE \n', mae_ds)
    mre_ds = pd.Series(mre_np)
    print('MRE \n', mre_ds)
    psnr_tf = tfimg.psnr(real_images_np, syn_images_np, max_val=1.0)
    psnr_ds = pd.Series(psnr_tf)
    print('PSNR \n', psnr_ds)
    ssim_tf = tfimg.ssim(real_images_np, syn_images_np, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim_ds = pd.Series(ssim_tf)
    print('SSIM \n', ssim_ds)
#    real_image_names_ds = pd.Series(real_image_names)
#    syn_image_names_ds = pd.Series(syn_image_names)
#    d = {'real_image': real_image_names_ds, 'syn_image': syn_image_names_ds, 'MAE': mae_ds, 'MRE': mre_ds, 'PSNR': psnr_ds, 'SSIM': ssim_ds}
#    stats_df = pd.DataFrame(data=d)
#    print(stats_df)
#    stats_file_name = os.path.join(root_dir, syn_image_dir, 'stats.csv')
#    stats_df.to_csv(stats_file_name, index=False)

# COMMAND ----------

PETSyn_validation()