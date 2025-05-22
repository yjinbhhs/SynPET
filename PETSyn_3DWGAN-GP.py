# Databricks notebook source
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import numpy as np
from os import listdir
from os.path import join
from scipy import io as sio
import torch
import os
import random
import SimpleITK as sitk

def get_loader(dataset, dataroot, 
               aug, phase, phase_T1_path, phase_FLAIR_path, phase_PET_path,
               patch_size_train, n_patch_train, image_size, n_patch_test,
               norm_T1, norm_FLAIR, norm_PET,
               batch_size, num_workers):

    if dataset == 'PETSyn':
        dataset = PETSyn(dataroot,
                         aug, phase, phase_T1_path, phase_FLAIR_path, phase_PET_path,
                         patch_size_train, n_patch_train, image_size, n_patch_test,
                         norm_T1, norm_FLAIR, norm_PET)
    elif dataset == 'XXX':
        dataset = None

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(phase == 'train'),
                                  num_workers=num_workers)
    return data_loader


class PETSyn(data.Dataset):
    def __init__(self, dataroot,
                 aug, phase, phase_T1_path, phase_FLAIR_path, phase_PET_path,
                 patch_size_train, n_patch_train, image_size, n_patch_test,
                 norm_T1, norm_FLAIR, norm_PET):
        self.dataroot = dataroot
        self.phase = phase
        self.phase_T1_path = phase_T1_path
        self.phase_FLAIR_path = phase_FLAIR_path
        self.phase_PET_path = phase_PET_path
        self.aug = aug

        # load all images and patching
        if self.phase == 'train':
            self.patch_size = patch_size_train
            self.n_patch = n_patch_train
        elif self.phase == 'valid':
            self.patch_size = image_size
            self.n_patch = n_patch_test

        self.T1_all = []
        self.FLAIR_all = []
        self.PET_all = []
        
        self.T1_names = os.listdir(os.path.join(self.dataroot, self.phase_T1_path))
        self.T1_names.sort()
#        self.T1_names = self.T1_names[0:10]
        self.FLAIR_names = os.listdir(os.path.join(self.dataroot, self.phase_FLAIR_path))
        self.FLAIR_names.sort()
#        self.FLAIR_names = self.FLAIR_names[0:10]
        self.PET_names = os.listdir(os.path.join(self.dataroot, self.phase_PET_path))
        self.PET_names.sort()
#        self.PET_names = self.PET_names[0:10]
        
        self.num_images = len(self.T1_names)
        if phase == 'train':
            print('Number of training images: {}'.format(self.num_images))
        elif phase == 'valid':
            print('Number of test images: {}'.format(self.num_images))
        
        for f1, f2, f3 in zip(self.T1_names, self.FLAIR_names, self.PET_names):
            print('Patching: ' + str(f1))
            print('Patching: ' + str(f2))
            print('Patching: ' + str(f3))

            # create the random index for cropping patches
            image_name = os.path.join(self.dataroot, self.phase_T1_path, f1)
            image_sitk = sitk.ReadImage(image_name)
            X = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
            indexes = get_random_patch_indexes(data=X,
                                               patch_size=self.patch_size, num_patches=self.n_patch,
                                               padding='VALID')

            # use index to crop patches
            
            X_patches = get_patches_from_indexes(image=X, indexes=indexes,
                                                 patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.T1_all.append(X_patches)

            image_name = os.path.join(self.dataroot, self.phase_FLAIR_path, f2)
            image_sitk = sitk.ReadImage(image_name)
            X = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
            X_patches = get_patches_from_indexes(image=X, indexes=indexes,
                                                 patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.FLAIR_all.append(X_patches)

            image_name = os.path.join(self.dataroot, self.phase_PET_path, f3)
            image_sitk = sitk.ReadImage(image_name)
            X = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
            X_patches = get_patches_from_indexes(image=X, indexes=indexes,
                                                 patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]
            self.PET_all.append(X_patches)


        self.T1_all = (np.concatenate(self.T1_all, 0) / norm_T1 - 0.5) * 2
        self.FLAIR_all = (np.concatenate(self.FLAIR_all, 0) / norm_FLAIR - 0.5) * 2
        self.PET_all = (np.concatenate(self.PET_all, 0) / norm_PET - 0.5) * 2
 
        # calculate the final total number after augmentation
        self.l = len(self.T1_all)

    def __getitem__(self, index):
        T1_tmp = self.T1_all[index]
        FLAIR_tmp = self.FLAIR_all[index]
        PET_tmp = self.PET_all[index]
   
        # data augmentation for training
        if self.aug and self.phase == 'train':
            if random.randint(0, 1):
                T1_tmp = np.flip(T1_tmp, axis=1)
                FLAIR_tmp = np.flip(FLAIR_tmp, axis=1)
                PET_tmp = np.flip(PET_tmp, axis=1)

            if random.randint(0, 1):
                T1_tmp = np.flip(T1_tmp, axis=2)
                FLAIR_tmp = np.flip(FLAIR_tmp, axis=2)
                PET_tmp = np.flip(PET_tmp, axis=2)

            if random.randint(0, 1):
                T1_tmp = np.flip(T1_tmp, axis=3)
                FLAIR_tmp = np.flip(FLAIR_tmp, axis=3)
                PET_tmp = np.flip(PET_tmp, axis=3)

            if random.randint(0, 1):
                T1_tmp = np.rot90(T1_tmp, axes=(1, 2))
                FLAIR_tmp = np.rot90(FLAIR_tmp, axes=(1, 2))
                PET_tmp = np.rot90(PET_tmp, axes=(1, 2))

            if random.randint(0, 1):
                T1_tmp = np.rot90(T1_tmp, axes=(1, 3))
                FLAIR_tmp = np.rot90(FLAIR_tmp, axes=(1, 3))
                PET_tmp = np.rot90(PET_tmp, axes=(1, 3))

            if random.randint(0, 1):
                T1_tmp = np.rot90(T1_tmp, axes=(2, 3))
                FLAIR_tmp = np.rot90(FLAIR_tmp, axes=(2, 3))
                PET_tmp = np.rot90(PET_tmp, axes=(2, 3))


        x_T1 = torch.FloatTensor(T1_tmp.copy())
        x_FLAIR = torch.FloatTensor(FLAIR_tmp.copy())
        x_PET = torch.FloatTensor(PET_tmp.copy())
        
        if len(self.T1_names) == self.l and self.phase == 'valid':
            valid_PET_name = os.path.join(self.dataroot, self.phase_PET_path, self.PET_names[index])
            return x_T1, x_FLAIR, x_PET, valid_PET_name
        else:
            return x_T1, x_FLAIR, x_PET

    def __len__(self):
        """Return the number of images after augmentation."""
        return self.l
 

# COMMAND ----------

from __future__ import print_function

import argparse
import os.path
import numpy as np
import math
from os.path import join, isfile
import SimpleITK as sitk
from six.moves import xrange

"""
  getRandomPatchIndexs -- indices for random patches
  getOrderedPatchIndexes  -- indices for a regularly sampled grid
  getPatchesFromIndexes -- getPatchesFromIndexes
  getRandomPatches -- (RandomPatchIndexes -> PatchesFromIndexes)
  getOrderedPatches -- (OrderedPatchIndexes -> PatchesFromIndexes)
  imagePatchRecon -- Recon Image given patches and indices

  getbatch3d -- get batch of 3d patches
  getbatch   -- get batch of images -- if 3D call getbatch3d

"""

# -------------------------------------------------------------------------------
#
# Core Patch Code
#
# -------------------------------------------------------------------------------
# Get Indices
# -------------------------------------------------------------------------------


def get_random_patch_indexes(data, patch_size=None, num_patches=1, padding='VALID'):
    """Get data patch samples from a regularly sampled grid

    Create a list of indexes to access patches from the tensor of data, where each
    row of the index list is a list of array of indices.

    Returns:
      indexes: the index of the data denoting the top-left corner of the patch
    """
    if patch_size is None:
        patch_size = data.shape

    dims = len(data.shape)
    indexes = np.zeros((num_patches,dims), dtype=np.int32)

    data_limits = list(data.shape)

    if padding == 'VALID':
        for i in range(0, dims):
            data_limits[i] -= patch_size[i]

    for j in range(0, num_patches):
        for i in range(0, dims):
            if data_limits[i] == 0:
                indexes[j, i] = 0
            else:
                indexes[j, i] = np.random.randint(0, data_limits[i])

    return indexes


def get_ordered_patch_indexes(data, patch_size=None, stride=None, padding='VALID'):
    """Get image patch samples from a regularly sampled grid

    Create the

    Returns:
      indexes: the index of the image denoting the top-left corner of the patch
    """

    dims = len(data.shape)

    if patch_size is None:
        internal_patch_size = data.shape[:]
    else:
        internal_patch_size = patch_size[:]
    for i in range(len(internal_patch_size), dims):
        internal_patch_size += [data.shape[i]]

    # Handle the stride
    if stride is None:
        stride = internal_patch_size[:]
    for i in range(len(stride), dims):
        stride += [data.shape[i]]

    total_patches = 1
    idx_all = []

    for i in range(0,dims):
      max_i = data.shape[i]
      if padding == 'VALID':
          max_i -= internal_patch_size[i]
      if max_i < 1:
          max_i = 1

      idx_all += [slice(1,max_i+1,stride[i])]

    grid = np.mgrid[idx_all]
    grid_size = grid.size
    indexes = np.transpose(grid.reshape(dims,int(grid_size/dims)))

    # Make sure to use 0 indexing
    indexes -= 1

    return indexes


# -------------------------------------------------------------------------------
# Get Patches
# -------------------------------------------------------------------------------


def get_patches_from_indexes(image, indexes, patch_size=None, padding='VALID', dtype=None):
    """Get image patches from specific positions in the image.

    Returns:
      patches: the image patches as a 4D numpy array
      indexes: the indexes of the image denoting the top-left corner of the patch in the image
               (just pass through really)
    """

    tmp_patch_size = list(image.shape)
    if patch_size is not None:
        # Ensure the patch size is of full data dimensions
        for i in range(0,min(len(image.shape), len(patch_size))):
            if patch_size[i] > 0:
                tmp_patch_size[i] = patch_size[i]

    if dtype is None:
        dtype = image.dtype

    dims = len(image.shape)
    num_patches = indexes.shape[0]

    patches_shape = (num_patches,)
    for i in range(0,dims):
        patches_shape += (tmp_patch_size[i],)
    patches = np.zeros(patches_shape, dtype=dtype)

    if padding == 'SAME':
        pad_slice = ()
        for i in range(0, dims):
            pad_slice += ((0, tmp_patch_size[i]),)
        image = np.pad(image, pad_slice, 'reflect')

    for i in range(0,num_patches):
        # Build the tuple of slicing indexes
        idx = ()
        for j in range(0,dims):
            idx += (slice(indexes[i,j],indexes[i,j]+tmp_patch_size[j]),)

        patches[i,...] = image[idx]

    return patches.astype(dtype)


def get_random_patches(image, patch_size=None, num_patches=1, padding='VALID', dtype=None):
    """Get image patch samples from a regularly sampled grid

    Create the

    Returns:
      patches: the image patches as a 4D numpy array
      indexes: the index of the image denoting the top-left corner of the patch
    """

    tmp_patch_size = list(image.shape[:])
    if patch_size is not None:
        # Ensure the patch size is of full data dimensions
        for i in range(0, min(len(image.shape), len(patch_size))):
            tmp_patch_size[i] = min(image.shape[i], patch_size[i])

    indexes = get_random_patch_indexes(image, tmp_patch_size, num_patches=num_patches, padding=padding)
    patches = get_patches_from_indexes(image, indexes, patch_size=tmp_patch_size, padding=padding, dtype=dtype)
    return [patches, indexes]


def get_ordered_patches(image, patch_size=None, stride=[1, 1, 1], num_patches=0, padding='VALID', dtype=None):
    """Get image patch samples from a regularly sampled grid

    Create the

    Returns:
      patches: the image patches as a 4D numpy array
      indexes: the index of the image denoting the top-left corner of the patch
    """

    tmp_patch_size = list(image.shape[:])
    if patch_size is not None:
        # Ensure the patch size is of full data dimensions
        for i in range(0,min(len(image.shape), len(patch_size))):
            tmp_patch_size[i] = min(image.shape[i], patch_size[i])

    indexes = get_ordered_patch_indexes(image, tmp_patch_size, stride=stride, padding=padding)

    total_patches = indexes.shape[0]
    if num_patches > total_patches:
        num_patches = total_patches

    if num_patches > 0:
        indexes = indexes[0:num_patches,...]

    patches = get_patches_from_indexes(image, indexes, patch_size=tmp_patch_size, padding=padding, dtype=dtype)

    return [patches, indexes]


# -------------------------------------------------------------------------------
#
# Image Patch Reconstruction
#
# -------------------------------------------------------------------------------
def image_patch_smooth_recon(output_size, patches, indexes, dtype=None, sigma=0.0):
    if dtype == None:
      dtype = patches.dtype

    dims = len(output_size)
    patch_size = []
    for i in range(1, dims+1):
      patch_size += [patches.shape[i]]

    # Check that the patches match the output shape, squeeze if necessary
    if len(patches.shape)-1 > dims:
      patches = np.squeeze(patches,axis=len(patches.shape)-1)

    padded_size = ()
    for i in range(0,dims):
      padded_size += (output_size[i]+patch_size[i],)
    padded_image = np.zeros(padded_size, dtype=np.float32)
    sum_image = np.zeros(padded_image.shape, dtype=np.float32)

    # Setup the weight mask
    weight_mask=np.zeros(patch_size,dtype=np.float32)
    mask_slice = ()
    for i in range(0,dims):
      half_i = 0.5*(patch_size[i]-1)
      mask_slice += (slice(-half_i,half_i+1),)
    mask_grid = np.mgrid[mask_slice]

    for i in range(0,dims):
      sigma_i = sigma*patch_size[i]
      scalar = 1.0/(sigma_i*sigma_i)
      weight_mask += scalar*pow(mask_grid[i],2.0)
    weight_mask = np.exp(-0.5*weight_mask)
    weight_mask[weight_mask<1e-8] = 0

    for i in xrange(0,patches.shape[0]):
      # Build the tuple of slicing indexes
      idx = ()
      for j in range(0,dims):
        idx += (slice(indexes[i,j],indexes[i,j]+patch_size[j]),)

      padded_image[idx] += np.multiply(patches[i,...],weight_mask)
      sum_image[idx] += weight_mask

    # Make sure the denominator is good
    sum_image[sum_image<1e-8] = 1
    image = np.true_divide(padded_image, sum_image)

    # Prepare the output
    output_idx = ()
    for i in range(0,dims):
      output_idx += (slice(0,output_size[i]),)
    output = image[output_idx]
    return output.astype(dtype)


def image_patch_recon(output_size, patches, indexes, dtype=None, sigma=0.0):
    if not(dtype):
      dtype = patches.dtype

    if sigma>0:
       return image_patch_smooth_recon(output_size,patches,indexes,dtype,sigma)

    dims = len(output_size)
    patch_size = []
    for i in range(1,dims+1):
      patch_size += [patches.shape[i]]

    # Check that the patches match the output shape, squeeze if necessary
    if len(patches.shape) - 1 > dims:
        patches = np.squeeze(patches,axis=len(patches.shape)-1)

    padded_size = ()
    for i in range(0,dims):
      padded_size += (output_size[i]+patch_size[i],)
    padded_image = np.zeros(padded_size, dtype=patches.dtype)
    sum_image = np.zeros(padded_image.shape, dtype=np.float32)

    for i in xrange(0,patches.shape[0]):
      # Build the tuple of slicing indexes
      idx = ()
      for j in range(0,dims):
        idx += (slice(indexes[i,j],indexes[i,j]+patch_size[j]),)

      padded_image[idx] += patches[i,...]
      sum_image[idx] += 1

    # Make sure the denominator is good
    sum_image[sum_image<1] = 1
    image = np.true_divide(padded_image, sum_image);

    # Prepare the output
    output_idx = ()
    for i in range(0,dims):
      output_idx += (slice(0,output_size[i]),)
    output = image[output_idx]
    return output.astype(dtype)


def image_patch_smooth_recon_one_hot(output_size, patches, indexes, num_classes, dtype=None,threed=False,sigma=0.0):

  if not(dtype):
    dtype = patches.dtype

  dims = len(output_size)-1
  patch_size = [1]*dims
  for i in range(0,len(patches.shape)-1):
    patch_size[i] = patches.shape[i+1]

  # Check that the patches match the output shape, squeeze if necessary
  if len(patches.shape) > dims+1:
    patches = np.squeeze(patches,axis=-1)
  if len(patches.shape)-1 < dims:
    patches = np.expand_dims(patches,axis=-1)

  # Get the max indexes
  max_index = [0]*dims
  for i in range(0,len(indexes)):
    for j in range(0,dims):
      if max_index[j] < indexes[i,j]:
        max_index[j] = indexes[i,j]

  padded_size = ()
  for i in range(0,dims):
    padded_size += (max_index[i]+patch_size[i],)
  padded_size += (output_size[-1],)
  padded_image = np.zeros(padded_size, dtype=np.float32)
  sum_image = np.zeros(padded_image.shape, dtype=np.float32)

  # Setup the weight mask
  weight_mask=np.zeros(patch_size,dtype=np.float32)
  mask_slice = ()
  for i in range(0,dims):
    half_i = 0.5*(patch_size[i]-1)
    mask_slice += (slice(-half_i,half_i+1),)
  mask_grid = np.mgrid[mask_slice]

  for i in range(0,dims):
    sigma_i = sigma*patch_size[i]
    scalar = 1.0/(sigma_i*sigma_i)
    weight_mask += scalar*pow(mask_grid[i],2.0)
  weight_mask = np.exp(-0.5*weight_mask)
  weight_mask[weight_mask<1e-8] = 0
  weight_mask = np.repeat(np.expand_dims(weight_mask,len(weight_mask)+1),num_classes,axis=dims)

  for i in xrange(0,patches.shape[0]):
    # Build the tuple of slicing indexes
    idx = ()
    for j in range(0,dims):
      idx += (slice(indexes[i,j],indexes[i,j]+patch_size[j]),)
    idx += (slice(0,num_classes),)

    p_i = patches[i,...].astype(int)
    hot_i = np.eye(num_classes)[p_i]
    padded_image[idx] += np.multiply(hot_i,weight_mask)
    sum_image[idx] += weight_mask

  # Make sure the denominator is good
  sum_image[sum_image<1e-8] = 1
  image = np.true_divide(padded_image, sum_image)

  # Prepare the output 
  output_idx = ()
  for i in range(0,dims):
    output_idx += (slice(0,output_size[i]),)
  output = image[output_idx]
  return output.astype(np.float32)


def image_patch_recon_one_hot(output_size, patches, indexes, num_classes, dtype=None,threed=False,sigma=0.0):

  if sigma>0:
     return image_patch_smooth_recon_one_hot(output_size,patches,indexes,num_classes,dtype,threed,sigma)

  if not(dtype):
    dtype = patches.dtype

  # Need to subtract one dim for one-hot encoding
  dims = len(output_size)-1
  patch_size = [1]*dims
  for i in range(0,len(patches.shape)-1):
    patch_size[i] = patches.shape[i+1]

  # Check that the patches match the output shape, squeeze if necessary
  if len(patches.shape) > dims+1:
    patches = np.squeeze(patches,axis=-1)
  if len(patches.shape)-1 < dims:
    patches = np.expand_dims(patches,axis=-1)

  # Get the max indexes
  max_index = [0]*dims
  for i in range(0,len(indexes)):
    for j in range(0,dims):
      if max_index[j] < indexes[i,j]:
        max_index[j] = indexes[i,j]

  padded_size = ()
  for i in range(0,dims):
    padded_size += (max_index[i]+patch_size[i],)
  # Add another one-hot dim back on the end
  padded_size += (output_size[-1],)
  padded_image = np.zeros(padded_size, dtype=patches.dtype)
  sum_image = np.zeros(padded_image.shape, dtype=np.float32)

  for i in xrange(0,patches.shape[0]):
    # Build the tuple of slicing indexes
    idx = ()
    for j in range(0,dims):
      idx += (slice(indexes[i,j],indexes[i,j]+patch_size[j]),)
    idx += (slice(0,num_classes),)

    p_i = patches[i,...].astype(int)
    hot_i = np.eye(num_classes, dtype=int)[p_i]
    padded_image[idx] += hot_i
    sum_image[idx] += 1

  # Make sure the denominator is good
  sum_image[sum_image<1] = 1
  image = np.true_divide(padded_image, sum_image);

  # Prepare the output 
  output_idx = ()
  for i in range(0,dims):
    output_idx += (slice(0,output_size[i]),)
  output = image[output_idx]
  return output.astype(np.float32)


# -------------------------------------------------------------------------------
# Crop Image
# -------------------------------------------------------------------------------
def crop_image(image, offset=None):

  dims = len(image.shape)
  offset_size = []
  for i in range(0,dims):
    offset_size += [0]
  if offset is not None:
    for i in range(0,min(dims,len(offset))):
      offset_size[i] = offset[i]

  crop_slice = ()
  for i in range(0,dims):
    crop_slice += (slice(offset_size[i],image.shape[i]-offset_size[i]),)

  return image[crop_slice]

# -----------------------------------------------------------------------------------
#
# Main Function
#
# -----------------------------------------------------------------------------------
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description='Load an image for patch sampling.')
#    parser.add_argument('input', nargs=1, type=str, help='NIfTI image input file.')
#    parser.add_argument('output', nargs=1, type=str, help='NIfTI image patch output file.')
#    parser.add_argument('-n','--num_samples', type=int, help='number of image patch samples to extract', default=0)
#    parser.add_argument('-r','--random', help='Perform random patch sampling from the image', action='store_true')
#    parser.add_argument('-p','--patch_size', type=int, nargs='+', help='Set the patch size in voxels', default=[1])
#    parser.add_argument('-s','--stride', type=int, nargs='+', help='Set the patch stride in voxels', default=[1])
#    parser.add_argument('--sigma', type=float, help='Reconstruction weight mask smoothing parameter, default=0.0', default=0.0)
#    parser.add_argument('--recon', help='File name for to create a reconstructed image from the sampled patches')
   #args = parser.parse_args()
#    args = parser.parse_args(args=['/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data/trainT1/006_S_6234_2018-02-19_T1_to_oasis_cropped.nii.gz',
#                                   '/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data/006_S_6234_2018-02-19_T1_to_oasis_cropped_patch', 
#                                   '-n', '50', 
#                                   '-r',
#                                   '-p', '64', '64', '64', 
#                                   '-s', '2', '2', '2',
#                                   '--sigma', '0.0', 
#                                   '--recon', '/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data/006_S_6234_2018-02-19_T1_to_oasis_cropped_recon.nii.gz'])


#    if not os.path.isfile(args.input[0]):
#        raise ValueError('Failed to find the file: ' + f)
#    print('Loading file: %s' % args.input[0])
#    orig_nifti_image = sitk.ReadImage(args.input[0])
#    image = sitk.GetArrayFromImage(orig_nifti_image).astype(np.float32)
#    print('Loaded image with data of size: '+str(image.shape))

  # Get the data dimensionality
#    dims = len(image.shape)

#    patch_size = []
#    for i in range(0,dims):
#        patch_size += [1]
  # Set the patch size from the input 
#    print(args.patch_size)
#    for i in range(0,min(dims,len(args.patch_size))):
#        patch_size[i] = min(image.shape[i],args.patch_size[i])

#    print('Patch size = %s' %(patch_size,))
#    print('Random sampling = %r' % args.random)

#    if args.random:
#        [patches, indexes] = get_random_patches(image, patch_size, num_patches=args.num_samples, padding='VALID')
#    else:
#        stride = []
#        for i in range(0,dims):
#            stride += [1]
#        for i in range(0,min(dims,len(args.stride))):
#            stride[i] = max(1,args.stride[i])
#        print('Stride: '+str(stride))

#        [patches, indexes] = get_ordered_patches(image, patch_size, stride=stride, padding='VALID', num_patches=args.num_samples)

#    print('Patch sampling complete.')
#    print('Got %d patches from the image...' % patches.shape[0])


#    out_patches = np.zeros(patch_size + [patches.shape[0]], dtype=image.dtype)
#    for i in range(0,patches.shape[0]):
#        out_patches[..., i] = patches[i, ...]
#        output = sitk.GetImageFromArray(out_patches[...,i])
#        print('Saving the patch image out: %s' % args.output[0]+str(i)+'.nii.gz')
#        sitk.WriteImage(output, args.output[0]+str(i)+'.nii.gz')


#    if args.recon:
#        print('Saving reconstructed image out: %s' % args.recon)
#        print(image.shape)
#        print(patches.shape)
#        print(indexes.shape)
#        r_image = image_patch_recon(image.shape, patches, indexes, sigma=args.sigma)
#        output = sitk.GetImageFromArray(r_image)
#        output.CopyInformation(orig_nifti_image)
#        sitk.WriteImage(output, args.recon)


# COMMAND ----------

#petsyn_train_loader = get_loader('PETSyn', '/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data', 
#                                         False, 'train', 'trainT1', 'trainFLAIR', 'trainPET',
#                                         [64, 64, 64], 8, [110, 192, 192], 1,
#                                         1, 1, 1,
#                                         20, 0)
#train_T1, train_FLAIR, train_PET = next(iter(petsyn_train_loader))
#print(train_T1.shape)

#petsyn_valid_loader = get_loader('PETSyn', '/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data', 
#                                         False, 'valid', 'testT1', 'testFLAIR', 'testPET',
#                                         [64, 64, 64], 8, [110, 192, 192], 1,
#                                         1, 1, 1,
#                                         1, 0)
#for test_T1, test_FLAIR, test_PET, PET_names in petsyn_valid_loader:
#    print(PET_names)

#x_real_PET_sitk = sitk.ReadImage('/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data/testPET/006_S_6209_2018-02-27_PET_to_oasis_SUVR_cropped.nii.gz')
#print('x_real_PET_sitk', x_real_PET_sitk)
#X = sitk.GetArrayFromImage(x_real_PET_sitk).astype(np.float32)
#print('X shape', X.shape)
#print('X[50,100,98]', X[50,100,98])
#x_fake_PET_sitk = sitk.GetImageFromArray(X)
#sitk.WriteImage(x_fake_PET_sitk, '/dbfs/mnt/bdh_mlai_mnt/yjin2/test.nii.gz')
#x_fake_PET_sitk = sitk.ReadImage('/dbfs/mnt/bdh_mlai_mnt/yjin2/test.nii.gz')
#X = sitk.GetArrayFromImage(x_fake_PET_sitk).astype(np.float32)
#print('X shape', X.shape)
#print('X[50,100,98]', X[50,100,98])

# COMMAND ----------

import tensorflow as tf
import numpy as np
#import cv2 
from math import log10

eps = 1e-12


def tf_prec_recall(p_true, p_pred):
    intersect = tf.reduce_mean( p_true*p_pred )
    precision = intersect / ( tf.reduce_mean( p_true ) + eps )
    recall    = intersect / ( tf.reduce_mean( p_pred ) + eps )
    return precision, recall


def tf_dice_score(p_true,p_pred):
    intersect = tf.reduce_mean(p_true*p_pred, axis=[0,1,2])
    union = eps + tf.reduce_mean(p_pred, axis=[0,1,2]) + tf.reduce_mean(p_true,axis=[0,1,2])
    dice = (2.*intersect+eps)/union
    return dice


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def psnr(sr_image, gt_image):

    assert sr_image.size(0) == gt_image.size(0) 

    peak_signal = (gt_image.max() - gt_image.min()).item()

    mse = (sr_image - gt_image).pow(2).mean().item()

    return 10 * log10(peak_signal ** 2 / mse)


def mae(sr_image, gt_image):
    assert sr_image.shape[0] == gt_image.shape[0] 
    
    threshold = 0.01
    sr_image_roi = sr_image[sr_image > threshold]
    gt_image_roi = gt_image[sr_image > threshold]
    mae = np.mean(np.absolute(sr_image_roi - gt_image_roi))

    return mae

def mre(sr_image, gt_image):
    assert sr_image.shape[0] == gt_image.shape[0]
    
    threshold = 0.01
    sr_image_roi = sr_image[sr_image > threshold]
    gt_image_roi = gt_image[sr_image > threshold]
    mre = np.mean(np.absolute(sr_image_roi - gt_image_roi)/sr_image_roi)

# COMMAND ----------

import tensorflow as tf


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step)

# COMMAND ----------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

''' 
Generators 
'''


class Generator_DUSENET(nn.Module):
    """Generator network."""
    def __init__(self, input_dim=2, conv_dim=64, repeat_num=5):
        super(Generator_DUSENET, self).__init__()
        block_init = []
        block_init.append(nn.Conv3d(input_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        block_init.append(nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True))
        block_init.append(nn.ReLU(inplace=True))
        block_init.append(ChannelSpatialSELayer3D(num_channels=conv_dim))
        self.block_init = nn.Sequential(*block_init)
        curr_dim = conv_dim
        block_init_outdim = curr_dim

        # Down-sampling layers.
        block_down1 = []
        block_down1.append(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_down1.append(nn.InstanceNorm3d(curr_dim * 2, affine=True, track_running_stats=True))
        block_down1.append(nn.ReLU(inplace=True))
        block_down1.append(ChannelSpatialSELayer3D(num_channels=curr_dim * 2))
        self.block_down1 = nn.Sequential(*block_down1)
        curr_dim = curr_dim * 2
        block_down1_outdim = curr_dim

        block_down2 = []
        block_down2.append(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_down2.append(nn.InstanceNorm3d(curr_dim * 2, affine=True, track_running_stats=True))
        block_down2.append(nn.ReLU(inplace=True))
        block_down2.append(ChannelSpatialSELayer3D(num_channels=curr_dim * 2))
        self.block_down2 = nn.Sequential(*block_down2)
        curr_dim = curr_dim * 2
        block_down2_outdim = curr_dim

        block_down3 = []
        block_down3.append(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_down3.append(nn.InstanceNorm3d(curr_dim * 2, affine=True, track_running_stats=True))
        block_down3.append(nn.ReLU(inplace=True))
        block_down3.append(ChannelSpatialSELayer3D(num_channels=curr_dim * 2))
        self.block_down3 = nn.Sequential(*block_down3)
        curr_dim = curr_dim * 2
        block_down3_outdim = curr_dim

        # Bottleneck layers.
        block_bottle = []
        for i in range(repeat_num):
            block_bottle.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.block_bottle = nn.Sequential(*block_bottle)

        # Up-sampling layers.
        block_up1 = []
        block_up1.append(nn.ConvTranspose3d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_up1.append(nn.InstanceNorm3d(curr_dim // 2, affine=True, track_running_stats=True))
        block_up1.append(nn.ReLU(inplace=True))
        block_up1.append(ChannelSpatialSELayer3D(num_channels=curr_dim // 2))
        self.block_up1 = nn.Sequential(*block_up1)
        curr_dim = curr_dim // 2

        block_up2 = []
        block_up2.append(nn.ConvTranspose3d(curr_dim + block_down2_outdim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_up2.append(nn.InstanceNorm3d(curr_dim // 2, affine=True, track_running_stats=True))
        block_up2.append(nn.ReLU(inplace=True))
        block_up2.append(ChannelSpatialSELayer3D(num_channels=curr_dim // 2))
        self.block_up2 = nn.Sequential(*block_up2)
        curr_dim = curr_dim // 2

        block_up3 = []
        block_up3.append(nn.ConvTranspose3d(curr_dim + block_down1_outdim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_up3.append(nn.InstanceNorm3d(curr_dim // 2, affine=True, track_running_stats=True))
        block_up3.append(nn.ReLU(inplace=True))
        block_up3.append(ChannelSpatialSELayer3D(num_channels=curr_dim // 2))
        self.block_up3 = nn.Sequential(*block_up3)
        curr_dim = curr_dim // 2

        block_final = []
        block_final.append(nn.Conv3d(curr_dim + block_init_outdim, input_dim, kernel_size=7, stride=1, padding=3, bias=False))
        block_final.append(nn.Tanh())
        self.block_final = nn.Sequential(*block_final)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        x_init = self.block_init(x)

        x_down1 = self.block_down1(x_init)
        x_down2 = self.block_down2(x_down1)
        x_down3 = self.block_down3(x_down2)

        x_bottle = self.block_bottle(x_down3)

        x_up1 = self.block_up1(x_bottle)
        x_up2 = self.block_up2(torch.cat((x_up1, x_down2), 1))
        x_up3 = self.block_up3(torch.cat((x_up2, x_down1), 1))

        x_final = self.block_final(torch.cat((x_up3, x_init), 1))
        return x_final


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        output_tensor = (self.cSE(input_tensor) + self.sSE(input_tensor))/2
        return output_tensor

''' 
Discriminators
'''


class Discriminator_DC(nn.Module):
    """
    Discriminator network with PatchGAN

    """
    def __init__(self, input_dim=2, image_size=128, conv_dim=64, repeat_num=5):
        super(Discriminator_DC, self).__init__()
        layers = []
        layers.append(nn.Conv3d(input_dim, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv = nn.Conv3d(curr_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False)
              
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv(h)
        return out_src

# COMMAND ----------

#from model import Generator_DUSENET
#from model import Discriminator_DC
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import SimpleITK as sitk
#from util.util import *
import numpy as np
import os
import time
import datetime
from tqdm import tqdm
import csv
import tensorflow.image as tfimg
from torch.utils.tensorboard import SummaryWriter
#from tensorboard_logger import Logger


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, petsyn_train_loader,
                 petsyn_valid_loader,
                 config):
        """Initialize configurations."""

        self.config = config

        # Data loader.
        self.petsyn_train_loader = petsyn_train_loader
        self.petsyn_valid_loader = petsyn_valid_loader
 
        # Model configurations.
        self.patch_size = config.patch_size_train if config.mode == 'train' else config.patch_size_test

        self.g = 'DUSENET'
        self.d = 'DC'
        self.ch_num = 2

        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_rec = config.lambda_rec
        self.lambda_pair = config.lambda_pair
        self.lambda_gp = config.lambda_gp


        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.sample_num = 10

        # Test configurations.
        self.test_iters = config.test_iters
        self.image_size = config.image_size
        self.patch_size_test = config.patch_size_test

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.validate_step = config.validate_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()


    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['PETSyn']:
          
            if self.g == 'DUSENET':
                self.G = Generator_DUSENET(self.ch_num, self.g_conv_dim, self.g_repeat_num)

            if self.d == 'DC':
                self.D = Discriminator_DC(self.ch_num, self.patch_size[0], self.d_conv_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        
        dummy_input = torch.rand(1, 2, self.patch_size[0], self.patch_size[1], self.patch_size[2])
        G_input_names = ['Real MRI']
        G_output_names =  ['Synthetic PET']
        D_input_names = ['Synthetic PET']
        D_output_names = ['Real/Fake']
        G_onnx_name = os.path.join(self.model_save_dir, 'Generator_DUSENET.onnx')
        D_onnx_name = os.path.join(self.model_save_dir, 'Discriminator_DC.onnx')
        torch.onnx.export(self.G, dummy_input, G_onnx_name, verbose=True, input_names=G_input_names, output_names=G_output_names)
        torch.onnx.export(self.D, dummy_input, D_onnx_name, verbose=True, input_names=D_input_names, output_names=D_output_names)
        
#        with SummaryWriter(log_dir=self.log_dir, comment='Generator_DUSENET') as w:
#            w.add_graph(self.G, (dummy_input, ), verbose=True)
        
#        with SummaryWriter(log_dir=self.log_dir, comment='Discriminator_DC') as w:
#            w.add_graph(self.D, (dummy_input, ), verbose=True)
        
            
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            
        self.G.to(self.device)
        self.D.to(self.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        # D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        # self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'PETSyn':
            data_train_loader = self.petsyn_train_loader
            data_valid_loader = self.petsyn_valid_loader
 
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        
        # Initialize validation measures
        psnr_mean_set = []
        mae_mean_set = []
        ssim_mean_set = []
        valid_step_count = []
        g_loss_fake_set = []
        g_loss_rec_set = []
        g_loss_pair_set = []
        g_loss_set = []
        d_loss_real_set = []
        d_loss_fake_set = []
        d_loss_gp_set = []
        d_loss_set = []
        
        rng = np.random.default_rng(888)
        valid_sample_ind = rng.choice(len(data_valid_loader), size=self.sample_num, replace=False)
        valid_sample_ind.sort()
        
        # Initialize Tensorboard Logs
        loss = {}
                
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            data_train_iter = iter(data_train_loader)
            x_real_T1, x_real_FLAIR, x_real_PET = next(data_train_iter)
            x_real_MR = torch.cat([x_real_T1, x_real_FLAIR], dim=1)
            x_real_PET = torch.cat([x_real_PET, x_real_PET], dim=1)
            x_real_MR = x_real_MR.to(self.device)           # Input images.
            x_real_PET = x_real_PET.to(self.device)
            
                     
            if self.d == 'DC':
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                # Compute loss with real images.
                out_src = self.D(x_real_PET)
                d_loss_real = - torch.mean(out_src)

                # Compute loss with fake images.
                x_fake_PET = self.G(x_real_MR)
                out_src = self.D(x_fake_PET.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real_PET.size(0), self.ch_num, 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real_PET.data + (1 - alpha) * x_fake_PET.data).requires_grad_(True)
                out_src = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.

                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_gp'] = d_loss_gp.item()
                loss['D/loss'] = d_loss.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i + 1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake_PET = self.G(x_real_MR)
                    out_src = self.D(x_fake_PET)
                    g_loss_fake = - torch.mean(out_src)
 
                    # Target-to-original domain.
                    x_reconst = self.G(x_fake_PET)
                    g_loss_rec = torch.mean(torch.abs(x_real_MR - x_reconst))

                    # Target-target paired loss
                    g_loss_pair = torch.mean(torch.abs(x_fake_PET - x_real_PET))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_pair * g_loss_pair
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_pair'] = g_loss_pair.item()
                    loss['G/loss'] = g_loss.item()
 
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

  
            # =================================================================================== #
            #                                 5. Validation on test set                           #
            # =================================================================================== #
            if (i+1) % self.validate_step == 0:
                with torch.no_grad():
                    
                    val_bar = tqdm(data_valid_loader)

                    avg_psnr = AverageMeter()
                    avg_mae = AverageMeter()
                    avg_ssim = AverageMeter()
                    valid_sub_ind = 0
                    sample_path = os.path.join(self.sample_dir, 'step_{}'.format(i+1))
                    if not os.path.exists(sample_path):
                        os.makedirs(sample_path)
                    
                    for (x_real_T1_val, x_real_FLAIR_val, x_real_PET_val, x_PET_name_val) in val_bar:
                    # Prepare input images and target domain labels.
                        if np.isin(valid_sub_ind, valid_sample_ind):
                            
                            x_real_T1_val = np.squeeze(x_real_T1_val).numpy()
                            x_real_FLAIR_val = np.squeeze(x_real_FLAIR_val).numpy()
                            x_real_PET_val = self.denorm(x_real_PET_val)
                            x_real_PET_val = np.squeeze(x_real_PET_val).numpy()
                            patch_indexes = get_ordered_patch_indexes(data=x_real_T1_val, patch_size=self.patch_size_test, 
                                                                      stride=[1,1,1], padding='VALID')
                            x_real_T1_val_patches = get_patches_from_indexes(image=x_real_T1_val, indexes=patch_indexes,
                                                                             patch_size=self.patch_size_test, padding='VALID', dtype=None)
                            x_real_T1_val_patches = torch.from_numpy(x_real_T1_val_patches[:, np.newaxis, :, :, :])
                            x_real_FLAIR_val_patches = get_patches_from_indexes(image=x_real_FLAIR_val, indexes=patch_indexes,
                                                                                patch_size=self.patch_size_test, padding='VALID', dtype=None)
                            x_real_FLAIR_val_patches = torch.from_numpy(x_real_FLAIR_val_patches[:, np.newaxis, :, :, :])
                            x_real_PET_val_patches = get_patches_from_indexes(image=x_real_PET_val, indexes=patch_indexes,
                                                                              patch_size=self.patch_size_test, padding='VALID', dtype=None)
                            x_real_PET_val_patches = torch.from_numpy(x_real_PET_val_patches[:, np.newaxis, :, :, :])
                            x_real_MR_val_patches = torch.cat([x_real_T1_val_patches, x_real_FLAIR_val_patches], dim=1)
                            x_fake_PET_val_patches = torch.zeros(x_real_MR_val_patches.shape)
                            x_real_MR_val_patches = x_real_MR_val_patches.to(self.device)
                            x_fake_PET_val_patches = x_fake_PET_val_patches.to(self.device)
                        

                    # Translate images
                            #x_fake_PET_val_patches = self.G(x_real_MR_val_patches)
                            # Reduce GPU memory usage
                            for patch_num in range(x_real_MR_val_patches.shape[0]):
                                tmp_patch = x_real_MR_val_patches[patch_num, ...]
                                tmp_patch = tmp_patch[np.newaxis, :, :, :, :]
                                x_fake_PET_val_patches[patch_num, ...] = self.G(tmp_patch)
                            x_fake_PET_val_patches = torch.mean(x_fake_PET_val_patches, 1, True).cpu()
                            x_fake_PET_val_patches = self.denorm(x_fake_PET_val_patches)
                            x_fake_PET_val_patches = x_fake_PET_val_patches.numpy()
                            x_fake_PET_val_patches = np.squeeze(x_fake_PET_val_patches, axis=1)
                            x_fake_PET_val = image_patch_recon(x_real_PET_val.shape, x_fake_PET_val_patches, patch_indexes, sigma=0.0)
                        
                      
                    # Calculate metrics
                            #x_real_PET_val = torch.from_numpy(x_real_PET_val)
                            #x_fake_PET_val = torch.from_numpy(x_fake_PET_val)
                            mae_ = mae(x_fake_PET_val, x_real_PET_val)
                            psnr_ = tfimg.psnr(x_fake_PET_val, x_real_PET_val, max_val=1.0)
                            ssim_ = tfimg.ssim(x_fake_PET_val, x_real_PET_val, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
                            avg_psnr.update(psnr_)
                            avg_mae.update(mae_)
                            avg_ssim.update(ssim_)


                            # Save into nii for future analysis
                            x_real_PET_sitk = sitk.ReadImage(x_PET_name_val[0])
                            x_fake_PET_sitk = sitk.GetImageFromArray(x_fake_PET_val)
                            x_fake_PET_sitk.CopyInformation(x_real_PET_sitk)
                            x_tmp = x_PET_name_val[0].split('/')[-1]
                            x_tmp = x_tmp.split('.')[0]
                            x_fake_PET_name = x_tmp+'_syn_'+ 'step_{}_'.format(i+1)+'.nii.gz'
                            sitk.WriteImage(x_fake_PET_sitk, os.path.join(sample_path, x_fake_PET_name))
                        
                        valid_sub_ind += 1
                                        
                        message = 'PSNR: {:4f} '.format(avg_psnr.avg)
                        message += 'MAE: {:4f} '.format(avg_mae.avg)
                        message += 'SSIM: {:4f} '.format(avg_ssim.avg)
                        val_bar.set_description(desc=message)
                        
                    print('Saved fake images into {}...'.format(sample_path))
                    print('PSNR: {:4f} '.format(avg_psnr.avg))
                    print('MAE: {:4f} '.format(avg_mae.avg))
                    print('SSIM: {:4f} '.format(avg_ssim.avg))
                    psnr_mean_set.append(avg_psnr.avg)
                    mae_mean_set.append(avg_mae.avg)
                    ssim_mean_set.append(avg_ssim.avg)
                    g_loss_fake_set.append(loss['G/loss_fake'])
                    g_loss_rec_set.append(loss['G/loss_rec'])
                    g_loss_pair_set.append(loss['G/loss_pair'])
                    g_loss_set.append(loss['G/loss'])
                    d_loss_real_set.append(loss['D/loss_real'])
                    d_loss_fake_set.append(loss['D/loss_fake'])
                    d_loss_gp_set.append(loss['D/loss_gp'])
                    d_loss_set.append(loss['D/loss'])
                    
                    valid_step_count.append(i+1)
       
   
                    # save all validate metrics 
        train_valid_history = {
            'Step': valid_step_count,
            'PSNR': np.round(psnr_mean_set, 4),
            'MAE': np.round(mae_mean_set, 4),
            'SSIM': np.round(ssim_mean_set, 4),
            'G/loss_fake': np.round(g_loss_fake_set, 4),
            'G/loss_rec': np.round(g_loss_rec_set, 4),
            'G/loss_pair': np.round(g_loss_pair_set, 4),
            'G/loss': np.round(g_loss_set, 4),
            'D/loss_real': np.round(d_loss_real_set, 4),
            'D/loss_fake': np.round(d_loss_fake_set, 4),
            'D/loss': np.round(d_loss_set, 4)
            }
          
        with open(os.path.join(self.sample_dir, 'train_valid_metrics.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(train_valid_history.keys())
            writer.writerows(zip(*[train_valid_history[key] for key in train_valid_history.keys()]))
            print('Saved training validation metrics into {}...'.format(self.sample_dir))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'PETSyn':
            data_valid_loader = self.petsyn_valid_loader
        
        # Initialize validation measures
        avg_psnr = []
        avg_mae = []
        avg_ssim = []
        valid_PET_names = []
        
        with torch.no_grad():
            val_bar = tqdm(data_valid_loader)
  
            for (x_real_T1_val, x_real_FLAIR_val, x_real_PET_val, x_PET_name_val) in val_bar:
            # Prepare input images and target domain labels.
                x_real_T1_val = np.squeeze(x_real_T1_val).numpy()
                x_real_FLAIR_val = np.squeeze(x_real_FLAIR_val).numpy()
                x_real_PET_val = self.denorm(x_real_PET_val)
                x_real_PET_val = np.squeeze(x_real_PET_val).numpy()
                patch_indexes = get_ordered_patch_indexes(data=x_real_T1_val, patch_size=self.patch_size_test, 
                                                          stride=[1,1,1], padding='VALID')
                x_real_T1_val_patches = get_patches_from_indexes(image=x_real_T1_val, indexes=patch_indexes,
                                                                 patch_size=self.patch_size_test, padding='VALID', dtype=None)
                x_real_T1_val_patches = torch.from_numpy(x_real_T1_val_patches[:, np.newaxis, :, :, :])
                x_real_FLAIR_val_patches = get_patches_from_indexes(image=x_real_FLAIR_val, indexes=patch_indexes,
                                                                    patch_size=self.patch_size_test, padding='VALID', dtype=None)
                x_real_FLAIR_val_patches = torch.from_numpy(x_real_FLAIR_val_patches[:, np.newaxis, :, :, :])
                x_real_PET_val_patches = get_patches_from_indexes(image=x_real_PET_val, indexes=patch_indexes,
                                                                  patch_size=self.patch_size_test, padding='VALID', dtype=None)
                x_real_PET_val_patches =torch.from_numpy(x_real_PET_val_patches[:, np.newaxis, :, :, :])
                x_real_MR_val_patches = torch.cat([x_real_T1_val_patches, x_real_FLAIR_val_patches], dim=1)
                x_fake_PET_val_patches = torch.zeros(x_real_MR_val_patches.shape)
                x_real_MR_val_patches = x_real_MR_val_patches.to(self.device)
                x_fake_PET_val_patches = x_fake_PET_val_patches.to(self.device)
  
            # Translate images.
                #x_fake_PET_val_patches = self.G(x_real_MR_val_patches)
                for patch_num in range(x_real_MR_val_patches.shape[0]):
                    tmp_patch = x_real_MR_val_patches[patch_num, ...]
                    tmp_patch = tmp_patch[np.newaxis, :, :, :, :]
                    x_fake_PET_val_patches[patch_num, ...] = self.G(tmp_patch)
                x_fake_PET_val_patches = torch.mean(x_fake_PET_val_patches, 1, True).cpu()
                x_fake_PET_val_patches = self.denorm(x_fake_PET_val_patches)
                x_fake_PET_val_patches = x_fake_PET_val_patches.numpy()
                x_fake_PET_val_patches = np.squeeze(x_fake_PET_val_patches, axis=1)
                x_fake_PET_val = image_patch_recon(x_real_PET_val.shape, x_fake_PET_val_patches, patch_indexes, sigma=0.0)
              
            # Calculate metrics
                #x_real_PET_val = torch.from_numpy(x_real_PET_val)
                #x_fake_PET_val = torch.from_numpy(x_fake_PET_val)
                valid_mae = mae(x_fake_PET_val, x_real_PET_val)
                valid_psnr = tfimg.psnr(x_real_PET_val, x_fake_PET_val)
                valid_ssim = tfimg.ssim(x_fake_PET_val, x_real_PET_val, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
                avg_psnr.append(valid_psnr)
                avg_mae.append(valid_mae)
                avg_ssim.append(valid_ssim)
                
            # Save into nii for future analysis
                result_path = os.path.join(self.result_dir, 'images')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
               
                x_real_PET_sitk = sitk.ReadImage(x_PET_name_val[0])
                x_fake_PET_sitk = sitk.GetImageFromArray(x_fake_PET_val)
                x_fake_PET_sitk.CopyInformation(x_real_PET_sitk)
                x_tmp = x_PET_name_val[0].split('/')[-1]
                x_tmp = x_tmp.split('.')[0]
                x_fake_PET_name = x_tmp+'_syn.nii.gz'
                sitk.WriteImage(x_fake_PET_sitk, os.path.join(result_path, x_fake_PET_name))
                
                valid_PET_names.append(x_tmp)
                
                message = 'PSNR: {:4f} '.format(np.mean(avg_psnr))
                message += 'MAE: {:4f} '.format(np.mean(avg_mae))
                message += 'SSIM: {:4f} '.format(np.mean(avg_ssim))
            val_bar.set_description(desc=message)
            print('PSNR: {:4f} '.format(avg_psnr.avg))
            print('MAE: {:4f} '.format(avg_mae.avg))
            print('SSIM: {:4f} '.format(avg_ssim.avg))
            
            # Save all validate metrics 
            test_history = {
                'PET': valid_PET_names,
                'PSNR': np.round(avg_psnr, 4),
                'MAE': np.round(avg_mae, 4),
                'SSIM': np.round(avg_ssim, 4)
                }
          
            with open(os.path.join(self.result_dir, 'test_metrics.csv'), 'w') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(test_history.keys())
                writer.writerows(zip(*[test_history[key] for key in test_history.keys()]))
                print('Saved test metrics into {}...'.format(self.result_dir))


# COMMAND ----------

import os
import argparse
#from solver import Solver
#from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    config.log_dir = os.path.join(config.dataroot, 'outputs', config.name, config.log_dir)
    config.model_save_dir = os.path.join(config.dataroot, 'outputs', config.name, config.model_save_dir)
    config.sample_dir = os.path.join(config.dataroot, 'outputs', config.name, config.sample_dir)
    config.result_dir = os.path.join(config.dataroot, 'outputs', config.name, config.result_dir)
    config.train_T1_path = 'trainT1'
    config.train_FLAIR_path = 'trainFLAIR'
    config.train_PET_path = 'trainPET'
    config.vaild_T1_path = 'testT1'
    config.valid_FLAIR_path = 'testFLAIR'
    config.valid_PET_path = 'testPET'
    
    
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    if config.dataset in ['PETSyn']:
        petsyn_train_loader = get_loader(config.dataset, config.dataroot, 
                                         config.aug, 'train', config.train_T1_path, config.train_FLAIR_path, config.train_PET_path,
                                         config.patch_size_train, config.n_patch_train, config.image_size, config.n_patch_test,
                                         config.norm_T1, config.norm_FLAIR, config.norm_PET,
                                         config.batch_size, config.num_workers)
        petsyn_valid_loader = get_loader(config.dataset, config.dataroot,
                                        config.aug, 'valid', config.vaild_T1_path, config.valid_FLAIR_path, config.valid_PET_path,
                                        config.patch_size_train, config.n_patch_train, config.image_size, config.n_patch_test,
                                        config.norm_T1, config.norm_FLAIR, config.norm_PET,
                                        1, config.num_workers)
        
    # Solver for training and testing.
    solver = Solver(petsyn_train_loader, 
                    petsyn_valid_loader,
                    config)

    if config.mode == 'train':
        if config.dataset in ['PETSyn']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['PETSyn']:
            solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='xxx', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Model configuration.
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=3, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=4, help='number of strided conv layers in D')

    parser.add_argument('--lambda_rec', type=float, default=5, help='weight for reconstruction loss')
    parser.add_argument('--lambda_pair', type=float, default=15, help='weight for pair-wise reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')

 
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='PETSyn', choices=['PETSyn', 'XXX'])
    parser.add_argument('--dataroot', type=str, default='/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data', help='data root')
    parser.add_argument('--norm_T1', type=float, default=1, help='T1 normalization by dividing')
    parser.add_argument('--norm_FLAIR', type=float, default=1, help='FLAIR normalization by dividing')
    parser.add_argument('--norm_PET', type=float, default=1, help='PET normalization by dividing')

    parser.add_argument('--n_patch_train', type=int, default=8, help='# of patch cropped for each image')
    parser.add_argument('--patch_size_train', nargs='+', type=int, default=[64, 64, 64], help='patch size to crop')
    parser.add_argument('--aug', default=False, action='store_true', help='use augmentation')
    parser.add_argument('--batch_size', type=int, default=3, help='mini-batch size')

    parser.add_argument('--num_iters', type=int, default=2000000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=3, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--n_patch_test', type=int, default=1, help='# of patch cropped for each image')
    parser.add_argument('--patch_size_test', nargs='+', type=int, default=[80, 96, 80], help='test patch size to crop')
    parser.add_argument('--image_size', nargs='+', type=int, default=[110, 192, 192], help='image size')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_save_dir', type=str, default='models')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--result_dir', type=str, default='results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=500)
    parser.add_argument('--validate_step', type=int, default=500)

    #config = parser.parse_args()
    config = parser.parse_args(args=['--name', 'PETSyn-3DWGAN-GP', 
                                     '--mode', 'train',
                                     '--g_conv_dim', '64',
                                     '--d_conv_dim', '64',
                                     '--g_repeat_num', '3',
                                     '--d_repeat_num', '4',
                                     '--lambda_rec', '5',
                                     '--lambda_pair', '15',
                                     '--lambda_gp', '10',
                                     '--dataset', 'PETSyn',
                                     '--dataroot', '/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data',
                                     '--norm_T1', '1',
                                     '--norm_FLAIR', '1',
                                     '--norm_PET', '1',
                                     '--n_patch_train', '100',
                                     '--patch_size_train', '64', '64', '64',
                                     '--aug', 
                                     '--batch_size', '5',
                                     '--num_iters', '26800',
                                     '--num_iters_decay', '18760',
                                     '--g_lr', '0.0001',
                                     '--d_lr', '0.0001',
                                     '--n_critic', '3',
                                     '--beta1', '0.5',
                                     '--beta2', '0.999',
                                     '--resume_iters', False,
                                     '--test_iters', '26800',
                                     '--n_patch_test', '1',
                                     '--patch_size_test', '104', '192', '192',
                                     '--image_size', '110', '192', '192',
                                     '--num_workers', '0',
                                     '--use_tensorboard', 'True',
                                     '--log_dir', 'logs',
                                     '--model_save_dir', 'models',
                                     '--sample_dir', 'samples',
                                     '--result_dir', 'results',
                                     '--log_step', '10',
                                     '--lr_update_step', '18670',
                                     '--model_save_step', '268',
                                     '--validate_step', '1000'])
    print(config)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main(config)


# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC log_dir = '/dbfs/mnt/bdh_mlai_mnt/yjin2/PETSyn/gan_data/outputs/PETSyn-3DWGAN-GP/logs'
# MAGIC %tensorboard --logdir $log_dir