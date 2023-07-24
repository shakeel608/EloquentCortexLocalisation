#Author: Dr. Shakeel Ahmad Sheikh
#Affiliation: PostDoctoral Research Scientist, CITEC, University of Bielefeld, Germany
#Date: June 1, 2023
#Description: This script describes the dataloader of the network for Eloquent Cortex Tumor Detection

import os
import sys
import tempfile
from glob import glob
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
import nibabel as nib
import numpy as np
from monai.data import  DataLoader
from dict2csv import list_of_dicts_to_csv

class ECDLoader():
    def __init__(self, num_samples, path, SEED):
        """
        num_saampls: Generate num_samples of 3D volumetric images
        """

        self.num_samples = num_samples
        self.path = path
        self.SEED = SEED

    def generate_3d_data(self, tempdir):
        """Synthetic Dataset"""
    
        print(f"generating synthetic data to {tempdir} (this may take a while)")

        for i in range(self.num_samples):
            # create a temporary directory and 40 random image, mask pairs
            im, seg = create_test_image_3d(240, 240, 155, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            print("Synthetic data generated,",im.shape, seg.shape)
            nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))

            n = nib.Nifti1Image(seg, np.eye(4))
            nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

        images = sorted(glob(os.path.join(tempdir, "img*.nii.gz")))
        segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))
        train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:int(self.num_samples*0.8)], segs[:int(self.num_samples*0.8)])]
        val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-int(self.num_samples*0.8):], segs[-int(self.num_samples*0.8):])]
        test_files = [{"img": img, "seg": seg} for img, seg in zip(images[-int(self.num_samples*0.8):], segs[-int(self.num_samples*0.8):])]

        return train_files, val_files, test_files


    def load_BraTsTumorData(self):
        """Loads BraTs 202 Tuor Dataset
        Contains T1, T2, T1CE, FLAIR Modalities
        T1 We are using only T1 moality for benchmarking studies"""

        print(f"Loadng BraTs202 Brain Tumor Dataset of 300 (Train) + 65 (Val) Patients\n")
        
        img_dir = os.path.join(os.path.join(self.path, "data"), "T1")
        label_dir = os.path.join(os.path.join(self.path, "label"), "seg")
        #print("get cwd",os.getcwd(), self.path, img_dir, label_dir)

        images = sorted(glob(os.path.join(img_dir, "BraTS*.nii.gz")))
        segs = sorted(glob(os.path.join(label_dir, "*seg.nii.gz")))

        #print(images[:10], "\n\n\n", segs[:10])

        """For Reproducibility and generating same Dataset Partitions"""
        #import random 
        #combined = list(zip(images, segs))
        #random.seed(self.SEED)
        #print("SEED",self.SEED)
        #random.shuffle(combined)
        #images, segs = zip(*combined)

        #print(images[:10], "\n\n\n", segs[:10])
        #exit(0)

        train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:int(len(segs)*0.8)], segs[:int(len(segs)*0.8)])]
        val_files = [{"img": img, "seg": seg} for img, seg in zip(images[int(len(segs)*0.8):int(len(segs)*0.9)], segs[int(len(segs)*0.8):int(len(segs)*0.9)])]
        test_files = [{"img": img, "seg": seg} for img, seg in zip(images[int(len(segs)*0.9):], segs[int(len(segs)*0.9):])]
        
        list_of_dicts_to_csv(test_files, 'test_files.csv')
        #print(test_files); 
        #exit(0)
        #print(len(train_files), len(val_files), len(test_files)); exit(0)
        #trainData, trainSeg = images[:int(len(segs)*0.8)], segs[:int(len(segs)*0.8)] 
        #valData, valSeg = images[int(len(segs)*0.8):], segs[int(len(segs)*0.8):] 
        #return trainData, trainSeg, valData, valSeg
        return train_files, val_files, test_files




