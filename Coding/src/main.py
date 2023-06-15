#Author: Dr. Shakeel Ahmad Sheikh
#Affiliation: PostDoctoral Research Scientist, CITEC, University of Bielefeld, Germany
#Date: June 1, 2023
#Description: This script describes the main function which begins training of the network for Eloquent Cortex Tumor Detection

import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    MapTransform,
    NormalizeIntensity
)
from monai.visualize import plot_2d_or_3d_image

#CustomCreated Classes
from unet_model import UNet3D
from dataloader import ECDLoader
from train import TrainWrapper

from load_data import ECDDataLoader

# Custom transform to apply NormalizeIntensity to a specific key
class CustomNormalizeIntensity(MapTransform):
    def __call__(self, data):
        image = data["img"]  # Get the image data
        #print("UnNormalised",image[0][:,:,23][:,145])
        normalized_image = NormalizeIntensity(nonzero=True)(image)  # Apply NormalizeIntensity
        data["img"] = normalized_image  # Update the data dictionary
        #print("Normalised",normalized_image[0][:,:,23][:,145])
        #exit(0)
        return data


def main(tmpdir):
    #print(tmpdir);exit(0)
    batch_size = 4
    num_samples = 40  #Number of samples to generate
    path = "/Users/dr.shakeel/PostDocWork/Datasets/MICCAI_BraTS2020_Data/"
    data_reader = ECDLoader(num_samples, path)

    syntheticTrain = True
    if syntheticTrain:
        train_files, val_files = data_reader.generate_3d_data(tmpdir)
    else:
        train_files, val_files = data_reader.load_BraTsTumorData()
    
    #trainData, trainSeg, valData, valSeg = data_reader.load_BraTsTumorData()
    #print("Len of Train and Val",len(trainData), len(valData))
    #train_dataset = ECDDataLoader(trainData, trainSeg)
    #val_dataset = ECDDataLoader(valData, valSeg)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    
    #for it, lab in train_loader:
    #    print (it.shape, lab.shape)


    #exit(0)


    #print(f"Loaded BraTS2020 data\n",len(train_files), len(val_files)); exit(0)

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            #ScaleIntensityd(keys="img"),
            CustomNormalizeIntensity(keys=["img"]),
            #NormalizeIntensity(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=4, neg=4, num_samples=8
                ),
            #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            CustomNormalizeIntensity(keys=["img"]),
            #NormalizeIntensity(keys="img"),
            #ScaleIntensityd(keys="img"),
        ]
    )

    # define dataset, data loader
    
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=batch_size to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=0, collate_fn=list_data_collate)
    check_data = monai.utils.misc.first(check_loader)
    #print("Checking Shapes",check_data["img"].shape, check_data["seg"].shape)

    #print(f"Loaded BraTS2020 data\n",len(train_files), len(val_files))  #; exit(0)

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    #train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=batch_size to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    #for i, batch_data in enumerate(train_loader):
    #    print("Batch Data = >>\t",i, batch_data['img'].shape, batch_data['seg'].shape)

    #exit(0)
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #unet_ecd = UNet3D()
    model = UNet3D().to(device)
    #from monai.networks import nets
    #model = nets.UNet(
    #             spatial_dims=3,
    #             in_channels=1,
    #             out_channels=1,
    #             channels=(16, 32, 64, 128, 256),
    #             strides=(2, 2, 2, 2)) 
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    #print("model==", model)
    print("optim==", optimizer)
    #print("loss==", loss_function)
    #exit(0)

    num_epochs = 50
    
    initiateTraining = TrainWrapper(train_loader, val_loader, model, optimizer, loss_function, device, train_ds, num_epochs)
    for epoch in range(num_epochs):
        initiateTraining.train(epoch)
        best_metric, best_metric_epoch = initiateTraining.val(epoch)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")





if __name__ == "__main__":
    import datetime
    #import wandb
    #wandb.init(project="EloquentCD-3DUNet")
    #run_name = un_name = "ECD_Expt_"+datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
    #wandb.run.name = run_name
    trainWithSyntheticData = True
    if trainWithSyntheticData:
        with tempfile.TemporaryDirectory() as tempdir:
            main(tempdir)
    else:
        main("BraTS2020")
    #wandb.finish()

