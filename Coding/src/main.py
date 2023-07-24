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
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from load_data import ECDDataLoader
#import random
import wandb
import os
os.environ["WANDB_API_KEY"] = "c5a734b88c6861719c0e735d88a7fe8cf9bd4a88"

wandb.login()

monai.utils.set_determinism(seed=0, additional_settings=None)


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


#Converting Brain Tumor Labels i.e Multiclass llabels into Multi-Label Segmentation task using One-Hot Format

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    #2023 New  Challenge
    label 1: NCR Necrotic
    label 2: ED Peritumoral edematous/invaded tissue 
    label 3: ET Enhancing Tumor
    label 0: Everything else

    """

    def __call__(self, data):
        d = dict(data)
        #print("Data Unique in dict",torch.unique(d['img']), torch.unique(d['seg']))
        for key in self.keys:
            result = []
            print("keys",key, self.keys); exit(0)
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d



def main(tmpdir, SEED):
    
    # Set the random seed for reproducibility
    
    #random.seed(SEED)
    #np.random.seed(SEED)
    #torch.manual_seed(SEED)
    #torch.cuda.manual_seed_all(SEED)
    
    #print(tmpdir);exit(0)
    pos_neg_num = 4
    batch_size = 4
    num_samples = 40  #Number of samples to generate
    #path = "/Users/dr.shakeel/PostDocWork/Datasets/MICCAI_BraTS2020_Data/"  #Local Path

    ##Train 1251,with T1, T1CE, T2, FLAIR
    path = "/homes/ssheikh/postdoc_work/Datasets/BraTS2023/"   #2023BraTS Cluster Path
    data_reader = ECDLoader(num_samples, path, SEED)

    syntheticTrain = False
    if syntheticTrain:
        train_files, val_files, test_files = data_reader.generate_3d_data(tmpdir)
    else:
        train_files, val_files, test_files = data_reader.load_BraTsTumorData()
    
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
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            #ScaleIntensityd(keys="img"),
            CustomNormalizeIntensity(keys=["img"]),
            #NormalizeIntensity(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=pos_neg_num, neg=pos_neg_num, num_samples=pos_neg_num*2
                ),
            #RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            CustomNormalizeIntensity(keys=["img"]),
            #NormalizeIntensity(keys="img"),
            #ScaleIntensityd(keys="img"),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
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
    print("len",len(train_ds))
    #train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=batch_size to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
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
    
    #Test Dataset
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0, collate_fn=list_data_collate)
    #print("Test Dataset Length", len(test_ds));exit(0)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Device = {}".format(device)) 
    #unet_ecd = UNet3D()
    model = UNet3D().to(device)
    #from monai.networks import nets
    #model = nets.UNet(
    #             spatial_dims=3,
    #             in_channels=1,
    #             out_channels=1,
    #             channels=(16, 32, 64, 128, 256),
    #             strides=(2, 2, 2, 2))
    isSigmoidTrue = "SigmoidFalse"

    loss_function = monai.losses.DiceLoss(sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-3)
    #print("model==", model)
    print("optim==", optimizer)
    #scheduler = ExponentialLR(optimizer, gamma=0.5)   #Decay Factor in 1/2
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)   #Decay Factor in 1/10
    #print("loss==", loss_function)
    #exit(0)

    num_epochs = 500
    
    initiateTraining = TrainWrapper(train_loader, val_loader, model, optimizer, loss_function, device, train_ds, num_epochs, SEED)
    #for i, (name, param) in enumerate(model.state_dict().items()):
        #print(f" Before Train Parameter: {name}")
        #if name == "unet3d.model.2.conv.bias":
        #    print(f"Weights:\n{param}")
        #    print(i,"---------------------")
    for epoch in range(num_epochs):
        initiateTraining.train(epoch)
        best_metric, best_metric_epoch = initiateTraining.val(epoch)
        #for i, (name, param) in enumerate(model.state_dict().items()):
        #    print(f"After Train parameter: {name}")
        #    if name == "unet3d.model.2.conv.bias":
        #        print(f"weights:\n{param}")
        #        print(i,"---------------------")
        print("Learning rate = {}".format(optimizer.param_groups[0]['lr']))
        #scheduler.step()

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    print("\nLoading trained model for Inference\n")
    model_path = "best_metric_model_segmentation3d_dict.pth"
    net = UNet3D().to(device)
    net.load_state_dict(torch.load(model_path))
    #for name, param in model.state_dict().items():
    #    print(f"Loaded Parameter: {name}")
    #    if name == "unet3d.model.2.conv.bias":
    #        print(f"Weights:\n{param}")
    #        print(i,"---------------------")

    initiateTraining.evaluate(test_loader, net)




if __name__ == "__main__":
    import datetime
    wandb.init(project="EloquentCD-3DUNetBraTS2023_StepLR")
    #import wandb
    #wandb.init(project="EloquentCD-3DUNet")
    #run_name = un_name = "ECD_Expt_"+datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
    trainWithSyntheticData = False
    #SEEDS = [671, 204, 752, 634, 649, 254, 876, 690, 969, 44]
    SEEDS = [876]
    for SEED in SEEDS:
        run_name  = "ECD_Expt_"+datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
        wandb.run.name = run_name
        if trainWithSyntheticData:
            with tempfile.TemporaryDirectory() as tempdir:
                main(tempdir, SEED)
        else:
            #N=10 Experiments, so setup 10 random seeds 
            main("BraTS2020", SEED)
    wandb.finish()

