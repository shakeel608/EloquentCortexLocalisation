#Author: Dr. Shakeel Ahmad Sheikh
#Affiliation: PostDoc Research Scientist, CITEC, University of Bielefeld, Germany
#Date: May 25, 2023
#Description: This script extracts latent embeddings and final predictions (Shape of Tumor in 3D) from a 3D UNet (MONAI package) for Eloquent Cortex Tumor Detection 

import os
import sys
import logging

import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch

import monai
from pprint import pprint
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.data import list_data_collate, Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
)

class UNetEmbeds():
    def __init__(self, tempdir, isLatentEmbedsExtract=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.isLatentEmbedsExtract = isLatentEmbedsExtract
        #self.test_transforms = []
        self.dice_metric, self.post_trans = [], []
        self.tempdir = tempdir
        print("Retreiveng Data","*"*10)
        self.get_data()

    def extract_embedsUNet(self, model, test_loader, test_files):
        model.eval()
        #saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg")
        if self.isLatentEmbedsExtract:
            print("Extracting Latent Embeddngs from Bottleneck...")
            pass
        else:
            print("Extracting Output Masked Shapes...")
            with torch.no_grad():
                for test_data in test_loader:
                    test_images = val_data["img"].to(self.device) #Load 3D Image
                    #test_labels = val_data["seg"].to(self.device)  #Load 3D Seg Label
                    # define sliding window size and batch size for windows inference
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
                    test_outputs = [self.post_trans(i) for i in decollate_batch(test_outputs)]
                    #test_labels = decollate_batch(test_labels)
                    # compute metric for current iteration
                    #self.dice_metric(y_pred=test_outputs, y=test_labels)
                    #for test_output in test_outputs:
                        #saver(test_output)
                # aggregate the final mean dice result
                #print("evaluation metric:", self.dice_metric.aggregate().item())
                # reset the status
                #dice_metric.reset()

    def get_data(self):
        print(f"generating synthetic data to {tempdir} (this may take a while)")
        for i in range(5):
            im, _ = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
            n = nib.Nifti1Image(im, np.eye(4))
            nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))
        
        images = sorted(glob(os.path.join(self.tempdir, "im*.nii.gz")))
        #segs = sorted(glob(os.path.join(self.tempdir, "seg*.png")))
        #test_files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
        test_files = [{"img": img} for img in images]

        # define transforms for image and segmentation
        pre_test_transforms = Compose(
            [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            Orientationd(keys="img", axcodes="RAS"),
            Resized(keys="img", spatial_size=(96, 96, 96), mode="trilinear", align_corners=True),
            ScaleIntensityd(keys=["img"]),
            ]
        )
        test_ds = monai.data.Dataset(data=test_files, transform=pre_test_transforms)
        # sliding window inference need to input 1 image in every iteration
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        #self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        # define post transforms
        self.post_trans = Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                Invertd(
                    keys="pred",  # invert the `pred` data field, also support multiple fields
                    transform=pre_test_transforms,
                    orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
                    # then invert `pred` based on this information. we can use same info
                    # for multiple fields, also support different orig_keys for different fields
                    nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                    # to ensure a smooth output, then execute `AsDiscreted` transform
                    to_tensor=True,  # convert to PyTorch Tensor after inverting
                ),
                AsDiscreted(keys="pred", threshold=0.5),
                SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
            ]
        )

      #  model = UNet(
      #          spatial_dims=3,
      #          in_channels=1,
      #          out_channels=1,
      #          channels=(16, 32, 64, 128, 256),
      #          strides=(2, 2, 2, 2),
      #          num_res_units=2,
      #      ).to(self.device)
        model = torch.load("./pretrained_models/model.pt", map_location=self.device)
        #model.load_state_dict(torch.load("./pretrained_models/model.pt", map_location=self.device))
        #model.load_state_dict(torch.load("./pretrained_models/model.pt", map_location=self.device),strict=False)
        #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        #pprint("***Model Configuration***\n""", dir(model))
        #model.load_state_dict(model)
        exit(0)
        self.extract_embedsUNet(model, test_loader, test_files)



def main(tempdir):
    unet_embeds = UNetEmbeds(tempdir)

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tempdir:
	    main(tempdir)















