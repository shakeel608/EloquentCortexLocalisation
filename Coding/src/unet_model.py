#Author: Dr. Shakeel Ahmad Sheikh
#Affiliation: PostDoctoral Research Scientist, CITEC, University of Bielefeld, Germany
#Date: June 1, 2023
#Description: This script describes various models such as 3D UNet (MONAI package) for Eloquent Cortex Tumor Detection

from monai.networks import nets  
import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.unet3d = nets.UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2)
                #num_res_units=2,
                )
        #self.fc = nn.Linear(45,4)


    def forward(self, x):
        #print("xinp=",x.shape)
        x = self.unet3d(x)
        #print("xop=",x.shape)
        #exit(0)

        return x




class SegmentAnythng():
    def __init__(self):
        self.model = ''
