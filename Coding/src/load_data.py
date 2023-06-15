import torch
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms import Compose, ToTensor

class ECDDataLoader(Dataset):
    def __init__(self, img_path, lbl_path, transform=None):
        self.img_path = img_path
        self.lbl_path = lbl_path
        #self.transform = transform or Compose([ToTensor()])

    def __getitem__(self, index):
        image_path = self.img_path[index]
        label_path = self.lbl_path[index]

        # Load NIfTI image using nibabel
        image = nib.load(image_path)
        image_data = image.get_fdata()

        # Load NIfTI Seg Label using nibabel
        label = nib.load(label_path)
        label_data = label.get_fdata()
        # Apply transformations
        #image_data = self.transform(image_data)


        #print("image data and seg data", image.shape, label.shape)

        return image_data, label_data

    def __len__(self):
        return len(self.img_path)

