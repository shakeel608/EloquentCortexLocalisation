import os 
import monai
from monai.apps import download_and_extract
import glob
#Dataset Location
resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"

root_folder = "/Users/dr.shakeel/PostDocWork/Datasets/"
data_dst_dir  = os.path.join(root_folder, "Task09_Spleen")

compressed_data = os.path.join(root_folder, "Spleen3D.tar")

if not os.path.exists(data_dst_dir):
	download_and_extract(resource, compressed_data, root_folder)


train_mri      = sorted(glob.glob(os.path.join(data_dst_dir, "imagesTr", "*.nii.gz")))
train_lbls_mri = sorted(glob.glob(os.path.join(data_dst_dir, "labelsTr", "*.nii.gz")))

#print("train_mri = {}".format(train_mri))

#Create Dictionary for Img Path and Label Names as a Key Value Pair 

MRI_data_dict = [
		{'img3d': image_mri, "label": label_mri}
		for image_mri, label_mri in zip(train_mri, train_lbls_mri)		
		]
print("MRI DIC===>", len(MRI_data_dict))
train_data, val_data = MRI_data_dict[:int(len(MRI_data_dict)*0.8)], MRI_data_dict[int(len(MRI_data_dict)*0.8):]
print('Tol Len', len(train_data), len(val_data))
