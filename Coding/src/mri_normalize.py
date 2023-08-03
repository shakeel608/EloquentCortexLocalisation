import numpy as np
import os 
import argparse
import glob
import nibabel as nib
from mri2graph import MRI2Graph
#This class preprocess MRI data by normalisation and by computing dataset statistics

class PreprocessMRI():
    def __init__(self, args):
        self.src_dir = args.src_dir
        self.out_dir = args.out_dir
        self.modality = args.modality   #Ensure Order T1, T1CE, T2, FLAIR
        self.main_folder = args.mri_prefix
        self.standardize = True
        #print("Arguments", self.src_dir, self.modality, self.main_folder)
        self.mod_path = self.get_mri_paths()
        self.num_nodes = args.num_nodes
        #print("Modality", self.mod_path, list(self.mod_path.keys()))
        self.dataset_stats = self.extract_statistics()
        self.dataset_mean = np.array(self.dataset_stats[0], dtype=np.float32)
        self.dataset_std = np.array(self.dataset_stats[1], dtype=np.float32)
        print("Dataset Extracted Multimodal Mean", self.dataset_mean, self.dataset_std)
        #exit(0)
        #self.mri2graph()
        self.mri_to_graph = MRI2Graph(self.modality, self.mod_path, self.dataset_mean, self.dataset_std, self.num_nodes)
        self.mri_to_graph.mri2graph()
        #exit(0)



    def normalize(self, img_array,flatten_img=True):
        #print("In Normalise", img_array.shape)
        #print("length", len(img_array.shape))
        #if len(img_array.shape) > 4:
        #    maxes = np.quantile(img_array, 0.9995, axis=(2,3,4)).astype(np.float32)
        #    norm_img = img_array/maxes[:,:, np.newaxis, np.newaxis, np.newaxis]
        #else:
            #maxes = np.quantile(img_array, 0.9995, axis=(1,2,3)).astype(np.float32)  #99% quantile from full MRI Image SIngle Modality
            #norm_img = img_array/maxes[:,np.newaxis, np.newaxis, np.newaxis]
            
        maxes = np.quantile(img_array, 0.9995).astype(np.float32)  #99% quantile from full MRI Image SIngle Modality
        norm_img = img_array/maxes
        #print("Max value and min", maxes, np.newaxis)
        
        #print("norm image", norm_img[50,70:150,50:150])
        #print("original image", img_array[50,70:150,50:150])
        return norm_img

    
#    def mri2graph(self):
#        """This function first standarizes each MRI image and 
#        then extracts super voxels and converts image to graph with node features"""
#        
#        print("Extrating Graph Supervoxels from MRI Image")
#        num_modalities = len(self.modality)
#        
#        if num_modalities > 1:
#            pass
#        else:
#            modality_path = self.mod_path[self.modality[0].split(".")[0].split("_")[-1]]
#            mri_files = os.listdir(modality_path)
#            for i, mri in enumerate(mri_files):
#                img_path = os.path.join(modality_path, mri)
#                img_data = nib.load(img_path)
#                mri_array = img_data.get_fdata()
#                print("Load SHape", mri_array.shape)
#                mri_array = np.transpose(mri_array, (2,0,1))
#                mri_array = self.normalize(mri_array)
#                stnd_mri = self.standardize_img(mri_array, self.dataset_mean, self.dataset_std)
#                print("Means and sigma dataset", self.dataset_mean, self.dataset_std)
#
#                #print("Standarizzed image", mri_array[50,70:150,50:150])
#
#                print("Standarized image from Function", stnd_mri[50,70:150,50:150])
#                break


    def standardize_img(self, img_array, mean, std):
        centered = img_array - mean
        standardized = centered/std
        return standardized


    def read_mri(self, mri, modality_path, idx):
        """
        Reads and return normalised MRI image using Nibabel package
        idx : index for modality
        0: T1
        1: T1CE
        2: T2
        3: FLAIR

        """
        img = mri.rpartition("_")[0] + self.modality[idx]
        img_path = os.path.join(modality_path, img)
        mri_data = nib.load(img_path)
        mri_array = mri_data.get_fdata()
        mri_array = np.transpose(mri_array, (2,0,1))
        mri_norm = self.normalize(mri_array)
        return mri_norm


    """This function computes mean and std modality wise .ie.e seperately for each modality"""
    def extract_statistics(self):
        print("Computing dataset statistics\n")
        img_mod_mean = []
        img_mod_std = []
        #Read modality wise 
        mod_folders = list(self.mod_path.keys())
        num_modalities = len(self.modality)
        print("Self mod", num_modalities)
        
        image_modalities = []  #For Storing data from each modality 
        
        #Read same file from all modalities for concatentation purposes
        if num_modalities > 1:
            modality_path_t1 = self.mod_path[self.modality[0].split(".")[0].split("_")[-1]]  # 0 for T1 Modality
            modality_path_t1ce = self.mod_path[self.modality[1].split(".")[0].split("_")[-1]]
            #modality_path_t2 = self.mod_path[self.modality[2].split(".")[0].split("_")[-1]]
            #modality_path_flair = self.mod_path[self.modality[3].split(".")[0].split("_")[-1]]
            #print(modality_path_t1, modality_path_t1ce, modality_path_t2, modality_path_flair)
            mri_files = os.listdir(modality_path_t1)
        else:
            modality_path = self.mod_path[self.modality[0].split(".")[0].split("_")[-1]]  # 0 for T1 Modality
            mri_files = os.listdir(modality_path)
        if num_modalities > 1:
            
            for i, mri in enumerate(mri_files):
                image_from_each_modality = []
                #For T1 Modality
                mri_norm_t1 = self.read_mri(mri, modality_path_t1, 0)
                image_from_each_modality.append(mri_norm_t1)
                
                #For T1CE Modality 
                #img = mri.rpartition("_")[0]  + self.modality[1]
                #img_path_t1ce = os.path.join(modality_path_t1ce, img)
                #mri_data_t1ce = nib.load(img_path_t1ce)
                #mri_array_t1ce = mri_data_t1ce.get_fdata()
                #mri_array_t1ce = np.transpose(mri_array_t1ce, (2,0,1))
                #mri_norm_t1ce = self.normalize(mri_array_t1ce)
                #image_from_each_modality.append(mri_norm_t1ce)
                #print("MRI data Format Changed to D X W X H", mri_norm_t1.shape, mri_array_t1ce.shape)
                
                mri_norm_t1ce = self.read_mri(mri, modality_path_t1ce, 1)
                image_from_each_modality.append(mri_norm_t1ce)
                
                #For T2 Modality 
                #mri_norm_t2 = self.read_mri(mri, modality_path_t2, 0)
                #image_from_each_modality.append(mri_norm_t2)
                
                #For FLAIR Modality 
                #mri_norm_flair = self.read_mri(mri, modality_path_flair, 0)
                #print("array_norm funct",mri_norm_falir[50,70:150,70:150])
                #image_from_each_modality.append(mri_norm_flair)
               

                #print("1 read M", mri_norm_t1.shape, mri_norm_t1ce.shape,len(image_modalities))
                image_from_each_modality = np.stack(image_from_each_modality, 0)  #Stacking NOrmalised 3D Images 
                image_modalities.append(image_from_each_modality) 
                #print("Combined Modality", np.stack(image_from_each_modality, 0).shape)
                
                if i ==3:
                    break
        #Single modality
        else:
            for i, img in enumerate(mri_files):
                img_path = os.path.join(modality_path, img)
                mri_norm = self.read_mri(img, modality_path, 0)
                image_modalities.append(mri_norm)
                #print("Stat", mri_norm.shape)
                if i==3:
                    break

        #print("Dataset Moda mean", np.mean(img_mod_mean), np.median(img_mod_mean, axis=0));exit(0)
        patient_mri_allmodals_normalize = np.stack(image_modalities, 0)
        
        #print("Patient sample Before Normalise", patient_mri_allmodals_normalize.shape) #; exit(0)
        #patient_mri_allmodals_normalize = self.normalize(patient_mri_allmodals)  #Normalisation
        #print("Patient sample After Normalise", patient_mri_allmodals_normalize.shape) #; exit(0)
        #exit(0)
        


        if num_modalities == 1:
            #print("Image mode", img_mod_mean)
            #dataset_mean, dataset_median = np.mean(img_mod_mean), np.median(img_mod_mean, axis=0)
            #dataset_std, dataset_std_median = np.std(img_mod_std), np.median(img_mod_std, axis=0)
            dataset_mean = np.mean(patient_mri_allmodals_normalize)
            dataset_std = np.std(patient_mri_allmodals_normalize)
            print("Dataset Moda mean and std", dataset_mean, dataset_std)
            #exit(0)
            return dataset_mean, dataset_std

        else:
            #Mean and STD across each Channel/Modality for each sample 
            #mean = np.mean(patient_mri_allmodals, axis=(2,3,4))  #N X C X D X W X H
            #std  = np.std(patient_mri_allmodals, axis=(2,3,4))
            #print("Means", mean.shape, std.shape)
            #Channel wise Stats
            #dataset_mean_channel = np.mean(mean, axis=0)
            #dataset_std_channel = np.mean(std, axis=0)
            #Stats Over entire dataset
            dataset_mean = np.mean(patient_mri_allmodals_normalize)
            dataset_std = np.std(patient_mri_allmodals_normalize)
            print("Dataset Multimodal Mean", dataset_mean, dataset_std)

            return dataset_mean, dataset_std




    """This function returns a dictionary with key as modaity/seg label and value as path to folders"""
    def get_mri_paths(self):

        #Contains Folder of FLAIR, T1, T2, T1CE adn SEG Labels
        data_dirs = glob.glob(f"{self.src_dir}**/{self.main_folder}*/", recursive=True)

        #Creates a dictiionary pair with keys as modality/seg and values as paths to the respective folders
        data_dict = {m.split("/")[-2]:m for m in data_dirs if m.split("/")[-2] in ['flair', 't1', 't1ce', 't2', 'seg']}


        return data_dict
    
if __name__ == "__main__":

    cla_parser = argparse.ArgumentParser()   #CLA Command Line Arguments 

    cla_parser.add_argument('-src', '--src_dir', default="", help='Input Source Directory, type=str')
    cla_parser.add_argument('-out', '--out_dir', default="", help='Output Directory, type=str')
    cla_parser.add_argument('-mod','--modality', nargs="+", default=["_flair.nii.gz","_t1.nii.gz","_t1ce.nii.gz","_t2.nii.gz"],help="Modality Type FLAIR, T1, T1CE or T2.")
    cla_parser.add_argument('-mri_prefix','--mri_prefix', help="A main directory 'BraTS2023' which contains other sub modalities ")
    cla_parser.add_argument('-nn','--num_nodes', default=300,  help="Number of Nodes in a Graph")
    
    args = cla_parser.parse_args()
    preprocess = PreprocessMRI(args)





