# Author Dr. Shakeel A. Sheikh
# Date: May 2, 2023
# Description: This script helps to visualise 3D medical images of type DICOM and NII

"""Ensures dir structure as below
PatientsFolder
    |---PatientID
	|---ImageType  "DICOM" or "NII"
	    |---Image Slices
"""

import numpy as np
import matplotlib.pyplot as plt
import os, glob
import pydicom
import sys
import matplotlib.path as mplPath
import nibabel as nib
import sys 

class IndexReader():
	"""Reads and Visualise 3D MRI Images"""
	def __init__(self, ax, X, patient_id):
		self.ax = ax
		self.X = X
		self.rows, self.cols, self.slices = self.X.shape
		self.sliceindex = self.slices//2
		ax.set_title("MRI of Patient = " + str(patient_id))
		self.im = ax.imshow(self.X[:,:,self.sliceindex])   #128//2 == 64 Mid for scrol up and down
		self.update()
	
	def update(self):
		self.im.set_data(self.X[:,:,self.sliceindex])
		self.ax.set_ylabel("Slice Number: %s" % self.sliceindex)
		self.im.axes.figure.canvas.draw()

	def onscroll(self, event):
		print("Event Type",event.button, event.step)
		if event.button == 'up':
			self.sliceindex = (self.sliceindex + 1) % self.slices
		else:
			self.sliceindex = (self.sliceindex - 1) % self.slices
		
		self.update()
			
class MediVisio():
	
	def __init__(self, patients_dir, img_type, img_label, alpha=0.999, mask=True):
		self.patients_dir = patients_dir
		self.img_type = img_type  #Should be folder name in the director Patients > PatientID > DICOM/NII
		self.plots = []  #For Slices DICOM Images
		self.img3D = []  #NII already in 3D Tensor
		self.img_label = img_label
		#Overlay acts as a Transparency on original image
		self.applymask = mask
		self.alpha = alpha
	
	def apply_mask(self, pixel_data, voxel_mask):
		voxel_mask = voxel_mask.astype(float) / np.max(voxel_mask)
		print("Mask Seg", voxel_mask)
		masked_inp = (1 - self.alpha) * pixel_data + self.alpha * voxel_mask

		return masked_inp

	def stack_slices(self, plots):
		
		img3d = np.dstack(plots)  #Stacks all slices in sequential order
		return img3d
	
	def check_user_input(self):
		
		"""Press 'q' for exit or press 'Enter' to continue"""
		user_input = input("\nPress 'Enter' to continue to visualise other patients or 'q' to exit\n")
		if user_input.lower() == "q":
			print("Exiting MediVisio\n")
			exit(0)

	def read_data(self):
		
		"""Read Medical Images and Sends to ndexTracker for Visualisation Purposes"""
		for patient_id, patient in enumerate(glob.glob(self.patients_dir+"*")):  #* means all patients
			self.data3D_dir = patient + "/" +self.img_type
			for i, itm in enumerate(sorted(glob.glob(self.data3D_dir +"/*"))):
				itm_label = itm.rsplit("/",2)[0] + '/' + self.img_label + '/'+itm.rsplit("/",1)[-1]
				print("PatientID",i, itm)
				"""For MRI Medical Images"""
				if self.img_type == "DICOM": 
					mri = pydicom.dcmread(itm)
					print("MRI Pixel Data",i,mri.PatientID, mri.pixel_array.shape)
					pixel = mri.pixel_array
					pixel = pixel*1 + (-1024)
					if sel.applymask:
						mri_label = pydico.dcmread(itm_label)
						voxel_mask = mri_label.pixel_array
						pixel = self.apply_mask(pixel, voxel_mask)
					self.plots.append(pixel)
					patient_id = mri.PatientID

				"""For NII Medical Images"""
				if self.img_type == "NII":
					nii = nib.load(itm)
					print("label", itm_label)
					nii_label = nib.load(itm_label)
					print("NII Data",nii.get_fdata().shape)
					pixel_data = nii.get_fdata()
					pixel_data = pixel_data*1 + (-1024)
					
					#Plot With Mask Segmentation
					if self.applymask:
						voxel_mask = nii_label.get_fdata()
						pixel_data = self.apply_mask(pixel_data, voxel_mask)

					patient_id = nii.header.get('db_name')
					if pixel_data.shape[2] == 1:
					    self.plots.append(pixel_data)
					else:
					    fig, ax = plt.subplots(1,1)
					    #self.img3D = pixel_data*1 + (-1024)
					    self.img3D = pixel_data
					    slice_tracker = IndexReader(ax, self.img3D, patient_id)
					    fig.canvas.mpl_connect('scroll_event', slice_tracker.onscroll)
					    plt.show()
					    
					    self.check_user_input()  #To visualise other patients
			"""For Concatenatd Slices"""
			if self.plots:
			    fig, ax = plt.subplots(1,1)
			    self.img3D = self.stack_slices(self.plots)
			    slice_tracker = IndexReader(ax, self.img3D, patient_id)
			    fig.canvas.mpl_connect('scroll_event',slice_tracker.onscroll)
			    plt.show()
			    self.check_user_input()  #To visualise other patients
		

	
def main():
	patients_dir = os.getcwd() + "/../../Datasets/EloquentCortex/" + "patients/"
	medi_vis = MediVisio(patients_dir, "NII", "NII_Labels")  #alpha = 0.9 , applymask=True by Default
	medi_vis.read_data()

if __name__ == "__main__":
    main()



