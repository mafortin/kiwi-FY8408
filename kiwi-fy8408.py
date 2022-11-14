"""
Short script to automatically count the number of seeds in a kiwi. Assignment #1 of FY8408.

Author: Marc-Antoine Fortin, NTNU, November 2022
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom as dicom
import cc3d
import cv2 as cv
import skimage

####### THIS SECTION REQUIRES USER INPUT##########################3
# Path to the folder containing all kiwi images (VARY FOR EACH USER)
path2folder = '/home/mafortin/OneDrive/PhD/PhD_Courses/FY8408/Kiwi-images/3D/'
# All 3D volumes/images we acquired during the experiment
all_acqs = ['E27', 'E28', 'E26', 'E30', 'E18']
# Explicitly choose one of the 3D images we acquired.
acq = all_acqs[2]
###############################################################

# Steps to get the fullpath to the 3D volume
kiwidir = os.listdir(path2folder)
subdir = [x for x in kiwidir if acq in x] #that's why I love python
filename = 'EnIm1.dcm' #they are all called the same
fullpath2dcm = os.path.join(path2folder, subdir[0], filename)

# load dicom image
img = dicom.dcmread(fullpath2dcm)
# Get the actual image array
data = img.pixel_array #int16 data type
#data2 = data.astype("uint8")
#print(data2.dtype)
dims = data.shape

thresh_data = np.zeros(dims)
mask = np.zeros(dims) #np.zeros((1,dims[-1]))

# Threshold the kiwi image
for sli in range(dims[-1]):

        thresh_otsu = skimage.filters.threshold_otsu(data[:,:,sli])
        mask[:,:,sli] = data[:,:,sli] > thresh_otsu*1.5

data_masked = data*mask

# Show the original and the thresholded image
"""
fig, axs = plt.subplots(1,2)
fig.suptitle(acq)
axs[0].imshow(data[int(dims[0]/2), :, :], cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(data_masked[int(dims[0]/2), :, :], cmap='gray')
axs[1].set_title('Thresholded Image')
axs[1].axis('off')
plt.show()
"""


# Show several 2D slices of the mask/masked image
fig, axs = plt.subplots(2, 4, figsize=(15, 6))
axs = axs.ravel()

for i, a in enumerate(axs):

        s = i*round(dims[0]/10)
        fig.suptitle(acq + ' - Thresholded Image')
        axs[i].imshow(mask[s, :, :], cmap='gray')
        axs[i].set_title('Slice #' + str(s))
        axs[i].axis('off')

plt.show()


# 3D Connected Component Analysis
connectivity = 18 # number of pixels that composes one seed (for 3D, only 6, 18 or 26 are accepted)

labels_in = mask #the input to the 3D connected components algorithm is the mask (not the masked image)
labels_out, N = cc3d.connected_components(labels_in, connectivity=connectivity, return_N=True)
print('Number of labels/kiwi seeds: ', N)

