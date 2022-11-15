"""
Short script to automatically count the number of seeds in a kiwi. Assignment #1 of FY8408.

Author: Marc-Antoine Fortin, NTNU, November 2022
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pydicom as dicom
import cc3d
import cv2
import skimage

all_acqs = ['E27', 'E28', 'E26', 'E30', 'E18']
# Explicitly choose one of the 3D images we acquired.
acq = all_acqs[2]
matplotlib.use('TkAgg')

###############################################################

def load_image(path2folder):
    # All 3D volumes/images we acquired during the experiment
    # Steps to get the fullpath to the 3D volume
    kiwidir = os.listdir(path2folder)
    # kiwidir = os.listdir(path2folder)
    subdir = [x for x in kiwidir if acq in x]  # that's why I love python
    filename = 'EnIm1.dcm'  # they are all called the same
    fullpath2dcm = os.path.join(path2folder, subdir[0], filename)

    # load dicom image
    img = dicom.dcmread(fullpath2dcm)
    # Get the actual image array
    data = img.pixel_array  # int16 data type
    # data2 = data.astype("uint8")
    # print(data2.dtype)
    return data


def determine_start_end(data):
    # determine start and end slice manually
    dims = data.shape
    plt.ion()
    start_image = 0
    pause = 0.9
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for slice in range(start_image, dims[0] - 1):
        ax.imshow(data[slice, :, :], cmap='gray')
        ax.set_title('Slice #{}'.format(slice))
        plt.show()
        plt.pause(pause)

    plt.close()
    plt.ioff()


def count_seeds_2(data, first_slice, last_slice):

    dims = data.shape
    ww, hh = data.shape[1:3]
    hh2 = hh // 2
    ww2 = ww // 2

    # define ellipse for mask
    mask = np.zeros(data.shape[1:3])
    center = (hh2, ww2)
    mask = cv2.ellipse(mask, center=center, axes=(45, 25), angle=0, startAngle=0, endAngle=360,
                       color=(255, 255, 255), thickness=-1)

    # other preprocessing test
    kernel = 30
    max_kernel = []
    local_max = []
    for slice in range(dims[0]):
        max_kernel.append(cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel)))
        local_max.append(cv2.morphologyEx(data[slice, ...], cv2.MORPH_CLOSE, max_kernel[slice], None, None, 1,
                                    cv2.BORDER_REFLECT101))

    gain_division = np.where(local_max == 0, 0, (data / local_max))
    gain_division = np.clip((255 * gain_division), 0, 255)
    gain_division = gain_division.astype("uint8")

    result = []
    cont_count = []
    for slice in range(data.shape[0]):
        masked = data[slice, ...] * mask
        thresh_otsu = skimage.filters.threshold_otsu(masked)
        res = masked > thresh_otsu * 1.2
        (contours, _) = cv2.findContours(res.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont_count.append(contours)
        result.append(res)

    result = np.asarray(result)
    # cut to part where seeds are visible
    result = result[first_slice:last_slice, ...]
    data = data[first_slice:last_slice, ...]
    gain_division = gain_division[first_slice:last_slice, ...]

    connectivity_3d = 6

    # the input to the 3D connected components algorithm is the mask (not the masked image)
    # TODO: refine region of interest to only check for seeds around
    labels_out, N = cc3d.connected_components(result, connectivity=connectivity_3d, return_N=True)
    print('Seeds in 3D contour: {}'.format(N))

    counter = 0
    connectivity_2d = 6
    for slice in range(result.shape[0]):
        _, N = cc3d.connected_components(result[slice, ...], connectivity=connectivity_2d, return_N=True)
        counter += N
    print('Seeds in 2D contour: {}'.format(counter))

    fig, axs = plt.subplots(1, 3)
    fig.suptitle(acq)
    axs[0].imshow(data[int(dims[0] / 2), :, :], cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(result[int(dims[0] / 2)], cmap='gray')
    axs[1].set_title('Masked Image')
    axs[1].axis('off')
    axs[2].imshow(gain_division[int(dims[0] / 2)], cmap='gray')
    axs[2].set_title('Gain-division Image')
    axs[2].axis('off')
    plt.show()


def count_seeds(data, first_slice, last_slice):
    dims = data.shape
    thresh_data = np.zeros(dims)
    mask = np.zeros(dims)  # np.zeros((1,dims[-1]))
    # TODO: wrong dimension? slice dim is 60 2d image is 100x128 - dim[1] dim[2]
    for slice in range(dims[0]):
        thresh_otsu = skimage.filters.threshold_otsu(data[slice, :, :])
        mask[slice, :, :] = data[slice, :, :] > thresh_otsu * 1.5

    # drop slices without seeds
    data_masked = data * mask

    mask_drop = mask[first_slice:last_slice, ...]

    # Show several 2D slices of the mask/masked image
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    axs = axs.ravel()

    for i, a in enumerate(axs):
        s = i * round(dims[0] / 10)
        fig.suptitle(acq + ' - Thresholded Image')
        axs[i].imshow(mask[s, :, :], cmap='gray')
        axs[i].set_title('Slice #' + str(s))
        axs[i].axis('off')

    plt.show()

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(acq)
    axs[0].imshow(data[int(dims[0] / 2), :, :], cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(data_masked[int(dims[0] / 2), :, :], cmap='gray')
    axs[1].set_title('Thresholded Image')
    axs[1].axis('off')
    plt.show()

    # 3D Connected Component Analysis
    # number of pixels that composes one seed (for 3D, only 6, 18 or 26 are accepted)
    connectivity = 18

    # the input to the 3D connected components algorithm is the mask (not the masked image)
    # TODO: refine region of interest to only check for seeds around
    labels_in = mask_drop
    labels_out, N = cc3d.connected_components(labels_in, connectivity=connectivity, return_N=True)
    print('Number of labels/kiwi seeds: ', N)


if __name__ == '__main__':
    # alternate path here
    path2folder = '/Users/felixtempel/Library/Mobile Documents/com~apple~CloudDocs/PhD/Courses/MRI/kiwi_images'
    # path2folder = '/home/mafortin/OneDrive/PhD/PhD_Courses/FY8408/Kiwi-images/3D/'
    # load image
    data = load_image(path2folder)

    # determine the start number of seed manually
    # first slice with seed == 5
    # last slice with seed == 47
    # determine_start_end(data)

    FIRST_SLICE = 5
    LAST_SLICE = 47

    count_seeds_2(data, FIRST_SLICE, LAST_SLICE)

