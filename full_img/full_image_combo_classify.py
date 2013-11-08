
import os
import sys
import time

import numpy as np
import mahotas

import glob
import h5py

execfile('full_image_cnn.py')
#from full_image_cnn import *

input_image_path = sys.argv[1]
combo_net_path = sys.argv[2]
output_image_path = sys.argv[3]

combo_net = ComboDeepNetwork(combo_net_path)

#image_path='D:/dev/datasets/isbi/train-input/train-input_0000.tif'
#gold_image_path='D:/dev/datasets/isbi/train-labels/train-labels_0000.tif'

image_path_format_string='D:/dev/datasets/LGN1/gold/images/2kSampAligned{0:04d}.tif'
gold_image_path_format_string='D:/dev/datasets/LGN1/gold/lxVastExport_8+12+13/Segmentation1-LX_8-12_export_s{0:03d}.png'

def normalize_image(original_image, saturation_level=0.005):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    return np.uint8(255 - norm_image)


input_image = np.float32(normalize_image(mahotas.imread(input_image_path)))

average_image = combo_net.apply_combo_net(input_image)


def write_image (output_path, data, image_num=0, downsample=1):
    if downsample != 1:
        data = np.float32(mahotas.imresize(data, downsample))
    maxdata = np.max(data)
    mindata = np.min(data)
    normdata = (np.float32(data) - mindata) / (maxdata - mindata)
    mahotas.imsave(output_path, np.uint16(normdata * 65535))

write_image(output_image_path, average_image)

print 'Classification complete.'
