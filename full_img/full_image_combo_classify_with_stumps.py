
import os
import sys
import time

import numpy as np
import mahotas

import glob
import h5py

#execfile('full_image_cnn.py')
from full_image_cnn import *

input_image_path = 'D:\\dev\\datasets\\LGN1\\gold\\images\\2kSampAligned01*.tif'
input_stump_path = 'D:\\dev\\datasets\\LGN1\\JoshProbabilities\\2kSampAligned01*.tif'
combo_net_file = 'D:\\dev\\Rhoana\\membrane_cnn\\results\\combo\\stump_combo_nets_ws_optimized\\deep_net_combo3_13November2013.h5'
output_path = 'D:\\dev\\datasets\\LGN1\\NetProbs\\'

image_files = sorted(glob.glob(input_image_path))
stump_files = sorted(glob.glob(input_stump_path))

image_downsample_factor = 1
image_inverted = False

# input_image_file = sys.argv[1]
# combo_net_file = sys.argv[2]
# output_image_file = sys.argv[3]

# if len(sys.argv) > 4:
#     image_downsample_factor = int(sys.argv[4])

# if len(sys.argv) > 5:
#     image_inverted = sys.argv[5] == 'i'

combo_net = ComboDeepNetwork(combo_net_file)

def normalize_image(original_image, saturation_level=0.005, invert=True):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    if invert:
        norm_image = 255 - norm_image
    return np.uint8(norm_image)

for image_i in range(len(image_files)):

    input_image_file = image_files[image_i]
    print input_image_file
    input_image = np.float32(normalize_image(mahotas.imread(input_image_file), invert=(not image_inverted)))

    stump_image = None
    if len(stump_files) > image_i:
        input_stump_file = stump_files[image_i]
        print input_stump_file
        stump_image = np.float32(normalize_image(mahotas.imread(input_stump_file), invert=(not image_inverted)))

    #input_image = input_image[512:1536,512:1536]
    #stump_image = stump_image[512:1536,512:1536]

    output_image_file = output_path + os.path.basename(input_image_file)

    if image_downsample_factor != 1:
        input_image = mahotas.imresize(input_image, image_downsample_factor)
        stump_image = mahotas.imresize(stump_image, image_downsample_factor)

    average_image = combo_net.apply_combo_net(input_image, stump_input=stump_image)
    #average_image, parts = combo_net.apply_combo_net(input_image, stump_input=stump_image, return_parts=True)

    def write_image (output_file, data, image_num=0, downsample=1):
        if downsample != 1:
            data = np.float32(mahotas.imresize(data, downsample))
        maxdata = np.max(data)
        mindata = np.min(data)
        normdata = (np.float32(data) - mindata) / (maxdata - mindata)
        mahotas.imsave(output_file, np.uint16(normdata * 65535))

    write_image(output_image_file, average_image)

    #for part_i, part in enumerate(parts):
    #    write_image(output_image_file + '.part{0}.tif'.format(part_i), part)

    print 'Classification complete.'
