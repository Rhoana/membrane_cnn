import mahotas
import scipy.ndimage
import scipy.misc
import numpy as np
import gzip
import cPickle
import glob
import os
import h5py
import partition_comparison
from datetime import date

#param_path = 'D:/dev/Rhoana/membrane_cnn/results/good3/'
param_path = 'D:/dev/Rhoana/membrane_cnn/results/stump_combo/'
param_files = glob.glob(param_path + "*.h5")

param_files = [x for x in param_files if x.find('.ot.h5') == -1]

date_string = date.today().strftime('%d%B%Y')
combo_file = 'D:/dev/Rhoana/membrane_cnn/results/combo/deep_net_combo{0}_{1}.h5'.format(len(param_files), date_string)

combo_output_h5 = h5py.File(combo_file, 'w')
combo_output_h5['/nets'] = len(param_files)

for i, param_file in enumerate(param_files):

    net_string = '/net{0}'.format(i)
    print param_file

    offset_file = param_file.replace('.h5', '.sm.ot.h5')

    offset_h5 = h5py.File(offset_file, 'r')
    best_offset = offset_h5['/best_offset'][...]
    best_sigma = offset_h5['/best_sigma'][...]
    offset_h5.close()

    combo_output_h5[net_string + '/best_offset'] = best_offset
    combo_output_h5[net_string + '/best_sigma'] = best_sigma

    network_h5 = h5py.File(param_file, 'r')

    nlayers = network_h5['/layers'][...]

    combo_output_h5[net_string + '/layers'] = nlayers
    combo_output_h5[net_string + '/iterations'] = network_h5['/iterations'][...]
    combo_output_h5[net_string + '/verror'] = network_h5['/verror'][...]
    combo_output_h5[net_string + '/terror'] = network_h5['/terror'][...]

    downsample = 1
    if param_file.find('_ds2') != -1:
        downsample = 2
    elif param_file.find('_ds4') != -1:
        downsample = 4

    combo_output_h5[net_string + '/downsample_factor'] = downsample

    if param_file.find('Stump') != -1:
        combo_output_h5[net_string + '/stumpin'] = True

    print 'Network {0} has {1} layers.'.format(i, nlayers)

    for layer in range(nlayers):

        layer_string = '/layer{0}/'.format(layer)
        layer_type = network_h5[layer_string + 'type'][...]

        combo_output_h5[net_string + layer_string + 'type'] = str(layer_type)

        if layer_type == 'Convolution':

            combo_output_h5[net_string + layer_string + 'weights'] = network_h5[layer_string + 'weights'][...]
            combo_output_h5[net_string + layer_string + 'bias'] = network_h5[layer_string + 'bias'][...]
            combo_output_h5[net_string + layer_string + 'maxpoolsize'] = network_h5[layer_string + 'maxpoolsize'][...]

        elif layer_type == 'FullyConnected':

            combo_output_h5[net_string + layer_string + 'weights'] = network_h5[layer_string + 'weights'][...]
            combo_output_h5[net_string + layer_string + 'bias'] = network_h5[layer_string + 'bias'][...]
            combo_output_h5[net_string + layer_string + 'ksize'] = network_h5[layer_string + 'ksize'][...]

        elif layer_type == 'LogisticRegression':

            combo_output_h5[net_string + layer_string + 'weights'] = network_h5[layer_string + 'weights'][...]
            combo_output_h5[net_string + layer_string + 'bias'] = network_h5[layer_string + 'bias'][...]

        else:
            raise Exception("Unknown layer type: {0}".format(layer_type))

    network_h5.close()

combo_output_h5.close()
