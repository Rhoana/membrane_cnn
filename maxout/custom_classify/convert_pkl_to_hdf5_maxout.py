from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse

import sys
import glob
import h5py

import hashlib
import shutil

import numpy as np

input_path = sys.argv[1]

if input_path.endswith('.pkl'):
    param_files = [input_path]
else:
    param_files = glob.glob(input_path + '*.pkl')

print "Found {0} .pkl files.".format(len(param_files))

for input_paramfile in param_files:

    output_paramfile=input_paramfile.replace('.pkl', '.h5')

    print 'Opening file {0}.'.format(input_paramfile)

    model = serial.load(input_paramfile)

    print 'Loaded progress file.'

    h5file = h5py.File(output_paramfile, 'w')

    nlayers = len(model.layers)
    h5file['/layers'] = nlayers

    for layeri, layer in enumerate(model.layers):

        layer_string = '/layer{0}/'.format(layeri)

        layer_type = layer.__class__.__name__
        h5file[layer_string + 'type'] = layer_type

        if layer_type == 'MaxoutConvC01B':
            W, b = layer.get_params()
            h5file[layer_string + 'weights'] = W.get_value()
            h5file[layer_string + 'bias'] = b.get_value()
            h5file[layer_string + 'pool_shape'] = layer.pool_shape
            h5file[layer_string + 'pool_stride'] = layer.pool_stride
            h5file[layer_string + 'num_channels'] = layer.num_channels
            h5file[layer_string + 'num_pieces'] = layer.num_pieces
            h5file[layer_string + 'kernel_shape'] = layer.kernel_shape
            h5file[layer_string + 'kernel_stride'] = layer.kernel_stride
        elif layer_type == 'Softmax':
            b, W = layer.get_params()
            h5file[layer_string + 'weights'] = W.get_value()
            h5file[layer_string + 'bias'] = b.get_value()
            h5file[layer_string + 'input_dim'] = layer.input_dim
        else:
            print "Error: Unknown layer type: {0}".format(layer_type)

    h5file.close()

    print 'Wrote network settings to h5 file {0}.'.format(output_paramfile)
