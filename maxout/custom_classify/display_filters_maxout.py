import sys
import glob
import h5py

import hashlib
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

input_path = sys.argv[1]

h5file = h5py.File(input_path, 'r')

nlayers = h5file['/layers'][...]

for layeri in range(1):

    layer_string = '/layer{0}/'.format(layeri)

    layer_type = h5file[layer_string + 'type'][...]

    W = h5file[layer_string + 'weights'][...]
    b = h5file[layer_string + 'bias'][...]

    print W.shape

    if len(W.shape) == 4:
        for f in range(W.shape[0]):
            for c in range(W.shape[3]):
                plt.imshow(W[f,:,:,c], cmap=cm.gray)
                plt.show()

h5file.close()
