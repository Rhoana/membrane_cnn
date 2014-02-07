import time
import sys
import numpy as np
import Image

#from lib_maxout_gpu import *
from lib_maxout_python import *
#from lib_maxout_theano import *
#from lib_maxout_theano_batch import *

def normalize_image_float(original_image, saturation_level=0.005):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    return norm_image / 255.0

_, model_path, img_in, img_out = sys.argv[0:4]

# For lib_maxout_theano_batch we can control batch size
# batch_size = 1024
# if len(sys.argv) > 4:
#     batch_size = int(sys.argv[4])
# network = DeepNetwork(model_path, batch_size=batch_size)

network = DeepNetwork(model_path)

input_image = normalize_image_float(np.array(Image.open(img_in)))
nx, ny = input_image.shape

pad_by = network.pad_by
pad_image = np.pad(input_image, ((pad_by, pad_by), (pad_by, pad_by)), 'symmetric')

start_time = time.time()

output = network.apply_net(pad_image, perform_pad=False)

print 'Complete in {0:1.4f} seconds'.format(time.time() - start_time)

im = Image.fromarray(np.uint8(output * 255))
im.save(img_out)

print "Image saved."

import h5py
f = h5py.File(img_out.replace('.tif', '') + '.h5')
f['/probabilities'] = output
f.close()

print "Probabilities saved."
