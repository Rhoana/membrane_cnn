from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import numpy as np
import Image

def normalize_image_float(original_image, saturation_level=0.005):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    return norm_image / 255.0

_, model_path, img_in, img_out = sys.argv

model = serial.load(model_path)

input_image = normalize_image_float(np.array(Image.open(img_in)))
nx, ny = input_image.shape

input_shape = model.input_space.shape
pad_by = np.max(input_shape) / 2
pad_image = np.pad(input_image, ((pad_by, pad_by), (pad_by, pad_by)), 'symmetric')

batch_size = 1024
model.set_batch_size(batch_size)

import theano.tensor as T

Xb = model.get_input_space().make_batch_theano()
Xb.name = 'Xb'
yb = model.get_output_space().make_batch_theano()
yb.name = 'yb'

ymf = model.fprop(Xb)
ymf.name = 'ymf'

from theano import function
batch_predict = function([Xb],[ymf])

# yl = T.argmax(yb,axis=1)
# mf1acc = 1.-T.neq(yl , T.argmax(ymf,axis=1)).mean()
# batch_acc = function([Xb,yb],[mf1acc])

X = np.zeros((1, input_shape[0], input_shape[1], batch_size), dtype=np.float32)
y = np.zeros(nx * ny, dtype=np.float32)
output_image = np.zeros((nx, ny), dtype=np.float32)

batch_count = 0
batch_start = 0
batchi = 0

assert isinstance(X.shape[0], (int, long))
assert isinstance(batch_size, py_integer_types)

for rotate in range(4):

    if rotate == 0:
        rotate_input = pad_image
    elif rotate == 1:
        rotate_input = pad_image[::-1,:].dimshuffle(1,0)
    elif rotate == 2:
        rotate_input = pad_image[::-1,:][:,::-1]
    else:
        rotate_input = pad_image.dimshuffle(1,0)[::-1,:]

    for xi in range(nx):
        for yi in range(ny):

            X[0, :, :, batchi] = pad_image[xi : xi + input_shape[0], yi : yi + input_shape[1]]

            batchi += 1

            if batchi == batch_size:
                # Classify and reset
                y[batch_start:batch_start + batch_size] = batch_predict(X)[0][:,0]
                batch_count += 1
                batch_start += batch_size
                batchi = 0

                print "Batch {0} done. Up to {1}.".format(batch_count, (rotate, xi, yi))

    if batchi > 0:
        y[batch_start:batch_start + batchi] = batch_predict(X)[0][:batchi,0]

    if rotate == 0:
        output_image += y.reshape(input_image.shape)
    elif rotate == 1:
        output_image += y.reshape(input_image.shape).dimshuffle(1,0)[::-1,:]
    elif rotate == 2:
        output_image += y.reshape(input_image.shape)[::-1,:][:,::-1]
    else:
        output_image += y.reshape(input_image.shape)[::-1,:].dimshuffle(1,0)

output_image = output_image / 4.0

im = Image.fromarray(np.uint8(output_image * 255))
im.save(img_out)

print "Image saved."

import h5py
f = h5py.File(img_out.replace('.tif', '') + '.h5')
f['/probabilities'] = output_image
f.close()

print "Probabilities saved."
