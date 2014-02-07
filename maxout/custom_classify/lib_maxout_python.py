# Library for full image cnn operations

import numpy as np
import scipy.ndimage
#from scipy.signal import convolve2d
#from scipy.signal import fftconvolve
#from numpy.fft import rfftn
#from numpy.fft import irfftn
import mahotas
import time
import h5py

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

class MaxoutMaxpoolLayer(object):
    def __init__(self, nkernels, ninputs, kernel_size, stride_in, maxpool_size, maxout_size, W, b):
        self.ninputs = ninputs
        self.nkernels = nkernels
        self.kernel_size = kernel_size
        self.maxpool_size = maxpool_size
        self.maxout_size = maxout_size
        self.stride_in = stride_in
        self.stride_out = stride_in * maxpool_size
        self.noutputs = nkernels / maxpool_size
        # Size of previous convolution operation (for fft result cache)
        self.prev_conv_size = 0
        # Input / output footprint - set once full network has been constructed
        self.input_footprint = 0
        self.output_footprint = 0

        self.W = W
        self.b = b

    def apply_layer(self, input_image):

        # Calculate feed-forward result
        assert(input_image.shape[1] == self.ninputs)

        #output = np.zeros(output_size, dtype=np.float32)
        output = np.tile(self.b, (input_image.shape[0], 1, 1, 1))

        crop_low = (self.kernel_size - 1) / 2
        crop_high = (self.kernel_size) / 2

        for batchi in range(input_image.shape[0]):

            # Apply convolution

            for channeli in range(self.ninputs):

                channel_input = input_image[batchi, channeli, :, :]
                channel_filters = self.W[:,channeli,:,:]

                for filteri in range(self.nkernels):

                    # Space domain convolution (ndimage)
                    output[batchi, filteri, :, :] += scipy.ndimage.convolve(
                       channel_input,
                       channel_filters[filteri,:,:],
                       mode='constant')[crop_low:-crop_high, crop_low:-crop_high]

                #output[batchi, filteri, :, :] += self.b[filteri, :, :]

            if batchi % 100 == 99:
                print "MO Layer: Convolution batch {0}, of {1} complete.".format(batchi + 1, input_image.shape[0])

        # Apply maxout
        if self.maxout_size != 1:
            maxout_temp = None
            for i in xrange(self.maxout_size):
                this_slice = output[:,i::self.maxout_size,:,:]
                if maxout_temp is None:
                    maxout_temp = this_slice
                else:
                    maxout_temp = np.maximum(maxout_temp, this_slice)
            output = maxout_temp
            print "MO Layer: Applied maxout."

        # Apply maxpool
        if self.maxpool_size != 1:
            maxpool_temp = None
            for offset_x in range(self.maxpool_size):
                for offset_y in range(self.maxpool_size):
                    this_slice = output[:, :, offset_x::self.maxpool_size, offset_y::self.maxpool_size]
                    if maxpool_temp is None:
                        maxpool_temp = this_slice
                    else:
                        maxpool_temp = np.maximum(maxpool_temp, this_slice)
            output = maxpool_temp
            print "MO Layer: Applied maxpool."

        print "MO Layer: Complete."

        return output


class SoftmaxLayer(object):
    def __init__(self, ninputs, noutputs, kernel_size, stride, W, b):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.kernel_size = kernel_size
        self.stride_in = stride
        self.stride_out = stride
        # Input / output footprint - set once full network has been constructed
        self.input_footprint = 0
        self.output_footprint = 0
        
        self.W = W
        self.b = b

    def apply_layer(self, input_image):
        # Calculate feed-forward result
        assert(input_image.shape[1] == self.ninputs)

        nbatches = input_image.shape[0]
        output_size = (nbatches, self.noutputs, input_image.shape[2] - self.kernel_size + 1, input_image.shape[3] - self.kernel_size + 1)

        output = np.zeros(output_size, dtype=np.float32)

        for batchi in range(nbatches):
            # Apply dot product
            rolled_input = np.rollaxis(np.rollaxis(input_image[batchi, :, :, :], 2), 2)
            output[batchi, :, :, :] = (np.dot(rolled_input.flatten(), self.W) + self.b).reshape(output_size[1:4])

            if batchi % 100 == 99:
                print "SM Layer: Done batch {0}, of {1}.".format(batchi + 1, nbatches)

        #Apply softmax
        #print output
        maxes = np.amax(output, axis=1).reshape((nbatches, 1, output_size[2], output_size[3]))
        maxes = np.tile(maxes, (1,2,1,1))
        e = np.exp(output - maxes)
        esum = np.sum(e, axis=1).reshape((nbatches, 1, output_size[2], output_size[3]))
        esum = np.tile(esum, (1,2,1,1))
        output = e / esum

        print "SM Layer: Complete."

        return output


class DeepNetwork(object):
    def __init__(self, filename):

        network_h5 = h5py.File(filename, 'r')

        self.nlayers = network_h5['/layers'][...]

        print 'Network has {0} layers.'.format(self.nlayers)

        if '/downsample_factor' in network_h5:
            self.downsample = network_h5['/downsample_factor'][...]
        else:
            self.downsample = 1

        self.best_sigma = 0
        self.best_offset = (0,0)

        all_layers = []
        stride_in = 1

        for layer_i in range(self.nlayers):

            layer_string = '/layer{0}/'.format(layer_i)
            layer_type = network_h5[layer_string + 'type'][...]

            if layer_type == 'MaxoutConvC01B':

                layer_weights = network_h5[layer_string + 'weights'][...]
                layer_bias = network_h5[layer_string + 'bias'][...]
                layer_maxpoolsize = int(network_h5[layer_string + 'pool_shape'][...][0])
                layer_maxoutsize = int(network_h5[layer_string + 'num_pieces'][...])

                # Arrange weights as [kernels, inputs, ksize, ksize]
                layer_weights = np.rollaxis(layer_weights, 3, 0)[:,:,::-1,::-1]

                new_layer = MaxoutMaxpoolLayer(
                    layer_weights.shape[0], layer_weights.shape[1], layer_weights.shape[2],
                    stride_in, layer_maxpoolsize, layer_maxoutsize, W=layer_weights, b=layer_bias)

            elif layer_type == 'Softmax':

                layer_weights = network_h5[layer_string + 'weights'][...]
                layer_bias = network_h5[layer_string + 'bias'][...]
                layer_ksize = network_h5[layer_string + 'ksize'][...][0]

                new_layer = SoftmaxLayer(
                    layer_weights.shape[0] / (layer_ksize ** 2), layer_weights.shape[1], layer_ksize,
                    stride_in, W=layer_weights, b=layer_bias)

            else:
                raise Exception("Unknown layer type: {0}".format(layer_type))

            all_layers.append(new_layer)

            stride_in = new_layer.stride_out

        # Calculate network footprint and therefore pad size
        footprint = 1
        for layer in range(self.nlayers-1, -1, -1):
            all_layers[layer].output_footprint = footprint
            if layer == self.nlayers - 1:
                footprint = all_layers[layer].kernel_size
            else:
                footprint = footprint * all_layers[layer].maxpool_size - 1 + all_layers[layer].kernel_size
            all_layers[layer].input_footprint = footprint

        self.all_layers = all_layers
        self.pad_by = int(self.downsample * (footprint // 2))


    def apply_net(self, input_image, perform_downsample=False, perform_pad=False, perform_upsample=False, perform_blur=False, perform_offset=False):

        if perform_pad:
            input_image = np.pad(input_image, ((self.pad_by, self.pad_by), (self.pad_by, self.pad_by)), 'symmetric')

        if perform_downsample and self.downsample != 1:
            input_image = np.float32(mahotas.imresize(input_image, 1.0/self.downsample))

        nx = input_image.shape[0] - self.all_layers[0].input_footprint + 1
        ny = input_image.shape[1] - self.all_layers[0].input_footprint + 1
        nbatches = nx * ny
        layer_temp = np.zeros((nbatches, 1, self.all_layers[0].input_footprint, self.all_layers[0].input_footprint), dtype=np.float32)

        batchi = 0
        for x in range(nx):
            for y in range(ny):
                #print (x,y)
                layer_temp[batchi, :, :, :] = input_image[x:(x + self.all_layers[0].input_footprint), y:(y + self.all_layers[0].input_footprint)]
                batchi += 1

        assert batchi == nbatches

        for layeri in range(len(self.all_layers)):
            print 'Layer {0}.'.format(layeri)
            layer_temp = self.all_layers[layeri].apply_layer(layer_temp)

        output_image = layer_temp[:,0,0,0].reshape(nx, ny)

        if perform_upsample:
            output_image = np.float32(mahotas.imresize(output_image, self.downsample))

        if perform_blur and self.best_sigma != 0:
            output_image = scipy.ndimage.filters.gaussian_filter(output_image, self.best_sigma)

        if perform_offset:
            #Translate
            output_image = np.roll(output_image, self.best_offset[0], axis=0)
            output_image = np.roll(output_image, self.best_offset[1], axis=1)

        # Crop to valid size
        #output_image = output_image[self.pad_by:-self.pad_by,self.pad_by:-self.pad_by]

        return output_image
