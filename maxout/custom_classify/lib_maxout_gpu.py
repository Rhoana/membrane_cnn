# Library for full image cnn operations

import numpy as np
import scipy.ndimage
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from numpy.fft import rfftn
from numpy.fft import irfftn
import mahotas
import time
import h5py

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray

BLOCK_BATCHES = 1024
BLOCK_PIXELS = 1

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

gpu_maxout_layer_source = """
__global__ void maxout_layer( float* input, float* filters, float* bias, float* output,
    int batches, int channels, int width, int height,
    int nfilters, int filter_width, int filter_height,
    int output_width, int output_height,
    int maxout_size, int maxpool_size)
{
    //int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
    int ochannel_index = blockIdx.x * blockDim.x + threadIdx.x;
    int oi = blockIdx.y * blockDim.y + threadIdx.y;
    int oj = blockIdx.z * blockDim.z + threadIdx.z;

    int conv_size = (width - filter_width + 1);
    int conv_size2 = conv_size * conv_size;
    int wh = width * height;
    int input_batchsize = wh * channels;
    int filter_wh = filter_width * filter_height;
    int output_wh = output_width * output_height;
    int output_batchsize = output_wh * (nfilters / maxout_size);

    int start_filter = ochannel_index / maxout_size;
    int end_filter = start_filter + maxout_size - 1;

    if (ochannel_index < nfilters / maxout_size && oi < output_width && oj < output_height)
    {

        for (int batch_index = 0; batch_index < batches; ++batch_index)
        {

                float current_max;

                // Calculate convolution result for output pixel oi, oj with all filters
                for(int filter_index = start_filter; filter_index <= end_filter; ++filter_index )
                {
                    // Maxpool region
                    for (int i = oi * maxpool_size; i < (oi + 1) * maxpool_size; ++i)
                    {
                        for (int j = oj * maxpool_size; j < (oj + 1) * maxpool_size; ++j)
                        {

                            float conv_sum = 0;

                            // Convolve for all channels
                            for(int c = 0; c < channels; ++c)
                            {
                                for (int fi = 0; fi < filter_width; ++fi)
                                {
                                    for (int fj = 0; fj < filter_height; ++fj)
                                    {
                                        if (i + fi < width && j + fj < height)
                                        {
                                            float in_pix = input[(i + fi) + (j + fj) * width + c * wh + batch_index * input_batchsize];
                                            float filt_pix = filters[fi + fj * filter_width + (filter_index * channels + c) * filter_wh];
                                            conv_sum += in_pix * filt_pix;
                                        }
                                    }
                                }
                            }

                            // Add pixel-wise bias
                            conv_sum += bias[i + j * conv_size + filter_index * conv_size2];

                            // Maxout across channels and maxpool across pixels
                            if (((filter_index % maxout_size == 0) && (i % maxpool_size == 0) && (j % maxpool_size == 0)) ||
                                (conv_sum > current_max))
                            {
                                current_max = conv_sum;
                            }

                        }
                    }

                    if (filter_index % maxout_size == maxout_size - 1)
                    {
                        output[oi + oj * output_width + (filter_index / maxout_size) * output_wh + batch_index * output_batchsize] = current_max;
                    }
                }

        }

    }
}
"""

gpu_softmax_layer_source = """
__global__ void softmax_layer( float* input, float* filters, float* bias, float* output,
    int batches, int channels, int width, int height,
    int nfilters, int filter_size,
    int output_width, int output_height)
{
    int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
    int oi = blockIdx.y * blockDim.y + threadIdx.y;
    int oj = blockIdx.z * blockDim.z + threadIdx.z;

    int wh = width * height;
    int input_batchsize = wh * channels;
    int output_wh = output_width * output_height;
    int output_batchsize = output_wh * nfilters;

    if (batch_index < batches && oi < output_width && oj < output_height)
    {
        float current_max;

        for(int filter_index = 0; filter_index < nfilters; ++filter_index )
        {
            float dot_product = 0;

            // Calculate dot product for output pixel oi, oj
            for (int fi = 0; fi < filter_size; ++fi)
            {
                for (int fj = 0; fj < filter_size; ++fj)
                {
                    for(int c = 0; c < channels; ++c)
                    {
                        float in_pix = input[(oi + fi) + (oj + fj) * width + c * wh + batch_index * input_batchsize];
                        float filt_pix = filters[filter_index + c * nfilters + fi * channels * nfilters + fj * filter_size * channels * nfilters];
                        dot_product += in_pix * filt_pix;
                    }
                }
            }

            dot_product += bias[filter_index];

            if ((filter_index == 0) || (dot_product > current_max))
            {
                current_max = dot_product;
            }

            output[oi + oj * output_width + filter_index * output_wh + batch_index * output_batchsize] = dot_product;

        }

        // Softmax

        float esum = 0;

        for(int filter_index = 0; filter_index < nfilters; ++filter_index )
        {
            float softout = output[oi + oj * output_width + filter_index * output_wh + batch_index * output_batchsize];
            softout = __expf(softout - current_max);
            //softout = expf(softout - current_max);
            esum += softout;
            output[oi + oj * output_width + filter_index * output_wh + batch_index * output_batchsize] = softout;
        }

        for(int filter_index = 0; filter_index < nfilters; ++filter_index )
        {
            output[oi + oj * output_width + filter_index * output_wh + batch_index * output_batchsize] /= esum;
        }

    }
}
"""

gpu_maxout_layer = nvcc.SourceModule(gpu_maxout_layer_source).get_function('maxout_layer')
gpu_softmax_layer = nvcc.SourceModule(gpu_softmax_layer_source).get_function('softmax_layer')

class MaxoutMaxpoolLayer(object):
    def __init__(self, nkernels, ninputs, kernel_size, stride_in, maxpool_size, maxout_size, W, b):
        self.ninputs = ninputs
        self.nkernels = nkernels
        self.kernel_size = kernel_size
        self.maxpool_size = maxpool_size
        self.maxout_size = maxout_size
        self.stride_in = stride_in
        self.stride_out = stride_in
        self.noutputs = nkernels / maxpool_size
        # Size of previous convolution operation (for fft result cache)
        self.prev_conv_size = 0
        # Input / output footprint - set once full network has been constructed
        self.input_footprint = 0
        self.output_footprint = 0

        self.W = gpuarray.to_gpu(W.copy())
        self.b = gpuarray.to_gpu(b)

    def apply_layer(self, input_image, nbatches):

        # start with convoludion output size (before maxout and maxpool operations)
        output_size = (nbatches, self.noutputs, self.output_footprint, self.output_footprint)
        print output_size

        block = (int(self.noutputs), 4, 4)
        grid = (int((self.noutputs - 1) / block[0] + 1), int((self.input_footprint - 1) / block[1] + 1), int((self.input_footprint - 1) / block[2] + 1))

        if not isinstance(input_image, gpuarray.GPUArray):
            input_image = gpuarray.to_gpu(input_image)

        d_maxout_result = gpuarray.zeros(long(np.prod(output_size)), np.float32).reshape(output_size)

        gpu_maxout_layer(input_image, self.W, self.b, d_maxout_result,
            np.int32(nbatches), np.int32(self.ninputs), np.int32(self.input_footprint), np.int32(self.input_footprint),
            np.int32(self.W.shape[0]), np.int32(self.W.shape[2]), np.int32(self.W.shape[3]),
            np.int32(output_size[2]), np.int32(output_size[3]),
            np.int32(self.maxout_size), np.int32(self.maxpool_size),
            block=block, grid=grid)

        print "MO Layer: Complete."

        return d_maxout_result


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

        self.W = gpuarray.to_gpu(W)
        self.b = gpuarray.to_gpu(b)

    def apply_layer(self, input_image, nbatches):

        # Calculate feed-forward result
        output_size = (nbatches, self.noutputs, self.output_footprint, self.output_footprint)
        print output_size

        block = (BLOCK_BATCHES, BLOCK_PIXELS, BLOCK_PIXELS)
        grid = (int((nbatches - 1) / block[0] + 1), int((self.input_footprint - 1) / block[1] + 1), int((self.input_footprint - 1) / block[2] + 1))

        if not isinstance(input_image, gpuarray.GPUArray):
            input_image = gpuarray.to_gpu(input_image)

        d_softmax_result = gpuarray.zeros(long(np.prod(output_size)), np.float32).reshape(output_size)

        gpu_softmax_layer(input_image, self.W, self.b, d_softmax_result,
            np.int32(nbatches), np.int32(self.ninputs), np.int32(self.input_footprint), np.int32(self.input_footprint),
            np.int32(self.W.shape[1]), np.int32(self.input_footprint),
            np.int32(output_size[2]), np.int32(output_size[3]),
            block=block, grid=grid)

        print "SM Layer: Complete."

        return d_softmax_result


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
                layer_maxpoolsize = network_h5[layer_string + 'pool_shape'][...][0]
                layer_maxoutsize = network_h5[layer_string + 'num_pieces'][...]

                # Arrange weights as [kernels, inputs, ksize, ksize]
                layer_weights = np.rollaxis(layer_weights, 3, 0)

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
        print layer_temp.shape

        batchi = 0
        for x in range(nx):
            for y in range(ny):
                #print (x,y)
                layer_temp[batchi, :, :, :] = input_image[x:(x + self.all_layers[0].input_footprint), y:(y + self.all_layers[0].input_footprint)]
                batchi += 1

        assert batchi == nbatches

        output = np.zeros(nbatches, dtype=np.float32)

        for block in range(nbatches / BLOCK_BATCHES + 1):

            block_from = block * BLOCK_BATCHES
            block_to = min((block+1) * BLOCK_BATCHES, layer_temp.shape[0])
            nbatches = block_to - block_from

            block_temp = layer_temp[block_from:block_to,:,:,:]

            for layeri in range(len(self.all_layers)):
                print layeri
                start_time = time.clock()
                block_temp = self.all_layers[layeri].apply_layer(block_temp, nbatches)
                end_time = time.clock()
                print('Layer time = %.2fm' % ((end_time - start_time) / 60.))

            if isinstance(block_temp, gpuarray.GPUArray):
                block_temp = block_temp.get()

            output[block_from:block_to] = block_temp[:,0,0,0]

        output = output.reshape(nx, ny)

        if perform_upsample:
            output = np.float32(mahotas.imresize(output, self.downsample))

        if perform_blur and self.best_sigma != 0:
            output = scipy.ndimage.filters.gaussian_filter(output, self.best_sigma)

        if perform_offset:
            #Translate
            output = np.roll(output, self.best_offset[0], axis=0)
            output = np.roll(output, self.best_offset[1], axis=1)

        # Crop to valid size
        #output = output[self.pad_by:-self.pad_by,self.pad_by:-self.pad_by]

        return output
