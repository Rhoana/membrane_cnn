# Library for full image cnn operations

import numpy as np

import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray

VALID_SIZE_CROP = False

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = asarray(newsize)
    currsize = array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

gpu_convolve_source = """
__global__ void convolve( float* input, float* filters, float* output,
    int stride, int width, int height, int channels, int filter_width, int filter_height, int nfilters, int output_width, int output_height)
{
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    int oj = blockIdx.y * blockDim.y + threadIdx.y;
    int si = oi % stride;
    int sj = oj % stride;
    int i = oi / stride;
    int j = oj / stride;

    if ( oi < output_width && oj < output_height )
    {
        // Calculate convolution result for output pixel oi, oj with all filters
        for(int filter_index = 0; filter_index < nfilters; ++filter_index )
        {
            float conv_sum = 0;
            // Repeat for all channels
            for(int c = 0; c < channels; ++c)
            {
                for (int fi = 0; fi < filter_width; ++fi)
                {
                    for (int fj = 0; fj < filter_height; ++fj)
                    {
                        if ((i + fi) * stride + si < width && (j + fj) * stride + sj < height)
                        {
                            float in_pix = input[(i + fi) * stride + si + ((j + fj) * stride + sj) * width + c * width * height];
                            float filt_pix = filters[(filter_width - 1 - fi) + (filter_height - 1 - fj) * filter_width + (filter_index * channels + c) * filter_width * filter_height];
                            conv_sum += in_pix * filt_pix;
                        }
                    }
                }
            }
            output[oi + oj * output_width + filter_index * output_width * output_height] = conv_sum;
        }
    }
}
"""

gpu_maxpool2_source = """
__global__ void maxpool2( float* input, float* bias, float* output,
    int stride, int width, int height, int channels )
{
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    int oj = blockIdx.y * blockDim.y + threadIdx.y;
    int si = oi % stride;
    int sj = oj % stride;
    int i = oi / stride;
    int j = oj / stride;

    if ( oi < width && oj < height )
    {
        // Repeat for all channels
       for(int c = 0; c < channels; ++c)
        {
            // Calculate dot product / tanh for pixel i, j
            float max = -1e38;
            for (int mi = i; mi < i + 2; ++mi)
            {
                for (int mj = j; mj < j + 2; ++mj)
                {
                    if ( mi * stride + si < width && mj * stride + sj < height )
                    {
                        float pix = input[mi * stride + si + (mj * stride + sj) * width + c * width * height];
                        if ( pix > max )
                        {
                            max = pix;
                        }
                    }
                }
            }
            output[oi + oj * width + c * width * height] = tanh( (max + bias[c]) );
        }
    }
}
"""

gpu_hidden_layer_source = """
__global__ void hidden_layer( float* input, float* w, float* bias, float* output,
    int stride, int width, int height, int channels, int w_width, int w_height, int nweights, int output_width, int output_height )
{
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    int oj = blockIdx.y * blockDim.y + threadIdx.y;
    int si = oi % stride;
    int sj = oj % stride;
    int i = oi / stride;
    int j = oj / stride;

    if ( oi < output_width && oj < output_height )
    {
        // Calculate dot product result for pixel oi, oj
        for(int w_index = 0; w_index < nweights; ++w_index )
        {
            float dot_product = 0;
            // Repeat for all channels
            for(int c = 0; c < channels; ++c)
            {
                for (int wi = 0; wi < w_width; ++wi)
                {
                    for (int wj = 0; wj < w_height; ++wj)
                    {
                        if ((i + wi) * stride + si < width && (j + wj) * stride + sj < height)
                        {
                            float in_pix = input[(i + wi) * stride + si + ((j + wj) * stride + sj) * width + c * width * height];
                            float filt_pix = w[w_index + wi * nweights + wj * nweights * w_width + c * nweights * w_width * w_height];
                            dot_product += in_pix * filt_pix;
                        }
                    }
                }
            }
            // Apply bias and tanh
            output[oi + oj * output_width + w_index * output_width * output_height] = tanh( (dot_product + bias[w_index]) );
        }
    }
}
"""

gpu_logistic_regression_source = """
__global__ void logistic_regression( float* input, float* w, float* bias, float* output,
    int stride, int width, int height, int channels, int w_width, int w_height, int nweights, int output_width, int output_height )
{
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    int oj = blockIdx.y * blockDim.y + threadIdx.y;
    int si = oi % stride;
    int sj = oj % stride;
    int i = oi / stride;
    int j = oj / stride;

    if ( oi < output_width && oj < output_height )
    {
        // Calculate dot product result for pixel oi, oj
        for(int w_index = 0; w_index < nweights; ++w_index )
        {
            float dot_product = 0;
            // Repeat for all channels
            for(int c = 0; c < channels; ++c)
            {
                for (int wi = 0; wi < w_width; ++wi)
                {
                    for (int wj = 0; wj < w_height; ++wj)
                    {
                        if ((i + wi) * stride + si < width && (j + wj) * stride + sj < height)
                        {
                            float in_pix = input[(i + wi) * stride + si + ((j + wj) * stride + sj) * width + c * width * height];
                            float filt_pix = w[w_index + wi * nweights + wj * nweights * w_width + c * nweights * w_width * w_height];
                            dot_product += in_pix * filt_pix;
                        }
                    }
                }
            }
            // Apply bias
            output[oi + oj * output_width + w_index * output_width * output_height] = (dot_product + bias[w_index]);
        }
    }
}
"""

gpu_logistic_regression_1to1_source = """
__global__ void logistic_regression_1to1( float* input, float* w, float* bias, float* output,
    int width, int height, int channels, int nweights )
{
    int oi = blockIdx.x * blockDim.x + threadIdx.x;
    int oj = blockIdx.y * blockDim.y + threadIdx.y;

    if ( oi < width && oj < height )
    {
        // Calculate dot product result for pixel oi, oj
        for(int w_index = 0; w_index < nweights; ++w_index )
        {
            float dot_product = 0;
            // Repeat for all channels
            for(int c = 0; c < channels; ++c)
            {
                float in_pix = input[oi + oj * width + c * width * height];
                float filt_pix = w[w_index + c * nweights];
                dot_product += in_pix * filt_pix;
            }
            // Apply bias
            output[oi + oj * width + w_index * width * height] = (dot_product + bias[w_index]);
        }
    }
}
"""

gpu_convolve = nvcc.SourceModule(gpu_convolve_source).get_function('convolve')
gpu_maxpool2 = nvcc.SourceModule(gpu_maxpool2_source).get_function('maxpool2')
gpu_hidden_layer = nvcc.SourceModule(gpu_hidden_layer_source).get_function('hidden_layer')
gpu_logistic_regression = nvcc.SourceModule(gpu_logistic_regression_source).get_function('logistic_regression')
gpu_logistic_regression_1to1 = nvcc.SourceModule(gpu_logistic_regression_1to1_source).get_function('logistic_regression_1to1')

class ConvolutionMaxpoolLayer(object):
    def __init__(self, nkernels, ninputs, kernel_size, stride_in, maxpool_size,
        weight_init=0.005, W=[], b=[]):
        self.ninputs = ninputs
        self.nkernels = nkernels
        self.kernel_size = kernel_size
        self.maxpool_size = maxpool_size
        self.stride_in = stride_in
        self.stride_out = stride_in * maxpool_size
        self.prev_conv_size = 0

        if W == []:
            self.W = (np.float32(np.random.random((nkernels, ninputs, kernel_size, kernel_size))) - 0.5) * weight_init * 2
        else:
            self.W = W

        if b == []:
            self.b = np.zeros((nkernels), dtype=np.float32)
        else:
            self.b = b

    def apply_layer(self, input_image=None, d_input_image=None):
        # Calculate feed-forward result
        if d_input_image is None:
            ishape = input_image.shape
            d_input_image = gpuarray.to_gpu(input_image)
        else:
            ishape = d_input_image.shape
            print 'reusing input {0}'.format(ishape)

        print type(d_input_image)

        assert(ishape[0] == self.ninputs)

        d_filters = gpuarray.to_gpu(self.W)

        channels = ishape[0]
        width = ishape[1]
        height = ishape[2]

        if VALID_SIZE_CROP:
            # valid size output
            output_size = (ishape[1] - self.kernel_size + 1, ishape[2] - self.kernel_size + 1)
        else:
            # same size output
            output_size = (ishape[1], ishape[2])

        block = (32, 32, 1)
        grid = (int((output_size[0] - 1) / block[0] + 1), int((output_size[1] - 1) / block[0] + 1))

        out_image = numpy.zeros((self.W.shape[0], output_size[0], output_size[1]), dtype=numpy.float32)
        d_conv_image = gpuarray.to_gpu(out_image)

        gpu_convolve(d_input_image, d_filters, d_conv_image,
            numpy.int32(self.stride_in), numpy.int32(width), numpy.int32(height), numpy.int32(channels),
            numpy.int32(self.W.shape[2]), numpy.int32(self.W.shape[3]), numpy.int32(self.W.shape[0]),
            numpy.int32(output_size[0]), numpy.int32(output_size[1]), block=block, grid=grid)

        # Debug intermeidate result
        #self.layer0_conv = d_conv_image.get()

        d_input_image = None
        d_filters = None

        d_out_image = gpuarray.to_gpu(out_image)
        out_image = None

        d_bias = gpuarray.to_gpu(self.b)

        gpu_maxpool2(d_conv_image, d_bias, d_out_image,
            numpy.int32(self.stride_in), numpy.int32(output_size[0]), numpy.int32(output_size[1]), numpy.int32(self.W.shape[0]), block=block, grid=grid)

        d_bias = None
        d_conv_image = None

        #output = d_out_image.get()
        #d_out_image = None

        #output = np.zeros((self.nkernels, output_size[0], output_size[1]), dtype=np.float32)
        #self.switches = np.zeros((self.nkernels, output_size[0], output_size[1]), dtype=np.uint32)

        print "CONV Layer: Complete ({0} pools).".format(self.stride_in ** 2)

        return d_out_image



class FullyConnectedLayer(object):
    def __init__(self, ninputs, noutputs, kernel_size, stride, weight_init=0.005, W=[], b=[]):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.kernel_size = kernel_size
        self.stride_in = stride
        self.stride_out = stride

        if W == []:
            self.W = (np.float32(np.random.random((ninputs * kernel_size ** 2, noutputs))) - 0.5) * weight_init * 2
        else:
            self.W = W

        if b ==[]:
            self.b = np.zeros((noutputs), dtype=np.float32)
        else:
            self.b = b

    def apply_layer(self, input_image=None, d_input_image=None):
        # Calculate feed-forward result
        if d_input_image is None:
            ishape = input_image.shape
            d_input_image = gpuarray.to_gpu(input_image)
        else:
            ishape = d_input_image.shape

        assert(ishape[0] == self.ninputs)

        if VALID_SIZE_CROP:
            # valid size output
            output_size = (ishape[1] - self.kernel_size + 1, ishape[2] - self.kernel_size + 1)
        else:
            # same size output
            output_size = (ishape[1], ishape[2])

        d_filters = gpuarray.to_gpu(self.W)
        d_bias = gpuarray.to_gpu(self.b)

        stride = self.stride_in
        width = output_size[0]
        height = output_size[1]
        channels = self.ninputs
        nfilters = self.W.shape[1]

        output_width = (width / stride - self.kernel_size + 1) * stride
        output_height = (height / stride - self.kernel_size + 1) * stride

        block = (32, 32, 1)
        grid = (int((output_width - 1) / block[0] + 1), int((output_height - 1) / block[0] + 1))

        out_image = numpy.zeros((nfilters, output_width, output_height), dtype=numpy.float32)
        d_out_image = gpuarray.to_gpu(out_image)
        out_image = None

        gpu_hidden_layer(d_input_image, d_filters, d_bias, d_out_image,
             numpy.int32(stride),  numpy.int32(width),  numpy.int32(height),  numpy.int32(channels),
             numpy.int32(self.kernel_size),  numpy.int32(self.kernel_size),  numpy.int32(nfilters),
             numpy.int32(output_width),  numpy.int32(output_height), block=block, grid=grid)

        d_input_image = None
        d_filters = None
        d_bias = None

        #output = d_out_image.get()
        #d_out_image = None

        print 'FC Layer: Complete ({0} pools)'.format(self.stride_in ** 2)

        return d_out_image



class LogisticRegressionLayer(object):
    def __init__(self, ninputs, noutputs, stride, W=[], b=[]):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.stride_in = stride
        self.stride_out = stride

        if W == []:
            self.W = np.zeros((ninputs, noutputs), dtype=np.float32)
        else:
            self.W = W

        if b ==[]:
            self.b = np.zeros((noutputs), dtype=np.float32)
        else:
            self.b = b

    def apply_layer(self, input_image=None, d_input_image=None):
        # Calculate feed-forward result
        if d_input_image is None:
            ishape = input_image.shape
            d_input_image = gpuarray.to_gpu(input_image)
        else:
            ishape = d_input_image.shape

        assert(ishape[0] == self.ninputs)

        d_filters = gpuarray.to_gpu(self.W)
        d_bias = gpuarray.to_gpu(self.b)

        stride = self.stride_in
        width = ishape[1]
        height = ishape[2]
        channels = ishape[0]
        filter_width = 1
        filter_height = 1
        nfilters = self.W.shape[1]

        # output_width = (width / stride - filter_width + 1) * stride
        # output_height = (height / stride - filter_height + 1) * stride

        block = (32, 32, 1)
        #grid = (int((output_width - 1) / block[0] + 1), int((output_height - 1) / block[0] + 1))
        grid = (int((width - 1) / block[0] + 1), int((height - 1) / block[0] + 1))

        out_image = np.zeros((self.noutputs, ishape[1], ishape[2]), dtype=np.float32)
        d_out_image = gpuarray.to_gpu(out_image)
        out_image = None

        # gpu_logistic_regression(d_input_image, d_filters, d_bias, d_out_image,
        #      numpy.int32(stride),  numpy.int32(width),  numpy.int32(height),  numpy.int32(channels),
        #      numpy.int32(filter_width),  numpy.int32(filter_height),  numpy.int32(nfilters),
        #      numpy.int32(output_width),  numpy.int32(output_height), block=block, grid=grid)
        gpu_logistic_regression_1to1(d_input_image, d_filters, d_bias, d_out_image,
             numpy.int32(width),  numpy.int32(height),  numpy.int32(channels),
             numpy.int32(nfilters), block=block, grid=grid)

        d_input_image = None
        d_filters = None
        d_bias = None

        output = d_out_image.get()
        d_out_image = None

        self.pre_softmax = output

        #Apply softmax
        maxes = np.amax(output, axis=0)
        maxes = np.tile(maxes, (2,1,1))
        e = np.exp(output - maxes)
        output = e / np.sum(e, axis=0)

        print 'LR Layer: Complete.'

        return output
