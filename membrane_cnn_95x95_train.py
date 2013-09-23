"""
Membrane classifier, based on Theano LeNet5 convolutional neural network:
http://deeplearning.net/tutorial/lenet.html

Training images generated from the ISBI 2013 challenge: 3D segmentation of neurites in EM images

Original images:
http://brainiac2.mit.edu/SNEMI3D/

Training datasets:
http://people.seas.harvard.edu/~seymourkb/TrainingData/

"""

import cPickle
import gzip
import os
import os.path
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.005, n_epochs=8000,
                    dataset='MembraneSamples_95x95x1_mp0.50_train5000_valid1000_test1000.pkl.gz',
                    nkerns=[32, 32, 32, 32], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    #nsplits = 10
    #current_split = 0
    #datasets = load_data(dataset.format(current_split, nsplits))

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (95, 95)  # this is the size of white and black patches

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 95*95)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 95, 95))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (95-4+1, 95-4+1)=(92, 92)
    # maxpooling reduces this further to (92/2, 92/2) = (46, 46)
    # 4D output tensor is thus of shape (batch_size,nkerns[0], 46, 46)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 95, 95),
            filter_shape=(nkerns[0], 1, 4, 4), poolsize=(2, 2))
    
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (46-5+1, 46-5+1)=(42,42)
    # maxpooling reduces this further to (42/2, 42/2) = (21,21)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1], 21, 21)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape = (batch_size, nkerns[0], 46, 46),
            filter_shape = (nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (21-4+1, 21-4+1) = (18, 18)
    # maxpooling reduces this further to (18/2, 18/2) = (9, 9)
    # 4D output tensor is thus of shape (nkerns[1], nkerns[2], 9, 9)
    layer2 = LeNetConvPoolLayer(rng, input=layer1.output,
            image_shape=(batch_size, nkerns[1], 21, 21),
            filter_shape=(nkerns[2], nkerns[1], 4, 4), poolsize=(2, 2))

    # Construct the fourth convolutional pooling layer
    # filtering reduces the image size to (9-4+1, 9-4+1) = (6, 6)
    # maxpooling reduces this further to (6/2, 6/2) = (3, 3)
    # 4D output tensor is thus of shape (nkerns[2], nkerns[3], 3,3)
    layer3 = LeNetConvPoolLayer(rng, input=layer2.output,
            image_shape=(batch_size, nkerns[2], 9, 9),
            filter_shape=(nkerns[3], nkerns[2], 4, 4), poolsize=(2, 2))

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer4_input = layer3.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer4 = HiddenLayer(rng, input=layer4_input, n_in=nkerns[3] * 3 * 3,
                         n_out=100, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer5 = LogisticRegression(input=layer4.output, n_in=100, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer5.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer5.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer5.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    

    ##### Attempt to load a progress file #####

    epoch = 0
    best_validation_loss = numpy.inf
    
    outfile = dataset.replace('.pkl.gz', '.progress.pkl.gz')
    if os.path.isfile(outfile):
        f = gzip.open(outfile, 'rb')
        iter, best_params, this_validation_loss, test_score = cPickle.load(f)
        f.close()
        epoch = numpy.floor(iter / n_train_batches)
        best_validation_loss = this_validation_loss
        layer5.W.set_value(best_params[0][0].get_value())
        layer5.b.set_value(best_params[0][1].get_value())
        layer4.W.set_value(best_params[1][0].get_value())
        layer4.b.set_value(best_params[1][1].get_value())
        layer3.W.set_value(best_params[2][0].get_value())
        layer3.b.set_value(best_params[2][1].get_value())
        layer2.W.set_value(best_params[3][0].get_value())
        layer2.b.set_value(best_params[3][1].get_value())
        layer1.W.set_value(best_params[4][0].get_value())
        layer1.b.set_value(best_params[4][1].get_value())
        layer0.W.set_value(best_params[5][0].get_value())
        layer0.b.set_value(best_params[5][1].get_value())
        print 'Loaded progress file. Up to epoch {0}, validation error {1}, test error {2}.'.format(epoch, this_validation_loss * 100, test_score * 100)

    # create a list of all model parameters to be fit by gradient descent
    params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates dictionary by automatically looping over all
    # (params[i],grads[i]) pairs.

    #updates = {}
    #for param_i, grad_i in zip(params, grads):
    #    updates[param_i] = param_i - learning_rate * grad_i
    
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look at this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = epoch * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:                                                                          
                
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_params = (layer5.params, layer4.params, layer3.params, layer2.params, layer1.params, layer0.params)

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    f = gzip.open(outfile,'wb', compresslevel=1)
                    cPickle.dump((iter, best_params, this_validation_loss, test_score),f)
                    f.close()
                    print 'Progress saved.'

            if patience <= iter:
                done_looping = True
                break

        #Load a new dataset split after each epoch

        # current_split = (current_split + 1) % nsplits

        # datasets = load_data(dataset.format(current_split, nsplits))

        # train_set_x, train_set_y = datasets[0]
        # valid_set_x, valid_set_y = datasets[1]
        # test_set_x, test_set_y = datasets[2]

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))


    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    return best_params

if __name__ == '__main__':
    best_params = evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
