import numpy as N
np = N
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.utils import serial
import cPickle
import gzip

class LGN(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, imgd=65, zd=1, ds=1, center = False, shuffle = False,
            one_hot = False, binarize = False, start = None,
            stop = None, axes=['b', 0, 1, 'c'],
            preprocessor = None,
            fit_preprocessor = False,
            fit_test_preprocessor = False):

        self.args = locals()

        if which_set not in ['train','valid','test']:
            raise ValueError('Unrecognized which_set value "%s".' %
                    (which_set,)+'". Valid values are ["train","valid",test"].')

        def dimshuffle(b01c):
            default = ('b', 0, 1, 'c')
            return b01c.transpose(*[default.index(axis) for axis in axes])

        if control.get_load_data():

            path = "${PYLEARN2_DATA_PATH}/lgn/"
            path = path + "LGN1_MembraneSamples_65x65x1_mp0.50_train50000_valid10000_test10000_seed11.pkl.gz"
            path = serial.preprocess(path)

            f = gzip.open(path, 'rb')
            train_set, valid_set, test_set = cPickle.load(f)
            f.close()

            if which_set == 'train':
                data = train_set
            elif which_set == 'valid':
                data = valid_set
            else:
                data = test_set

            input_shape = (imgd, imgd, zd)

            # f = h5py.file(path, 'r')
            # input_shape = f['input_shape'][...]

            # if which_set == 'train':
            #     data = f['/train_set'][...]
            # elif which_set == 'valid':
            #     data = f['/valid_set'][...]
            # else:
            #     data = f['/test_set'][...]

            # Convert images to float 0-1
            topo_view = data[0].astype(np.float32) / 255.0
            y = data[1]

            self.one_hot = one_hot
            if one_hot:
                one_hot = N.zeros((y.shape[0],2),dtype='float32')
                for i in xrange(y.shape[0]):
                    one_hot[i,y[i]] = 1.
                y = one_hot

            m = topo_view.shape[0]
            rows, cols, slices = input_shape
            topo_view = topo_view.reshape(m, rows, cols, slices)

            if center:
                topo_view -= topo_view.mean(axis=0)

            if shuffle:
                self.shuffle_rng = np.random.RandomState([1,2,3])
                for i in xrange(topo_view.shape[0]):
                    j = self.shuffle_rng.randint(m)
                    # Copy ensures that memory is not aliased.
                    tmp = topo_view[i,:,:,:].copy()
                    topo_view[i,:,:,:] = topo_view[j,:,:,:]
                    topo_view[j,:,:,:] = tmp
                    # Note: slicing with i:i+1 works for both one_hot=True/False.
                    tmp = y[i:i+1].copy()
                    y[i] = y[j]
                    y[j] = tmp

            super(LGN,self).__init__(topo_view = dimshuffle(topo_view), y = y, axes=axes)

            assert not N.any(N.isnan(self.X))

            if start is not None:
                assert start >= 0
                if stop > self.X.shape[0]:
                    raise ValueError('stop='+str(stop)+'>'+'m='+str(self.X.shape[0]))
                assert stop > start
                self.X = self.X[start:stop,:]
                if self.X.shape[0] != stop - start:
                    raise ValueError("X.shape[0]: %d. start: %d stop: %d" % (self.X.shape[0], start, stop))
                if len(self.y.shape) > 1:
                    self.y = self.y[start:stop,:]
                else:
                    self.y = self.y[start:stop]
                assert self.y.shape[0] == stop - start
        else:
            #data loading is disabled, just make something that defines the right topology
            topo = dimshuffle(np.zeros((1,65,65,1)))
            super(LGN,self).__init__(topo_view = topo, axes=axes)
            self.X = None

        if which_set == 'test':
            assert fit_test_preprocessor is None or (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)

    def adjust_for_viewer(self, X):
        return N.clip(X*2.-1.,-1.,1.)

    def adjust_to_be_viewed_with(self, X, other, per_example = False):
        return self.adjust_for_viewer(X)

    def get_test_set(self):
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        args['start'] = None
        args['stop'] = None
        args['fit_preprocessor'] = args['fit_test_preprocessor']
        args['fit_test_preprocessor'] = None
        return LGN(**args)
