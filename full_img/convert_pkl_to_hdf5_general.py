import cPickle
import gzip

import glob
import h5py

import hashlib
import shutil

import numpy as np

DEFAULT_MAXPOOL_SIZE = 2
input_path = 'D:/dev/Rhoana/membrane_cnn/results/stumpin/'
#input_path = 'D:/dev/Rhoana/membrane_cnn/results/PC/'
#input_path = 'D:/dev/Rhoana/membrane_cnn/results/resonance/progress2/'
#input_path = 'D:/dev/Rhoana/membrane_cnn/full_img/'

COPY_GOOD_RESULTS = False
good_output_path = 'D:/dev/Rhoana/membrane_cnn/results/good2/'

param_files = glob.glob(input_path + '*.pkl.gz')

for input_paramfile in param_files:

    output_paramfile=input_paramfile.replace('.pkl.gz', '.h5')

    print 'Opening file {0}.'.format(input_paramfile)

    f = gzip.open(input_paramfile, 'rb')
    packed = cPickle.load(f)
    if len(packed) == 2:
        iterations, best_params = packed
        this_validation_loss = 0.0
        test_score = 0.0
    else:
        iterations, best_params, this_validation_loss, test_score = packed
    f.close()

    print 'Loaded progress file.'

    print 'Trained up to interation {0}, v.error {1}, t.error {2}.'.format(iterations, this_validation_loss, test_score)

    h5file = h5py.File(output_paramfile, 'w')

    nlayers = len(best_params)
    h5file['/layers'] = nlayers
    h5file['/iterations'] = iterations
    h5file['/verror'] = this_validation_loss
    h5file['/terror'] = test_score

    for revlayer, params in enumerate(best_params):

        layeri = nlayers - revlayer - 1
        layer_string = '/layer{0}/'.format(layeri)

        if revlayer == 0:
            h5file[layer_string + 'type'] = 'LogisticRegression'
        elif revlayer == 1:
            h5file[layer_string + 'type'] = 'FullyConnected'
            # Determine k size from network structure
            prev_layer_outputs = best_params[revlayer+1][0].get_value().shape[0]
            this_layer_inputs = best_params[revlayer][0].get_value().shape[0]
            h5file[layer_string + 'ksize'] = int((this_layer_inputs / prev_layer_outputs)**0.5)
        else:
            h5file[layer_string + 'type'] = 'Convolution'
            # Can't determine maxpoolsize from network structure - use default value
            h5file[layer_string + 'maxpoolsize'] = DEFAULT_MAXPOOL_SIZE

        h5file[layer_string + 'weights'] = best_params[revlayer][0].get_value()
        h5file[layer_string + 'bias'] = best_params[revlayer][1].get_value()

        #print best_params[revlayer][0].get_value().shape
        #print 'layer {0} Wsum={1}.'.format(layeri, np.sum(best_params[revlayer][0].get_value()))
        #print 'layer {0} Wsum={1}.'.format(layeri, np.sum(h5file[layer_string + 'weights'][...]))

    h5file.close()

    if COPY_GOOD_RESULTS and this_validation_loss < 0.10 or (test_score > 0.0 and test_score < 0.10):
        print 'Good network.'
        hash_string = hashlib.md5(output_paramfile).hexdigest()
        rename_path = output_paramfile.replace(
            '.h5', '_{0}.h5'.format(hash_string[-8:])).replace(
            input_path[:-1], good_output_path[:-1]).replace(
            'LGN1_MembraneSamples', 'LGN1').replace(
            '00000', '00k').replace(
            '0000', '0k').replace(
            '000', 'k').replace(
            '_train', '_').replace(
            '_valid', '').replace(
            '_test', '').replace(
            '_seed', '_s').replace(
            'progress_anneal_rotmir', '_ar').replace(
            ', ', ',')
        shutil.copyfile(output_paramfile, rename_path)

    #print 'Wrote network settings to h5 file {0}.'.format(output_paramfile)
