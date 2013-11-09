import mahotas
import scipy.ndimage
import scipy.misc
import numpy as np
import gzip
import cPickle
import glob
import os
import h5py

param_path = 'D:/dev/Rhoana/membrane_cnn/results/good3/'
param_files = glob.glob(param_path + "*.h5")

target_boundaries = mahotas.imread(param_path + 'boundaries.png') > 0

offset_max = 32

target_boundaries = target_boundaries[offset_max:-offset_max,offset_max:-offset_max]

for param_file in param_files:

    if param_file.find('.ot.h5') != -1:
        continue

    print param_file

    net_output_file = param_file.replace('.h5','\\0005_classify_output_layer6_0.tif')
    net_output = mahotas.imread(net_output_file)
    net_output = np.float32(net_output) / np.max(net_output)

    offset_file = param_file.replace('.h5', '.ot.h5')
    h5off = h5py.File(offset_file, 'r')
    best_offset = h5off['/best_offset'][...]
    h5off.close()

    xoffset, yoffset = best_offset

    best_score = 0
    best_thresh = 0
    best_sigma = 0
    best_result = None

    offset_output = np.roll(net_output, xoffset, axis=0)
    offset_output = np.roll(offset_output, yoffset, axis=1)

    #Crop
    offset_output = offset_output[offset_max:-offset_max,offset_max:-offset_max]

    for smooth_sigma in arange(0, 3, 0.1):

        smooth_output = scipy.ndimage.filters.gaussian_filter(offset_output, smooth_sigma)

        for thresh in arange(0.1,1,0.1):
            result = smooth_output > thresh

            if np.sum(result) == 0:
                continue

            true_positives = np.sum(np.logical_and(result == 0, target_boundaries == 0))
            false_positives = np.sum(np.logical_and(result == 0, target_boundaries > 0))
            true_negatives = np.sum(np.logical_and(result > 0, target_boundaries > 0))
            false_negatives = np.sum(np.logical_and(result > 0, target_boundaries == 0))

            precision = float(true_positives) / float(true_positives + false_positives)
            recall = float(true_positives) / float(true_positives + false_negatives)
            Fscore = 2 * precision * recall / (precision + recall)

            if Fscore > best_score:
                best_score = Fscore
                best_thresh = thresh
                best_sigma = smooth_sigma
                best_result = result

    print 'Best score of {0} for sigma {1}, thresh {2}.'.format(best_score, best_sigma, best_thresh)

    output_file = param_file.replace('.h5', '.sm.ot.h5')
    h5out = h5py.File(output_file, 'w')
    h5out['/best_score'] = best_score
    h5out['/best_offset'] = best_offset
    h5out['/best_thresh'] = best_thresh
    h5out['/best_sigma'] = best_sigma
    h5out.close()


