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

    best_score = 0
    best_offset = (0,0)
    best_thresh = 0
    best_result = None

    for xoffset in range(-offset_max, offset_max+1, 1):
        for yoffset in range(-offset_max, offset_max+1, 1):
            #Translate
            offset_output = np.roll(net_output, xoffset, axis=0)
            offset_output = np.roll(offset_output, yoffset, axis=1)

            #Crop
            offset_output = offset_output[offset_max:-offset_max,offset_max:-offset_max]

            for thresh in arange(0.1,0.9,0.1):
                result = offset_output > thresh

                true_positives = np.sum(np.logical_and(result == 0, target_boundaries == 0))
                false_positives = np.sum(np.logical_and(result == 0, target_boundaries > 0))
                true_negatives = np.sum(np.logical_and(result > 0, target_boundaries > 0))
                false_negatives = np.sum(np.logical_and(result > 0, target_boundaries == 0))

                precision = float(true_positives) / float(true_positives + false_positives)
                recall = float(true_positives) / float(true_positives + false_negatives)
                Fscore = 2 * precision * recall / (precision + recall)

                #print thresh
                #print Fscore

                if Fscore > best_score:
                    best_score = Fscore
                    best_offset = (xoffset, yoffset)
                    best_thresh = thresh
                    best_result = result

            #figsize(20,20);imshow(best_result,cmap=cm.gray)

    print 'Best score of {0} for offset {1}, thresh {2}.'.format(best_score, best_offset, best_thresh)

    output_file = param_file.replace('.h5', '.ot.h5')
    h5out = h5py.File(output_file, 'w')
    h5out['/best_score'] = best_score
    h5out['/best_offset'] = best_offset
    h5out['/best_thresh'] = best_thresh
    h5out.close()

