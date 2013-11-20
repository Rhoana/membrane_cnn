import mahotas
import scipy.ndimage
import scipy.misc
import numpy as np
import gzip
import cPickle
import glob
import os
import h5py
import partition_comparison

#param_path = 'D:/dev/Rhoana/membrane_cnn/results/good3/'
param_path = 'D:/dev/Rhoana/membrane_cnn/results/stumpin/'
param_files = glob.glob(param_path + "*.h5")

target_boundaries = mahotas.imread(param_path + 'boundaries.png') > 0

offset_max = 32

target_boundaries = target_boundaries[offset_max:-offset_max,offset_max:-offset_max]
target_segs = np.uint32(mahotas.label(target_boundaries)[0])

param_files = [x for x in param_files if x.find('.ot.h5') == -1]

average_result = np.zeros(target_boundaries.shape, dtype=np.float32)
nresults = 0

blur_radius = 3;
y,x = np.ogrid[-blur_radius:blur_radius+1, -blur_radius:blur_radius+1]
disc = x*x + y*y <= blur_radius*blur_radius

for param_file in param_files:

    if param_file.find('.ot.h5') != -1:
        continue

    print param_file

    #net_output_file = param_file.replace('.h5','\\0005_classify_output_layer6_0.tif')
    net_output_file = param_file.replace('.h5','\\0100_classify_output_layer6_0.tif')
    net_output = mahotas.imread(net_output_file)
    net_output = np.float32(net_output) / np.max(net_output)

    offset_file = param_file.replace('.h5', '.sm.ot.h5')
    h5off = h5py.File(offset_file, 'r')
    best_offset = h5off['/best_offset'][...]
    best_sigma = h5off['/best_sigma'][...]
    h5off.close()

    xoffset, yoffset = best_offset

    offset_output = scipy.ndimage.filters.gaussian_filter(net_output, float(best_sigma))

    offset_output = np.roll(offset_output, xoffset, axis=0)
    offset_output = np.roll(offset_output, yoffset, axis=1)

    #Crop
    offset_output = offset_output[offset_max:-offset_max,offset_max:-offset_max]

    average_result += offset_output

    nresults += 1

average_result = average_result / nresults

sigma_range = arange(0, 3, 0.5)
thresh_range = arange(0.05,0.7,0.02)

sigma_range = [0]
#thresh_range = [0.3]

all_voi_results = []

for smooth_sigma in sigma_range:

    best_score = Inf
    best_sigma = 0
    best_thresh = 0
    best_result = None

    smooth_output = scipy.ndimage.filters.gaussian_filter(average_result, smooth_sigma)
    max_smooth = 2 ** 16 - 1
    smooth_output = np.uint16((1 - smooth_output) * max_smooth)

    thresh_voi_results = []

    for thresh in thresh_range:

        below_thresh = smooth_output < np.uint16(max_smooth * thresh)

        #below_thresh = mahotas.morph.close(below_thresh.astype(np.bool), disc)
        #below_thresh = mahotas.morph.open(below_thresh.astype(np.bool), disc)

        seeds,nseeds = mahotas.label(below_thresh)

        if nseeds == 0:
            continue

        ws = np.uint32(mahotas.cwatershed(smooth_output, seeds))

        voi_score = partition_comparison.variation_of_information(target_segs.ravel(), ws.ravel())

        thresh_voi_results.append(voi_score)

        print 's={0:0.2f}, t={1:0.2f}, voi_score={2:0.4f}.'.format(smooth_sigma, thresh, voi_score)

        dx, dy = np.gradient(ws)
        result = np.logical_or(dx!=0, dy!=0)

        figsize(20,20)
        imshow(result, cmap=cm.gray)
        plt.show()

        if voi_score < best_score:
            best_score = voi_score
            best_sigma = smooth_sigma
            best_thresh = thresh

            dx, dy = np.gradient(ws)
            best_result = np.logical_or(dx!=0, dy!=0)

    all_voi_results.append(thresh_voi_results)

    # figsize(20,20)
    # imshow(best_result, cmap=cm.gray)
    # plt.show()

    print 'Best VoI score of {0} with {3} segments for sigma {1}, thresh {2}.'.format(best_score, best_sigma, best_thresh, nseeds)

plot_list = []
for voi_results in all_voi_results:
    handle = plot(thresh_range, voi_results)[0]
    plot_list.append(handle)
xlabel('Threshold')
ylabel('VoI Score')
legend(plot_list, [str(x) for x in sigma_range])
plt.show

figsize(20,20);
imshow(average_result,cmap=cm.gray)

# figsize(20,20);
# imshow(best_result,cmap=cm.gray)


