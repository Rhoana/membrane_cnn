import imread
import mahotas
import scipy.ndimage
import scipy.misc
import numpy as np
import gzip
import cPickle
import glob
import os
import sys
import random

EVEN_OUTPUT = False
PNG_OUTPUT = False
DOWNSAMPLE_BY = 2
display_output = False

#imgrad = 50
#imgrad = 47
#imgrad = 32
imgrad = 24
#imgrad = 15

# EVEN_OUTPUT = True
# imgrad = 14

if EVEN_OUTPUT:
    imgd = 2 * imgrad
else:
    imgd = 2 * imgrad + 1

zrad = 0

start_image = 64
nimages = 100

#test mode
# display_output = True
# ntrain = 10
# nvalid = 4
# ntest = 4

#small training dataset
# ntrain = 5000
# nvalid = 1000
# ntest = 1000

#large training dataset
# ntrain = 10000
# nvalid = 2000
# ntest = 2000

# ntrain = 20000
# nvalid = 4000
# ntest = 4000

ntrain = 50000
nvalid = 5000
ntest = 5000

#random_seeds = [7, 11, 13, 17, 19]
random_seeds = [7, 11]

image_input = sys.argv[1]
label_input = sys.argv[2]
nimages = 25

saturation_level = 0.005

def normalize_image(original_image):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    return np.uint8(255 - norm_image)

shrink_radius = 5
y,x = np.ogrid[-shrink_radius:shrink_radius+1, -shrink_radius:shrink_radius+1]
shrink_disc = x*x + y*y <= shrink_radius*shrink_radius

gblur_sigma = 1

min_border = np.ceil( np.sqrt( 2 * ( (imgrad + 1) ** 2 ) ) ) * DOWNSAMPLE_BY

mask = None


nsample_images = nimages - zrad * 2

membrane_proportion = 0.5

for seed in random_seeds:
    train_set = (np.zeros((ntrain, imgd*imgd), dtype=np.uint8), np.zeros(ntrain, dtype=np.uint8))
    valid_set = (np.zeros((nvalid, imgd*imgd), dtype=np.uint8), np.zeros(nvalid, dtype=np.uint8))
    test_set = (np.zeros((ntest, imgd*imgd), dtype=np.uint8), np.zeros(ntest, dtype=np.uint8))

    random.seed(seed)

    train_i = 0;
    valid_i = 0;
    test_i = 0;
    sample_count = 0;

    for imgi in range (zrad, nimages-zrad):

        file_index = start_image + imgi
        input_img = normalize_image(imread.imread_multi(image_input)[imgi])
        #print img_files[file_index]
        label_img = mahotas.imread(label_input)[imgi]
        #print seg_files[file_index]

        if len( label_img.shape ) == 3:
            label_img = label_img[ :, :, 0 ] * 2**16 + label_img[ :, :, 1 ] * 2**8 + label_img[ :, :, 2 ]

        input_vol = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
        input_vol[:,:] = input_img

        blur_img = scipy.ndimage.gaussian_filter(input_img, gblur_sigma)

        membrane = label_img==0;
        non_membrane = ~membrane

        if mask is None:
            mask = np.ones(input_img.shape, dtype=np.uint8)
            mask[:min_border,:] = 0;
            mask[-min_border:,:] = 0;
            mask[:,:min_border] = 0;
            mask[:,-min_border:] = 0;

        membrane_indices = np.nonzero(np.logical_and(membrane, mask))
        nonmembrane_indices = np.nonzero(np.logical_and(non_membrane, mask))

        #print 'Image {0} has {1} membrane and {2} non-membrane pixels.'.format(imgi, len(membrane_indices[1]), len(nonmembrane_indices[1]))

        train_target = np.int32(np.float32(ntrain) / nsample_images * (imgi - zrad + 1))
        valid_target = np.int32(np.float32(nvalid) / nsample_images * (imgi - zrad + 1))
        test_target = np.int32(np.float32(ntest) / nsample_images * (imgi - zrad + 1))

        while train_i < train_target or valid_i < valid_target or test_i < test_target:

            membrane_sample = random.random() < membrane_proportion

            if membrane_sample:
                randmem = random.randrange(len(membrane_indices[0]))
                (samp_i, samp_j) = (membrane_indices[0][randmem], membrane_indices[1][randmem])
                membrane_type = "membrane"
            else:
                randnonmem = random.randrange(len(nonmembrane_indices[0]))
                (samp_i, samp_j) = (nonmembrane_indices[0][randnonmem], nonmembrane_indices[1][randnonmem])
                membrane_type = "non-membrane"


            samp_vol = np.zeros((imgd, imgd), dtype=np.uint8)

            sample_img = input_vol[samp_i-min_border:samp_i+min_border, samp_j-min_border:samp_j+min_border]
            if np.random.uniform() > 0.5:
                sample_img = sample_img[::-1, ...]
            sample_img = np.rot90(sample_img, random.randrange(4))

            if DOWNSAMPLE_BY != 1:
                sample_img = np.uint8(mahotas.imresize(sample_img, 1.0 / DOWNSAMPLE_BY))

            mid_pix = min_border / DOWNSAMPLE_BY

            if EVEN_OUTPUT:
                samp_vol[:,:] = sample_img[mid_pix-imgrad:mid_pix+imgrad, mid_pix-imgrad:mid_pix+imgrad]
            else:
                samp_vol[:,:] = sample_img[mid_pix-imgrad:mid_pix+imgrad+1, mid_pix-imgrad:mid_pix+imgrad+1]

            if display_output:
                print 'Location ({0},{1},{2}).'.format(samp_i, samp_j, imgi)
                output = zeros((imgd, imgd), dtype=uint8)
                output[:,imgd] = samp_vol[:,:]
                figure(figsize=(5, 5))
                title(membrane_type)
                imshow(output, cmap=cm.gray)

            if train_i < train_target:
                train_set[0][train_i,:] = samp_vol.ravel()
                train_set[1][train_i] = membrane_sample
                train_i = train_i + 1
                sample_type = 'train'
            elif valid_i < valid_target:
                valid_set[0][valid_i,:] = samp_vol.ravel()
                valid_set[1][valid_i] = membrane_sample
                valid_i = valid_i + 1
                sample_type = 'valid'
            elif test_i < test_target:
                test_set[0][test_i,:] = samp_vol.ravel()
                test_set[1][test_i] = membrane_sample
                test_i = test_i + 1
                sample_type = 'test'

            #print "Sampled {5} at {0}, {1}, {2}, r{3:.2f} ({4})".format(samp_i, samp_j, imgi, rotation, sample_type, membrane_type)

            sample_count = sample_count + 1
            if sample_count % 5000 == 0:
                print "{0} samples ({1}, {2}, {3}).".format(sample_count, train_i, valid_i, test_i)

    print "Made a total of {0} samples ({1}, {2}, {3}).".format(sample_count, train_i, valid_i, test_i)

    if PNG_OUTPUT:
        outdir = "ISBI_MembraneSamples_{0}x{0}_mp{2:0.2f}\\train\\".format(imgd, 1, membrane_proportion)
        if not os.path.exists(outdir): os.makedirs(outdir)
        for imgi in range(ntrain):
            mahotas.imsave(outdir + "{0:08d}_{1}.png".format(imgi, train_set[1][imgi]), train_set[0][imgi,:].reshape((imgd,imgd)))

        outdir = "ISBI_MembraneSamples_{0}x{0}_mp{2:0.2f}\\valid\\".format(imgd, 1, membrane_proportion)
        if not os.path.exists(outdir): os.makedirs(outdir)
        for imgi in range(nvalid):
            mahotas.imsave(outdir + "{0:08d}_{1}.png".format(imgi, valid_set[1][imgi]), valid_set[0][imgi,:].reshape((imgd,imgd)))

        outdir = "ISBI_MembraneSamples_{0}x{0}_mp{2:0.2f}\\test\\".format(imgd, 1, membrane_proportion)
        if not os.path.exists(outdir): os.makedirs(outdir)
        for imgi in range(ntest):
            mahotas.imsave(outdir + "{0:08d}_{1}.png".format(imgi, test_set[1][imgi]), test_set[0][imgi,:].reshape((imgd,imgd)))

    if DOWNSAMPLE_BY != 1:
        ds_string = '_ds{0}b'.format(DOWNSAMPLE_BY)
    else:
        ds_string = ''

    outfile = "ISBI_MembraneSamples_{0}x{0}_mp{2:0.2f}_train{3}_valid{4}_test{5}_seed{6}{7}.pkl.gz".format(imgd, 1, membrane_proportion, ntrain, nvalid, ntest, seed, ds_string)

    print "Saving to {0}.".format(outfile)

    #Save the results
    f = gzip.open(outfile,'wb', compresslevel=1)
    cPickle.dump((train_set, valid_set, test_set),f)
    f.close()

    print "Saved."
