import mahotas
import scipy.ndimage
import scipy.misc
import gzip
import cPickle

imgrad = 47
imgd = 2 * imgrad + 1

zrad = 0
zd = 2 * zrad + 1

#test mode
#nimages = 10
#display_output = True
#ntrain = 5
#nvalid = 2
#ntest = 2

#live mode
nimages = 75
display_output = False
ntrain = 5000
nvalid = 1000
ntest = 1000

train_set = (np.zeros((ntrain, imgd*imgd*zd), dtype=np.uint8), np.zeros(ntrain, dtype=uint8))
valid_set = (np.zeros((nvalid, imgd*imgd*zd), dtype=np.uint8), np.zeros(nvalid, dtype=uint8))
test_set = (np.zeros((ntest, imgd*imgd*zd), dtype=np.uint8), np.zeros(ntest, dtype=uint8))

saturation_level = 0.005

def normalize_image(original_image):
	sorted_image = np.sort( np.uint8(original_image).ravel() )
	minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
	maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
	norm_image = np.uint8( (original_image - minval) * ( 255 / (maxval - minval)) )
	return norm_image

shrink_radius = 5
y,x = np.ogrid[-shrink_radius:shrink_radius+1, -shrink_radius:shrink_radius+1]
shrink_disc = x*x + y*y <= shrink_radius*shrink_radius

gblur_sigma = 1

min_border = ceil( sqrt( 2 * ( (imgrad + 1) ** 2 ) ) )

mask = None

nsample_images = nimages - zrad * 2

membrane_proportion = 0.5

random.seed(7)

train_i = 0;
valid_i = 0;
test_i = 0;
sample_count = 0;

for imgi in range (zrad, nimages-zrad):

	input_img = normalize_image(mahotas.imread('D:\\dev\\datasets\\isbi\\train-input\\train-input_{0:04d}.tif'.format(imgi)))
	#input_img = mahotas.imread('D:\\dev\\datasets\\isbi\\train-input\\train-input_{0:04d}.tif'.format(imgi))
	label_img = mahotas.imread('D:\\dev\\datasets\\isbi\\train-labels\\train-labels_{0:04d}.tif'.format(imgi))

	input_vol = zeros((input_img.shape[0], input_img.shape[1], zd), dtype=uint8)
	for zoffset in range (zd):
		if zd == zrad:
			input_vol[:,:,zoffset] = input_img
		else:
			input_vol[:,:,zoffset] = normalize_image(mahotas.imread('D:\\dev\\datasets\\isbi\\train-input\\train-input_{0:04d}.tif'.format(imgi - zrad + zoffset)))
			#input_vol[:,:,zoffset] = mahotas.imread('D:\\dev\\datasets\\isbi\\train-input\\train-input_{0:04d}.tif'.format(imgi - zrad + zoffset))

	blur_img = scipy.ndimage.gaussian_filter(input_img, gblur_sigma)

	boundaries = label_img==0;
	boundaries[0:-1,:] = np.logical_or(boundaries[0:-1,:],  diff(label_img, axis=0)!=0);
	boundaries[:,0:-1] = np.logical_or(boundaries[:,0:-1], diff(label_img, axis=1)!=0);

	# erode to be sure we include at least one membrane
	inside = mahotas.erode(boundaries == 0, shrink_disc)

	#display = input_img.copy()
	#display[np.nonzero(inside)] = 0
	#figure(figsize=(20,20))
	#imshow(display, cmap=cm.gray)

	seeds = label_img.copy()
	seeds[np.nonzero(inside==0)] = 0
	grow = mahotas.cwatershed(255-blur_img, seeds)

	membrane = np.zeros(input_img.shape, dtype=uint8)
	membrane[0:-1,:] = diff(grow, axis=0) != 0;
	membrane[:,0:-1] = np.logical_or(membrane[:,0:-1], diff(grow, axis=1) != 0);

	#display[np.nonzero(membrane)] = 2
	#figure(figsize=(20,20))
	#imshow(display, cmap=cm.gray)

	# erode again to avoid all membrane
	non_membrane = mahotas.erode(inside, shrink_disc)

	if mask is None:
		mask = ones(input_img.shape, dtype=uint8)
		mask[:min_border,:] = 0;
		mask[-min_border:,:] = 0;
		mask[:,:min_border] = 0;
		mask[:,-min_border:] = 0;

	membrane_indices = np.nonzero(np.logical_and(membrane, mask))
	nonmembrane_indices = np.nonzero(np.logical_and(non_membrane, mask))

	train_target = int32(float32(ntrain) / nsample_images * (imgi - zrad + 1))
	valid_target = int32(float32(nvalid) / nsample_images * (imgi - zrad + 1))
	test_target = int32(float32(ntest) / nsample_images * (imgi - zrad + 1))

	while train_i < train_target or valid_i < valid_target or test_i < test_target:

		membrane_sample = random.random() < membrane_proportion

		if membrane_sample:
			randmem = random.choice(len(membrane_indices[0]))
			(samp_i, samp_j) = (membrane_indices[0][randmem], membrane_indices[1][randmem])
			membrane_type = "membrane"
		else:
			randnonmem = random.choice(len(nonmembrane_indices[0]))
			(samp_i, samp_j) = (nonmembrane_indices[0][randnonmem], nonmembrane_indices[1][randnonmem])
			membrane_type = "non-membrane"

		# rotate by a random amount (linear interpolation)
		rotation = random.random()*360

		samp_vol = zeros((imgd, imgd, zd), dtype=uint8)

		for zoffset in range(zd):
			sample_img = input_vol[samp_i-min_border:samp_i+min_border, samp_j-min_border:samp_j+min_border, zoffset]
			sample_img = scipy.misc.imrotate(sample_img, rotation)
			samp_vol[:,:,zoffset] = sample_img[min_border-imgrad:min_border+imgrad+1, min_border-imgrad:min_border+imgrad+1]

		if display_output:
			output = zeros((imgd, imgd * zd), dtype=uint8)
			for out_z in range(zd):
				output[:,out_z * imgd : (out_z + 1) * imgd] = samp_vol[:,:,out_z]
			figure(figsize=(5, 5 * zd))
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

outfile = "MembraneSamples_{0}x{0}x{1}_mp{2:0.2f}_train{3}_valid{4}_test{5}.pkl.gz".format(imgd, zd, membrane_proportion, ntrain, nvalid, ntest)

print "Saving to {0}.".format(outfile)

#Save the results
f = gzip.open(outfile,'wb', compresslevel=1)
cPickle.dump((train_set, valid_set, test_set),f)
f.close()

print "Saved."
