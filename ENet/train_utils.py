import ENet
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.utils import to_categorical
from scipy.ndimage.filters import convolve
import numpy as np
import os


def get_image_mask_path(annotated_path, target_size):
	ann_arr = load_img(annotated_path, target_size=target_size)
	# target_size shaped array with True where pixel is annotated as a leaf
	return get_image_mask_arr(ann_arr)

def get_image_mask_arr(ann_img):
	ann_arr = img_to_array(ann_img)
	red_channel = ann_arr[:,:,0]
	return red_channel == 0


def get_image_label(annotated_path, target_size, classes):
	mask = get_image_mask_path(annotated_path, target_size)
	int_mask = mask.astype(int)
	return to_categorical(int_mask, classes)

def get_training_image(original_path, target_size):
	return img_to_array(load_img(original_path, target_size=target_size))	

def get_samples(original_dir, annotated_dir, target_size, classes=2):
	# requires filenames for matching annotated/original images to be the same
	X = []
	Y = []
	masks = []
	all_original_files = set(os.listdir(original_dir))
	for fname in os.listdir(annotated_dir):
		if fname in all_original_files:
			original_path = os.path.join(original_dir, fname)
			annotated_path = os.path.join(annotated_dir, fname)
			X.append(get_training_image(original_path, target_size))
			Y.append(get_image_label(annotated_path, target_size, classes))
			masks.append(get_image_mask_path(annotated_path, target_size))
	return np.array(X), np.array(Y), np.array(masks)
			

def extract_leaf(X, mask):
	X_copy = np.copy(X)
	mask = np.logical_not(mask)
	X_copy[:,:,0][mask] = 255
	X_copy[:,:,1][mask] = 255
	X_copy[:,:,2][mask] = 255
	return X_copy


def paste_leaf(leaf_img, annotate_img,  background_img):
	# all images MUST be the same size
	mask = get_image_mask_arr(annotate_img)
	leaf_arr = extract_leaf(img_to_array(leaf_img), mask)
	back_arr = img_to_array(background_img)
	return np.where(leaf_arr != 255, leaf_arr, back_arr)


def smooth_mask(mask, self_weight=0, size=11, threshold=.05):
    center = size/2
    kernal_area = size*size
    k = np.ones((size,size))
    center_val = kernal_area
    k[center][center] = center_val
    inp = mask.astype(int)
    zero_mask = np.zeros(mask.shape)
    one_mask = np.ones(mask.shape)
    smoothed = convolve(inp, k)
    threshold_val = kernal_area*self_weight + (kernal_area * threshold) - 1 
    return np.where(smoothed > threshold_val, one_mask, zero_mask).astype(bool)

