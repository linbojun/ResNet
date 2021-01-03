import pickle
import numpy as np
import tensorflow as tf
import os

def unpickle(file):
	with open(file, "rb") as fo:
		dict = pickle.load(fo, encoding = 'bytes')
	return dict

"""
def get_data(file_path):
	unpickle_file = unpickle(file_path)
	inputs = unpickle_file[b'data']
	labels  =unpickle_file[b'labels']

	inputs = np.array(inputs)
	labels = np.array(labels)

	inputs = inputs / 255.0
	inputs = tf.reshape(inputs, (-1, 3, 32, 32))
	inputs = tf.transpose(inputs, perm=[0,2,3,1])
	inputs = np.array(inputs, dtype = np.float32)

	return inputs, labels

"""
def get_data(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of
	normalized inputs (images) and an array of labels.
	You will want to first extract only the data that matches the
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels
	has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']
	# TODO: Do the rest of preprocessing!
	inputs = np.array(inputs)
	labels = np.array(labels)
	size = len(labels)

	indice = np.where((labels == first_class) | (labels == second_class))

	labels = labels[indice]
	inputs = inputs[indice]
	 #preprocess label
	labels = np.where(labels == first_class, 1, 0)
	labels = tf.one_hot(labels,2)

	#preprocess images
	inputs = inputs / 255.0
	inputs = tf.reshape(inputs, (-1, 3, 32, 32))
	inputs = tf.transpose(inputs, perm = [0,2,3,1])
	inputs = np.array(inputs, dtype = np.float32)
	print("In get_data labels: ",labels)

	return inputs, labels
