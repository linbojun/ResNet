from __future__ import absolute_import
from preprocess import get_data
import Util as util
from matplotlib import pyplot as plt


import os
import tensorflow as tf
import numpy as np
import random
import math


class ResNet(tf.keras.Model):
	def __init__(self):
		super(ResNet, self).__init__()
		self.batch_size = 100
		self.in_width = 32
		self.in_height = 32
		self.in_channels = 3
		#self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.015)
		self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.015,  momentum=0.3)
		self.loss_list = []

		#structure#
		self.conv_layer1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 7, strides=(2, 2), padding="same")

		#do max pool 3x3 with stride 2 here
		self.max_pool = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)
		"""
		self.res_block1 = util.residual_block(in_channel = 64, out_channel = 64)
		self.res_block2 = util.residual_block(in_channel = 64, out_channel = 64)
		self.res_block3 = util.residual_block(in_channel = 64, out_channel = 64)

		self.res_block4 = util.residual_block(in_channel = 64,out_channel = 128, downsample = True)

		self.res_block5 = util.residual_block(in_channel = 128, out_channel = 128)
		self.res_block6 = util.residual_block(in_channel = 128, out_channel = 128)
		self.res_block7 = util.residual_block(in_channel = 128, out_channel = 128)

		self.res_block8 = util.residual_block(in_channel = 128, out_channel = 256, downsample = True)

		self.res_block9 = util.residual_block(in_channel = 256, out_channel = 256)
		self.res_block10 = util.residual_block(in_channel = 256, out_channel = 256)
		self.res_block11 = util.residual_block(in_channel = 256, out_channel = 256)
		self.res_block12 = util.residual_block(in_channel = 256, out_channel = 256)
		self.res_block13 = util.residual_block(in_channel = 256, out_channel = 256)

		self.res_block14 = util.residual_block(in_channel = 256, out_channel = 512, downsample = True)

		self.res_block15 = util.residual_block(in_channel = 512, out_channel = 512)
		self.res_block16 = util.residual_block(in_channel = 512, out_channel = 512)
		"""
		#average pool
		self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
		self.dense1 = tf.keras.layers.Dense(1000, activation = 'softmax')
		self.dense2 = tf.keras.layers.Dense(2)





	def call(self, inputs):
		token = self.conv_layer1(inputs)
		token = util.batch_normal(token)
		token = tf.nn.relu(token)

		token = self.max_pool(token)
		"""
		token = self.res_block1(token)
		token = self.res_block2(token)
		token = self.res_block3(token)

		token = self.res_block4(token)

		token = self.res_block5(token)
		token = self.res_block6(token)
		token = self.res_block7(token)

		token = self.res_block8(token)

		token = self.res_block9(token)
		token = self.res_block10(token)
		token = self.res_block11(token)
		token = self.res_block12(token)
		token = self.res_block13(token)

		token = self.res_block14(token)

		token = self.res_block15(token)
		token = self.res_block16(token)
		"""
		token = util.residual_block(token, 64)
		token = util.residual_block(token, 64)
		token = util.residual_block(token, 64)

		token = util.residual_block_downsample(token, 128)

		token = util.residual_block(token, 128)
		token = util.residual_block(token, 128)
		token = util.residual_block(token, 128)

		token = util.residual_block_downsample(token, 256)

		token = util.residual_block(token, 256)
		token = util.residual_block(token, 256)
		token = util.residual_block(token, 256)
		token = util.residual_block(token, 256)
		token = util.residual_block(token, 256)

		token = util.residual_block_downsample(token, 512)

		token = util.residual_block(token, 512)
		token = util.residual_block(token, 512)


		token = self.average_pool(token)

		token = self.dense1(token)
		probs = self.dense2(token)
		return probs

	def loss(self, logits, labels):
		#return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs));
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))


	def accuracy(self, logits, labels):
		"""
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """

      
		#print("probs: ", probs)
		#print("labels: ",labels )
		#print("In accuracy(self, probs, labels): tf.argmax(probs, 1) = ", tf.argmax(probs, 1))

		#correct_predictions = tf.equal(tf.argmax(probs, 1), labels)
		correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	def train(self, images, labels):
		'''
		:param images: (num_image, width, height, num_channels)
		:param labels: (num_labels, num_class)
		'''
		n = images.shape[0]
		indice = tf.range(n)
		indice = tf.random.shuffle(indice)
		labels = tf.gather(labels, indice)
		images = tf.gather(images, indice)
		images = tf.image.random_flip_left_right(images)
		num_batch = int(n/self.batch_size)

		for i in range(num_batch):
			print("batch ", i, "of ", num_batch)
			start_pos = i * self.batch_size
			end_pos = (i+1) * self.batch_size
			image_batch = images[start_pos : end_pos]
			label_batch = labels[start_pos : end_pos]
			with tf.GradientTape() as tape:
				logits = self.call(image_batch)
				loss = self.loss(logits, label_batch)
				print("loss: ", loss)
				self.loss_list.append(loss)
			gradients = tape.gradient(loss, self.trainable_variables)
			self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
		return

	def test(self, test_images, test_labels):
		'''
		:param images: (num_image, width, height, num_channels)
		:param labels: (num_labels, num_class)
		'''
		n = test_labels.shape[0]
		num_batch = n//self.batch_size
		tot = 0.0

		for i in range(num_batch):
			image_batch = test_images[i * self.batch_size : (i+1) * self.batch_size]
			label_batch = test_labels[i * self.batch_size : (i+1) * self.batch_size]
			probs = self.call(image_batch)
			accur = self.accuracy(probs, label_batch)
			tot += accur
		print("accur in test: ", tot/num_batch)
		return tot/num_batch

	def visualize_loss(losses):
		"""
		Uses Matplotlib to visualize the losses of our model.
		:param losses: list of loss data stored from train. Can use the model's loss_list
		field

		NOTE: DO NOT EDIT

		return: doesn't return anything, a plot should pop-up
		"""
		x = [i for i in range(len(losses))]
		plt.plot(x, losses)
		plt.title('Loss per batch')
		plt.xlabel('Batch')
		plt.ylabel('Loss')
		plt.show()



def main():
	print("main() start")
	model = ResNet()
	print("finish build model")
	print("--------------------------------------")
	train_path = "data/train"
	test_path = "data/test"
	train_images, train_labels = get_data(train_path, first_class = 3, second_class = 5)
	test_images, test_labels = get_data(test_path,  first_class = 3, second_class = 5)
	print("train_images.shape : ",train_images.shape)
	print("test_image.shape: ",test_images.shape)
	print("--------------------------------------")


	for i in range(25):
		print("epoch ", i+1, "out of 25")
		model.train(train_images, train_labels)
		train_accur = model.test(train_images, train_labels)
		print("train accur: ", train_accur)
		print("-------------------------------------")

    
	test_accur = model.test(test_images, test_labels)
	print("final test_accur :", test_accur)
	visualize_loss(model.loss_list)






if __name__ == '__main__':
	main()


