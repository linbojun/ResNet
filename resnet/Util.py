import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


import os
import random
import math



def accuracy(logits, labels):
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

    #correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
    #print("Util, accuracy: logits %s" % (logits))
    #print("Util, accuracy: labels %s" % (labels))
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    #print("Util, accuracy: correct_predictions %s" % (correct_predictions))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

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



def test(model, test_inputs, test_labels):

#    Tests the model on the test inputs and labels. You should NOT randomly
#    flip images or do any extra preprocessing.
#    :param test_inputs: test data (all images to be tested),
#    shape (num_inputs, width, height, num_channels)
#    :param test_labels: test labels (all corresponding labels),
#    shape (num_labels, num_classes)
#    :return: test accuracy - this should be the average accuracy across
#    all batches


    num_batch = int(test_labels.shape[0] / model.batch_size)
    tot = 0.0
    for i in range(num_batch):
        image_batch = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        label_batch = test_labels[i*model.batch_size:(i+1)*model.batch_size]
        logits = model.call(image_batch)
        accur = accuracy(logits,label_batch)
        tot += accur
    print(tot / num_batch)
    return tot / num_batch



def train(model, images, labels, save_path):
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
    num_batch = int(n/model.batch_size)

    for i in range(num_batch):
    #for i in range(1):
        print("batch ", i, "of ", num_batch)
        start_pos = i * model.batch_size
        end_pos = (i+1) * model.batch_size
        image_batch = images[start_pos : end_pos]
        label_batch = labels[start_pos : end_pos]
        with tf.GradientTape() as tape:
            #logits = model.call(image_batch)
            predictions = model(image_batch, training = True)
            loss_val = model.loss(predictions, label_batch)
            #loss_val = model.loss(y_true=label_batch, y_pred=predictions)
            print("loss: %s" % (loss_val))
            print("__________________________")
            # print("""loss: %s, \n logits: %s 
            #     ===============""" % (loss_val, predictions))
            model.loss_list.append(loss_val)
        gradients = tape.gradient(loss_val, model.trainable_variables)
        model.optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    model.save(save_path)
    return
