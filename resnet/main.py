from preprocess import get_full_data, get_part_data
import Util as util
from matplotlib import pyplot as plt
from model import ResNet


import os
import re
import tensorflow as tf

import random
import argparse
import math

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type = str, default = 'test',
                        help = "train or test")
    parser.add_argument("--data_type", type = str, default = "full",
                        help = "part or full")

    arg = parser.parse_args()

    print("main() start")

    train_path = "../data/train"
    test_path = "../data/test"
    num_classes = 0
    
    if arg.data_type == "part":
        train_images, train_labels = get_part_data(train_path, first_class = 3, second_class = 5)
        test_images, test_labels = get_part_data(test_path,  first_class = 3, second_class = 5)
        num_classes = 2

    else:
        train_images, train_labels = get_full_data(train_path)
        test_images, test_labels = get_full_data(test_path)
        num_classes = 10






    print("finish build model")
    print("--------------------------------------")
    print("train_images.shape : ",train_images.shape)
    print("test_image.shape: ",test_images.shape)
    print("--------------------------------------")
    num_epoch = 50
    if arg.mode == "train":
        model = ResNet(num_classes)
        model.build(input_shape=(None, 32, 32, 3))
        model.summary()
        for i in range(num_epoch):
            print("epoch  %s out of %s" % (i+1, num_epoch))
            save_path = os.path.join("save_model_own",'resnet{}_epoch{}'.format(18, i+1))
            #util.train(model, train_images, train_labels, save_path)
            model.compile( optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"],)
            model.fit(x= train_images, y = train_labels, batch_size = 1000)
            
            test_accur = util.test(model, test_images, test_labels)
            print("""
################################################
#                               #
#       test_accur : %s         #
#                               #
################################################
                """ % (test_accur))
        util.visualize_loss(model.loss_list)
    elif arg.mode == "test":
        best_accur = 0
        model_name = None
        for file in os.listdir(os.path.join(os.getcwd(), "save_model_own")):
            if file.startswith('resnet{}_epoch'.format(18)):
                #token = re.findall("train_accur\d+\.\d+", file)[0].strip('train_accur')
                #print("$$token: ", token)
                #print("$$file: ", file, "model name :", model_name)
                current_accur = float(re.findall("train_accur\d+\.\d+", file)[0].strip('train_accur'))
                #print("current_accur: ", current_accur)
                if current_accur > best_accur:
                    model_name = file
                    best_accur = current_accur

        if model_name is None:
            raise ValueError("Did not have a model with format saved_model/")
        print("Using Model Name: {}".format(model_name))
        print("Loading model: %s" % (os.path.join("save_model_own", model_name)))
        model = tf.keras.models.load_model(os.path.join("save_model_own", model_name), compile = False)
        model.build(input_shape=(None, 32, 32, 3))
        model.summary()
        test_accur = util.test(model, test_images, test_labels)
        
        print("Test mode accur: ", test_accur)
        print("==============================")
    else:
        largest_epoch = 0
        model_name = None
        for file in os.listdir(os.path.join(os.getcwd(), "save_model_own")):
            if file.startswith('resnet{}_epoch'.format(18)):
                #token = re.findall("resnet18_epoch\d+", file)[0]
                #print(token[14:])
                #print("token:", int(re.findall("resnet18_epoch\d+", file)[0][14:]), "file:||",file)
                epoch = int(re.findall("resnet18_epoch\d+", file)[0][14:])
                if epoch > largest_epoch:
                    model_name = file
                    largest_epoch = epoch
        if model_name is None:
            raise ValueError("Did not have a model with format saved_model/")
        print("Using Model Name: {}".format(model_name))
        print("Loading model: %s" % (os.path.join("save_model_own", model_name)))
        model = tf.keras.models.load_model(os.path.join("save_model_own", model_name), compile = False)
        model.build(input_shape=(None, 32, 32, 3))
        model.summary()

        for i in range(largest_epoch, num_epoch):
            print("epoch  %s out of %s" % (i+1, num_epoch))
            save_path = os.path.join("save_model_own",'resnet{}_epoch{}'.format(18, i+1))
            #util.train(model, train_images, train_labels, save_path)
            model.compile( optimizer=tf.keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"],)
            model.fit(x= train_images, y = train_labels, batch_size = 1000)
            train_accur = util.test(model, train_images, train_labels)
            model.save(save_path + 'train_accur{:f}'.format(train_accur))
            test_accur = util.test(model, test_images, test_labels)
            print("""
################################################
#                                               #
#       test_accur : %s         #
#                                               #
################################################
                """ % (test_accur))
        util.visualize_loss(model.loss_list)






    
    





if __name__ == '__main__':
    main()