from preprocess import get_full_data, get_part_data
import Util as util
from matplotlib import pyplot as plt
from model import ResNet
#from resnet import resnet_18, resnet_34



import os
import tensorflow as tf

import random
import argparse
import math

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type = str, default = 'train',
                        help = "train or test")
    parser.add_argument("--data_type", type = str, default = "full",
                        help = "part or full")

    arg = parser.parse_args()

    print("main() start")

    train_path = "../data/train"
    test_path = "../data/test"
    
    if arg.data_type == "part":
        train_images, train_labels = get_part_data(train_path, first_class = 3, second_class = 5)
        test_images, test_labels = get_part_data(test_path,  first_class = 3, second_class = 5)
        #model = resnet_18(num_class = 2)
        mode = ResNet(num_classes = 2)

    else:
        train_images, train_labels = get_full_data(train_path)
        test_images, test_labels = get_full_data(test_path)
        model =ResNet(num_classes = 10)


    model.build(input_shape=(None, 32, 32, 3))
    model.summary()



    print("finish build model")
    print("--------------------------------------")
    print("train_images.shape : ",train_images.shape)
    print("test_image.shape: ",test_images.shape)
    print("--------------------------------------")
    num_epoch = 50
    if arg.mode == "train":
        for i in range(num_epoch):
            print("epoch  %s out of %s" % (i+1, num_epoch))
            save_path = os.path.join("save_model_own",'resnet{}_epoch{}_on_{}_data'.format(18, i+1, "full"))
            util.train(model, train_images, train_labels, save_path)
            train_accur = util.test(model, train_images, train_labels)
            print("train accur: ", train_accur)
            print("==============================")
            test_accur = util.test(model, test_images, test_labels)
            print("""
#################################
#                               #
#       test_accur : %s         #
#                               #
#################################
                """ % (test_accur))
        util.visualize_loss(model.loss_list)
    else:
        test_accur = util.test(model, test_images, test_labels)
        print("test accur: ", test_accur)
        print("==============================")


    
    





if __name__ == '__main__':
    main()