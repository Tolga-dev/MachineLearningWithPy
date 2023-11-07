# deep computer vision using convolutional neural networks
# shortly cnn
from functools import partial

import keras
# the architecture of the visual cortex
# LeNet-5 architecture, it is used to check handwritten check numbers
# convolutional and pooling layers

# Convolutional layers
# it is work so well for image recognitions:
# convolutional l1 -> convolutional l2 -> input-layer
# filters
# stacking multiple feature maps

# memory requirements
# another problem with cnn is that the convolutional layers require a huge amount of RAM

# pooling layers
# to reduce the computational load, memory usage and the number of parameters
# thereby limiting the risk of overfitting

# most common type of pooling layer.

# cnn architectures
# Lenet -5
# most know cnn architecture
# it is used for handwritten digit recognition

# alex net
# similar to lenet  5, only much larger and deeper,

# google net
# Xception


# classification and localization
# localizing an object in a picture can be expressed as a regression task.
# predict the horizontal and vertical coordinates of the object's center, width and height
# VGG Image Annotator, LabelImg,
# OpenLabeler, or ImgLab to label a picture

# what if the images contain multiple objects>
# object detection, the task of classifying and localizing multiple objects in an image

# fully convolutional networks
# YOLO
# semantic segmentation

# exercises
#


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import load_sample_image
import matplotlib as mpl

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")



def plot_color_image(image):
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")


def crop(image):
    return image[150:220, 130:250]


class TrainingExample:
    def __init__(self):
        print("ok")


    def PretrainedModels(self):
        china = load_sample_image('china.jpg') / 255
        flower = load_sample_image('flower.jpg') / 255
        images = np.array([china, flower])

        model = keras.applications.resnet50.ResNet50(weights="imagenet")
        images_resized = tf.image.resize(images, [224, 224])
        inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)

        Y_proba = model.predict(inputs)
        top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
        for image_index in range(len(images)):
            print("Image #{}".format(image_index))
        for class_id, name, y_proba in top_K[image_index]:
            print(" {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
        print()



    def ResNet(self):
        DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                                padding="SAME", use_bias=False)

        class ResidualUnit(keras.layers.Layer):
            def __init__(self, filters, strides=1, activation="relu", **kwargs):
                super().__init__(**kwargs)
                self.activation = keras.activations.get(activation)
                self.main_layers = [
                    DefaultConv2D(filters, strides=strides),
                    keras.layers.BatchNormalization(),
                    self.activation,
                    DefaultConv2D(filters),
                    keras.layers.BatchNormalization()]
                self.skip_layers = []
                if strides > 1:
                    self.skip_layers = [
                        DefaultConv2D(filters, kernel_size=1, strides=strides),
                        keras.layers.BatchNormalization()]

            def call(self, inputs):
                Z = inputs
                for layer in self.main_layers:
                    Z = layer(Z)
                skip_Z = inputs
                for layer in self.skip_layers:
                    skip_Z = layer(skip_Z)
                return self.activation(Z + skip_Z)

        model = keras.models.Sequential()
        model.add(DefaultConv2D(64, kernel_size=7, strides=2,
                                input_shape=[224, 224, 3]))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
        prev_filters = 64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            strides = 1 if filters == prev_filters else 2
            model.add(ResidualUnit(filters, strides=strides))
            prev_filters = filters
        model.add(keras.layers.GlobalAvgPool2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(10, activation="softmax"))
        model.summary()


    def average_pool(self):
        china = load_sample_image('china.jpg') / 255
        flower = load_sample_image('flower.jpg') / 255
        images = np.array([china, flower])

        max_pool = keras.layers.MaxPool2D(pool_size=2)

        cropped_images = np.array([crop(image) for image in images], dtype=np.float32)

        avg_pool = keras.layers.AvgPool2D(pool_size=2)
        output_avg = avg_pool(cropped_images)
        fig = plt.figure(figsize=(12, 8))
        gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Input", fontsize=14)
        ax1.imshow(cropped_images[0])  # plot the 1st image
        ax1.axis("off")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Output", fontsize=14)
        ax2.imshow(output_avg[0])  # plot the output for the 1st image
        ax2.axis("off")
        plt.show()


    def max_pool(self):
        china = load_sample_image('china.jpg') / 255
        flower = load_sample_image('flower.jpg') / 255
        images = np.array([china, flower])

        max_pool = keras.layers.MaxPool2D(pool_size=2)

        cropped_images = np.array([crop(image) for image in images], dtype=np.float32)
        output = max_pool(cropped_images)
        fig = plt.figure(figsize=(12, 8))
        gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Input", fontsize=14)
        ax1.imshow(cropped_images[0])  # plot the 1st image
        ax1.axis("off")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Output", fontsize=14)
        ax2.imshow(output[0])  # plot the output for the 1st image
        ax2.axis("off")
        plt.show()

    def ConvolutionalLayer(self):
        # each image is typically represented as a 3d tensor of shape
        # [height, width, chans], mini batch represented as 4d tenser of shape
        # fe,  [mini-batch ,height, width, chans]
        # bias term is 1d, [bias]

        # load two sample images, make two filters, apply and display
        china = load_sample_image('china.jpg') / 255
        flower = load_sample_image('flower.jpg') / 255
        images = np.array([china, flower])
        batch_size, height, width, channels = images.shape
        print(batch_size, height, width, channels)
        # # Create 2 filters
        filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
        filters[:, 3, :, 0] = 1  # vertical line
        filters[3, :, :, 1] = 1  # horizontal line

        outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
        outputs2 = tf.nn.conv2d(images, filters, strides=1, padding="VALID")

        # plt.imshow(outputs[0, :, :, 1], cmap="gray")  # plot 1st image's 2nd feature
        # plt.show()
        # plt.imshow(outputs2[0, :, :, 1], cmap="gray")  # plot 1st image's 2nd feature
        # plt.show()

        # for image_index in (0, 1):
        #     for feature_map_index in (0, 1):
        #         plt.subplot(2, 2, image_index * 2 + feature_map_index + 1)
        #         plot_image(outputs[image_index, :, :, feature_map_index])
        # plt.show()

        # plot_image(crop(images[0, :, :, 0]))
        # plt.show()
        #
        # for feature_map_index, filename in enumerate(["china_vertical", "china_horizontal"]):
        #     plot_image(crop(outputs[0, :, :, feature_map_index]))
        #     plt.show()
        #
        # plot_image(filters[:, :, 0, 0])
        # plt.show()
        # plot_image(filters[:, :, 0, 1])
        # plt.show()

        # using keras layers conv2d
        # np.random.seed(42)
        # tf.random.set_seed(42)
        #
        # conv = keras.layers.Conv2D(filters=2, kernel_size=7, strides=1,
        #                            padding="SAME", activation="relu", input_shape=outputs.shape)
        #
        # conv_outputs = conv(images)
        # print(conv_outputs.shape)


def program1():
    TrainingExample()


if __name__ == '__main__':
    program1()
