# -- coding: utf-8 --
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import cv2
from tensorflow.contrib.slim.python.slim.nets import vgg_size512
from tensorflow.contrib.slim.python.slim.nets import inception_resnet_v2
slim = tf.contrib.slim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CROP_SIZE = 512
EPS = 1e-12

configuration_seed = None
configuration_mode = "test"
configuration_output_dir = "./test_dataset/test_output/"
configuration_mode = "test"
configuration_checkpoint = "./model/gan_train_waike_0308/"
configuration_size = CROP_SIZE
configuration_flip = False
configuration_input_dir = "./test_dataset/test_input/"


class configuration:
    def __init__(self, mode, seed, output_dir, configuration_checkpoint, size, flip):
        self.mode = mode
        self.seed = seed
        self.output_dir = output_dir
        self.checkpoint = configuration_checkpoint
        self.size = size
        self.flip = flip
        self.input_dir = configuration_input_dir
        self.max_epochs = None
        self.max_steps = None
        self.summary_freq = 50
        self.trace_freq = 0
        self.display_freq = 0
        self.save_freq = 5000
        self.aspect_ratio = 1.0
        self.lab_colorization = None
        self.batch_size = 1
        self.which_direction = "AtoB"
        self.ngf = 64
        self.ndf = 64
        self.scale_size = 512
        self.lr = 0.0002
        self.beta1 = 0.5
        self.l1_weight = 100.0
        self.gan_weight = 1.0
        self.output_filetype = "png"


a = configuration(configuration_mode, configuration_seed, configuration_output_dir, configuration_checkpoint,
                  configuration_size, configuration_flip)
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model",
                               "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def seg_crop(pic_name):
	image = cv2.imread(pic_name)
	#image=cv2.resize(image,(image.shape[1]/3,image.shape[0]/3))
	image2=np.zeros(image.shape,np.uint8)
	image2=image.copy()
	shape=image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	#cv2.imshow("gray",gray)
	(_, thresh) = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)


	#cv2.imshow("thresh",thresh)

	thresh_bit = cv2.bitwise_not(thresh)


	(cnts, _) = cv2.findContours(thresh_bit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
	cv2.drawContours(image2, c, -1, (0, 255, 0), 5)
	#print(len(cnts))

	stencil = np.zeros(image.shape).astype(image.dtype)
	color=[255,255,255]

	cv2.fillConvexPoly(stencil,c,color)
	#cv2.imshow("stencil",stencil)
	#cv2.waitKey()

	result=cv2.bitwise_and(image2,stencil)

	#cv2.imshow("result",result)

	x,y,w,h=cv2.boundingRect(c)
	cropImg = image[y:y+h, x:x+w]
	#cv2.imshow("crop",cropImg)

	cropImg_gray=cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
	(_, thresh_crop) = cv2.threshold(cropImg_gray, 230, 255, cv2.THRESH_BINARY)
	#cv2.imshow("thresh_crop",thresh_crop)
	thresh_bit_crop = cv2.bitwise_not(thresh_crop)
	(cnts_crop, _) = cv2.findContours(thresh_bit_crop.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c_crop = sorted(cnts_crop, key=cv2.contourArea, reverse=True)[0]

	leftmost = tuple(c_crop[c_crop[:, :, 0].argmin()][0])
	rightmost = tuple(c_crop[c_crop[:, :, 0].argmax()][0])
	topmost = tuple(c_crop[c_crop[:, :, 1].argmin()][0])
	bottommost = tuple(c_crop[c_crop[:, :, 1].argmax()][0])

	height,width,channel=cropImg.shape
	topleft=(0,0)
	topright=(width,0)
	bottomleft=(0,height)
	bottomright=(width,height)

	top_side_l=topmost[0]
	top_side_r=width-topmost[0]

	bottom_side_l=bottommost[0]
	bottom_side_r=width-bottommost[0]
	mask_top_left=()
	mask_top_right=()
	mask_bottom_left=()
	mask_bottom_right=()

	if top_side_l<top_side_r:
		mask_top_left=topleft
		mask_top_right=(topmost[0],0)

		mask_bottom_left=bottomleft
		mask_bottom_right=(bottommost[0],height)
	else:
		mask_top_left=(topmost[0],0)
		mask_top_right=topright

		mask_bottom_left=(bottommost[0],height)
		mask_bottom_right=bottomright

	mask_crop=np.zeros(cropImg.shape,dtype=np.uint8)
	#roi_conners=np.array([[[mask_top_left],[mask_top_right],[mask_bottom_left],[mask_bottom_right]]],dtype=np.uint32)
	roi_conners=np.array([mask_top_left,mask_top_right,mask_bottom_right,mask_bottom_left],dtype=np.int32)
	#roi_conners = np.array(c_crop,dtype=np.uint32)
	white=[255,255,255]
	cv2.fillConvexPoly(mask_crop,roi_conners,white)

	masked_crop_image=cv2.bitwise_and(cropImg,mask_crop)
	#cv2.imshow("mask_crop_image",masked_crop_image)
	result_image=[]
	if top_side_l<top_side_r:
		result_image=masked_crop_image[0:height,0:bottommost[0]]
	else:
		result_image=masked_crop_image[0:height,bottommost[0]:bottommost[0]+width]

	return result_image



def GetFileNameAndExt(filename):
    (filepath, tempfilename) = os.path.split(filename);
    (shotname, extension) = os.path.splitext(tempfilename);
    return filepath, shotname, extension


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels],
                                      [1, 2, 2, 1], padding="SAME")
        return conv


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1



def load_examples(input_paths):

    #
    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name
    #
    # # if the image names are numbers, sort by the value rather than asciibetically
    # # having sorted inputs means that the outputs are sorted in test mode
    # if all(get_name(path).isdigit() for path in input_paths):
    #     input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    # else:
    #     input_paths = sorted(input_paths)
    #
    with tf.name_scope("load_images"):
        # path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        # wei changed
        #line_dataset = tf.data.TextLineDataset(input_paths)


        raw_input = tf.image.decode_png(tf.read_file(input_paths))



        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1]  # [height, width, channels]
            a_images = preprocess(raw_input[:, :width // 2, :])
            b_images = preprocess(raw_input[:, width // 2:, :])


    paths="None"

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)),
                         dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)
        print ("input image:", input_images)

    with tf.name_scope("target_images"):
        target_images = transform(targets)
        print ("target_images:", target_images)

    # paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=1)
    #inputs_batch, targets_batch = tf.train.batch([input_images, target_images], batch_size=1)
    print("expand before input_image",input_images)
    inputs_batch=tf.expand_dims(input_images, 0)
    print("expand after input image",inputs_batch)
    print("expand before target_images",target_images)
    targets_batch=tf.expand_dims(target_images, 0)
    print("expand after targets batch",targets_batch)


    return inputs_batch,targets_batch




def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((
                                                                 srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])

            epsilon = 6 / 29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + (
                                                                                                  xyz_normalized_pixels ** (
                                                                                                  1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []



    for i, in_path in enumerate([fetches["paths"]]):
        name, _ = os.path.splitext(os.path.basename(str(in_path).decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["outputs"]:
            # for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


# main():
if tf.__version__.split('.')[0] != "1":
    raise Exception("Tensorflow version 1 required")

# a=configuration(configuration_mode,configuration_seed,configuration_output_dir,configuration_checkpoint,configuration_size,configuration_flip)
if a.seed is None:
    a.seed = random.randint(0, 2 ** 31 - 1)

middle = a.seed
tf.set_random_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)

if not os.path.exists(a.output_dir):
    os.makedirs(a.output_dir)

if a.mode == "test":
    if a.checkpoint is None:
        raise Exception("checkpoint needed")

options = {"which_direction", "ngf", "ndf", "lab_colorization"}

#examples = load_examples()

with open(os.path.join(a.checkpoint, "options.json")) as f:
    for key, val in json.loads(f.read()).items():
        if key in options:
            print("loaded", key, "=", val)
            setattr(a, key, val)




def convert(image):
    if a.aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)




with tf.name_scope("parameter_count"):
    parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])


class PredictPixtoPix:
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.seed = random.randint(0, 2 ** 31 - 1)
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            self.example_paths = tf.placeholder(tf.string, None, 'image_path')

            #self.examples = load_examples(input_paths)
            self.example_inputs,self.example_targets=load_examples(self.example_paths)
            #self.model = create_model(self.examples.inputs, self.examples.targets)
            # #self.example_inputs,self.example_targets = load_name2image(self.example_paths)
            self.model=create_model(self.example_inputs,self.example_targets)
            self.inputs = deprocess(self.example_inputs)
            self.targets = deprocess(self.example_targets)
            self.outputs = deprocess(self.model.outputs)


            self.display_fetches = {
                "paths": self.example_paths,
                "inputs": tf.map_fn(tf.image.encode_png,
                                    tf.image.convert_image_dtype(self.inputs, dtype=tf.uint8, saturate=True),
                                    dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png,
                                     tf.image.convert_image_dtype(self.targets, dtype=tf.uint8, saturate=True),
                                     dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png,
                                     tf.image.convert_image_dtype(self.outputs, dtype=tf.uint8, saturate=True),
                                     dtype=tf.string, name="output_pngs"),
            }
            self.saver = tf.train.Saver(max_to_keep=1)
            self.logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
            self.sv = tf.train.Supervisor(logdir=self.logdir, save_summaries_secs=0, saver=None)
            self.sess = self.sv.prepare_or_wait_for_session()
            self.checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            #self.sess.run(tf.initialize_all_variables())
            self.saver.restore(self.sess, self.checkpoint)


    def Predict(self,input_path_input,op,input_path):
        # print ("input_path_original :", input_path_input)
        # print("input_path:", input_path)
        results=self.sess.run(op,feed_dict={input_path:input_path_input})
        filesets=save_images(results)

INPUT_SIZE_VGG=512
NUM_CLASS_VGG=2
class PredictVgg16:
    def __init__(self):
        self.graph=tf.Graph()

        with self.graph.as_default():
            self.image=tf.placeholder(tf.float32,shape=[INPUT_SIZE_VGG,INPUT_SIZE_VGG,3])
            self.image_middle = tf.cast(self.image, tf.float32) * (1. / 255) - 0.5
            self.image_reshape=tf.reshape(self.image_middle,[-1, INPUT_SIZE_VGG, INPUT_SIZE_VGG,3])
            self.logits, self.description = vgg_size512.vgg_16(self.image_reshape, num_classes=NUM_CLASS_VGG, is_training=False)
            self.eps = tf.constant(value=1e-10)
            self.flat_logits = self.logits + self.eps
            self.softmax = tf.nn.softmax(self.flat_logits)
            self.probability,self.label_out=tf.nn.top_k(tf.nn.softmax(self.flat_logits),k=1)
            self.slim = tf.contrib.slim
            self.variables_to_restore = self.slim.get_variables_to_restore()
            self.saver=tf.train.Saver(self.variables_to_restore)
            self.saver = tf.train.import_meta_graph('./model/vgg_16_4gan/vgg_16_all-ft_model_8000_1.0.meta')
            self.sess=tf.Session()
        with self.sess.as_default():
            with self.graph.as_default():
                self.checkpoint_file=tf.train.latest_checkpoint("./model/vgg_16_4gan/")
                self.saver.restore(self.sess,self.checkpoint_file)

    def predict(self,image_input_ori):

        image_input_resize = cv2.resize(image_input_ori, (INPUT_SIZE_VGG, INPUT_SIZE_VGG))
        #cv2.imshow("image_seg_resize",image_input_resize)
        class_o, probability_o = self.sess.run([self.label_out, self.probability], feed_dict={self.image: image_input_resize})
        return class_o,probability_o


INPUT_SIZE_INCEPTRES=448
NUM_CLASS_SIZE_INCEPTRES = 11

class PredictInceptRes:
    def __init__(self):
        self.graph=tf.Graph()

        with self.graph.as_default():
            self.image=tf.placeholder(tf.float32,shape=[INPUT_SIZE_INCEPTRES,INPUT_SIZE_INCEPTRES,3])
            #self.image_middle = tf.cast(self.image, tf.float32) * (1. / 255) - 0.5
            self.image_reshape=tf.reshape(self.image,[-1,INPUT_SIZE_INCEPTRES,INPUT_SIZE_INCEPTRES,3])
            with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()) as scope:
                self.logits, self.description = inception_resnet_v2.inception_resnet_v2(self.image_reshape, num_classes=NUM_CLASS_SIZE_INCEPTRES,
                                                                              is_training=False)
            #self.logits, self.description = vgg_size512.vgg_16(self.image_reshape, num_classes=NUM_CLASS_SIZE_INCEPTRES, is_training=False)
            self.eps = tf.constant(value=1e-10)
            self.flat_logits = self.logits + self.eps
            self.softmax = tf.nn.softmax(self.flat_logits)
            self.probability,self.label_out=tf.nn.top_k(tf.nn.softmax(self.flat_logits),k=5)
            self.slim = tf.contrib.slim
            self.variables_to_restore = self.slim.get_variables_to_restore()
            self.saver=tf.train.Saver(self.variables_to_restore)
            #self.saver = tf.train.import_meta_graph('/media/goerlab/My Passport/Welder_detection/code/20180226_Inception448_Neroro/model/model_0313_2/model.ckpt-31500.meta')
            self.sess=tf.Session()
        with self.sess.as_default():
            with self.graph.as_default():
                self.checkpoint_file=tf.train.latest_checkpoint("/media/goerlab/My Passport/Welder_detection/code/20180226_Inception448_Neroro/model/model_0313_2/")
                self.saver.restore(self.sess,self.checkpoint_file)

    def predict(self,image_path):
        fd = open(image_path, 'rb')
        image_byte_arry = bytearray(fd .read())
        nba = np.asarray(image_byte_arry, dtype=np.uint8)
        image_input_ori = cv2.imdecode(nba, 1)
        image_input_resize = cv2.resize(image_input_ori, (INPUT_SIZE_INCEPTRES, INPUT_SIZE_INCEPTRES))
        class_o, probability_o = self.sess.run([self.label_out, self.probability], feed_dict={self.image: image_input_resize})
        return class_o,probability_o



if __name__ == "__main__":

    input_queue_dir="/media/goerlab/My Passport/20180211_HistoryImage/HistoryImage/20180315/20180315-1xian-A-1/OK/0/jpg/"

    seg_dir = "./test_dataset/test_output/images/"


    pixpredict = PredictPixtoPix()

    VGG_predict=PredictVgg16()

    InceptRes_predict=PredictInceptRes()

    for i in os.listdir(input_queue_dir):
        start1 = time.time()
        image_origin = cv2.imread(input_queue_dir+i)

        incept_class,incept_probability=InceptRes_predict.predict(input_queue_dir+i)

        print("file:",str(input_queue_dir+i))
        print("Incept predict label:",incept_class)
        print("Incept predict probability:",incept_probability)
        #print("predict label:%d, predict probability:%f" %(incept_class[0],incept_probability[0]))
        if incept_class[0][0]==0:
            origin = cv2.resize(image_origin, (512, 512))

            blank_image = np.zeros((512, 512, 3), np.uint8)
            vis = np.concatenate((origin, blank_image), axis=1)
            file_name,extension=os.path.splitext(i)
            new_name=configuration_input_dir+file_name+".png"
            #print("gan_file:",new_name)
            cv2.imwrite(new_name, vis)
            #start1=time.time()
            pixpredict.Predict(new_name, pixpredict.display_fetches, pixpredict.example_paths)
            #seg_dir="./test_dataset/test_output/images/"
            seg_name=seg_dir+file_name+"-outputs.png"
            seg_image=cv2.imread(seg_name)
            #cv2.imshow("seg_origin",seg_image)
            crop_seg_image=seg_crop(seg_name)
            #cv2.imshow("crop_seg_image",crop_seg_image)
            #cv2.waitKey()

            class_o1, probability_o1 = VGG_predict.predict(crop_seg_image)

            #cv2.waitKey()

            print("Waike predict",class_o1)
            print ("Waike probability:",probability_o1)
        else:
            pass

        end1 = time.time()
        dur1 = end1 - start1
        print("gan_dur1:", dur1)


        #pixpredict.Pridict(pixpredict.display_fetches)
        # if os.path.exists(new_name):
        #     # 删除文件，可使用以下两种方法。
        #     os.remove(new_name)

