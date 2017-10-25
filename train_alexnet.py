from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy as np
import datetime
import os
import re

import tensorflow as tf

import global_define as gd
#import Alexnet
import PIL
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from PIL import Image
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
#from sklearn import datasets, metrics, cross_validation
import sklearn as sk
#from tensorflow.models.inception.inception.slim import scopes 
#from utils import tile_raster_images
from tensorflow.contrib.slim.python.slim.nets import alexnet
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
slim = tf.contrib.slim
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
#TEST_FILE='test.tfrecords'

flags = tf.app.flags
FLAGS = flags.FLAGS

time_value=re.sub(r'[^0-7]','',str(datetime.datetime.now()))
flags.DEFINE_string('tfrecord_dir', 
	'/home/goerlab/Welder_detection/dataset/20171023/tfrecord/', 'Directory to put the training data.')
#flags.DEFINE_string('filename', 'train.tfrecords', 'Directory to put the training data.')
flags.DEFINE_integer('batch_size',100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('num_epochs', None, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')

flags.DEFINE_integer('learning_rate', 0.02,'balabala')
flags.DEFINE_integer('max_steps', 50000,'balabala')
flags.DEFINE_string('model_dir','Modal/model'+str(time_value)+'/','balabala')
flags.DEFINE_string('tensorevents_dir','tensorboard_event/event_wth'+str(time_value)+'/','balabala')
flags.DEFINE_string('log_dir','Log_data/log'+str(time_value)+'/','balabala')
flags.DEFINE_string('pic_dir','Pic/Pictures_input'+str(time_value)+'/','balabala')

if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

if not os.path.exists(FLAGS.tensorevents_dir):
  os.makedirs(FLAGS.tensorevents_dir)

if not os.path.exists(FLAGS.model_dir):
	os.makedirs(FLAGS.model_dir)



if not os.path.exists(FLAGS.pic_dir):
	os.makedirs(FLAGS.pic_dir)

def read_and_decode(filename_queue):

	reader=tf.TFRecordReader()
	_,serialized_exampe=reader.read(filename_queue)
	features=tf.parse_single_example(serialized_exampe,
		features={
		'image_raw':tf.FixedLenFeature([],tf.string),
		'height':tf.FixedLenFeature([],tf.int64),
		'width':tf.FixedLenFeature([],tf.int64),
		'depth':tf.FixedLenFeature([],tf.int64),
		'label':tf.FixedLenFeature([],tf.int64)
		})
	image=tf.decode_raw(features['image_raw'],tf.uint8)
	print("shape:")
	print(tf.shape(image))
	#tf.reshape(image,[224,224,3])
	#print("decode images after set_shape:")
	#print(str(tf.shape(image)))
	#image.set_shape([gd.IMAGE_PIXELS])
	image.set_shape([gd.IMAGE_PIXELS*3])
	print("read and decode:")
	print(image)
	image=tf.cast(image,tf.float32)*(1./255)-0.5
	label=tf.cast(features['label'],tf.int32)
	return image,label

def do_eval(sess,eval_correct,log_name):
	true_count=0
	for step in xrange(FLAGS.batch_size):
		true_count+=sess.run(eval_correct)

	precision=float(true_count)/FLAGS.batch_size/FLAGS.batch_size
	# print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
 #            (FLAGS.batch_size, true_count, precision))
 	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f\n' %
            (FLAGS.batch_size*FLAGS.batch_size, true_count, precision))
	logfile=open(log_name,'a')
	logfile.write('  Num examples: %d  Num correct: %d  Precision : %0.04f\n' %
            (FLAGS.batch_size, true_count, precision))
	
	logfile.close()
	return precision

def calc_loss(logits,labels):
	batch_size=tf.size(labels)
	labels=tf.expand_dims(labels,1)
	indices=tf.expand_dims(tf.range(0,batch_size),1)

	concated=tf.concat([indices,labels],1)
	onehot_labels=tf.sparse_to_dense(concated,tf.stack([batch_size,gd.NUM_CLASSES]),1.0,0.0)
	print('onehot_labels:')
	print(onehot_labels)
	print('logits:')
	print(logits)
	cross_entropy=slim.losses.softmax_cross_entropy(logits,onehot_labels)
	loss=tf.reduce_mean(cross_entropy,name='xentropy_mean')
	tf.summary.scalar('xentropy_mean',loss)
	return loss



def inputs(train,batch_size,num_epochs):
	if not num_epochs:num_epochs=None
	if train=='train':
		filename=os.path.join(FLAGS.tfrecord_dir,gd.TRAIN_FILE)
	elif train=='validation':
		filename=os.path.join(FLAGS.tfrecord_dir,gd.VALIDATION_FILE)
	else:
		filename=os.path.join(FLAGS.tfrecord_dir,gd.TEST_FILE)

	with tf.name_scope('input'):
		filename_queue=tf.train.string_input_producer([filename],num_epochs=None)
		print(filename)
		image,label=read_and_decode(filename_queue)
		print("input image:")
		print(image)
		images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=1,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000)
	return images, sparse_labels

def run_training():

	with tf.Graph().as_default():
		with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
		#with slim.arg_scope(resnet_v2.resnet_arg_scope()):
			images,labels=inputs(train='train',batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
			images_test,labels_test=inputs(train='val',batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
			images=tf.reshape(images,[-1,gd.INPUT_SIZE,gd.INPUT_SIZE,3])
			print("images:")
			print(images)

			#logits,description=resnet_v2.resnet_v2_101(images,4,is_training=True)
			logits,description=alexnet.alexnet_v2(images,num_classes=4)
			print('logits:')
			print(logits)
			print('labels:')
			print(labels)
			#loss=slim.losses.softmax_cross_entropy(logits, labels)
			loss=calc_loss(logits,labels)
			optimizer=tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

			train_op=slim.learning.create_train_op(loss,optimizer)

			logdir=FLAGS.log_dir

			slim.learning.train(train_op,logdir,number_of_steps=1000,
				save_summaries_secs=300,save_interval_secs=600)

			#summary_op=tf.summary.merge_all()

		init_op=tf.initialize_all_variables()

		saver=tf.train.Saver()
		config=tf.ConfigProto()
		config.gpu_options.allow_growth=True

		with tf.Session(config=config) as sess:
			sess.run(init_op)

if __name__=="__main__":
	run_training()


