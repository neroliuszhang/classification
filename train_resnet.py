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
flags.DEFINE_integer('batch_size',2, 'Batch size.  '
                     'Must divide evenly into theQ dataset sizes.')
flags.DEFINE_integer('num_epochs', None, 'Batch size.  '
                     'Must divide evenly into theu dataset sizes.')

flags.DEFINE_integer('learning_rate', 0.02,'balabeala')
flags.DEFINE_integer('max_steps', 50000,'balabala')
flags.DEFINE_string('model_dir','Modal/model'+str(time_value)+'/','balabeala')
flags.DEFINE_string('tensorevents_dir','tensorboard_event/event_wth'+str(time_value)+'/','balabala')
flags.DEFINE_string('log_dir','Log_data/log'+str(time_value)+'/','balabnala')
flags.DEFINE_string('pic_dir','Pic/Pictures_input'+str(time_value)+'/','balabLala')

if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

log_name=FLAGS.log_dir+'/'+"log.txt"
f=open(log_name,'w')
f.close()


# if not os.path.exists(FLAGS.tensorevents_dir):
#   os.makedirs(FLAGS.tensorevents_dir)

# if not os.path.exists(FLAGS.model_dir):
# 	os.makedirs(FLAGS.model_dir)



# if not os.path.exists(FLAGS.pic_dir):
# 	os.makedirs(FLAGS.pic_dir)

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
	logits_calc=tf.squeeze(logits)
	cross_entropy=slim.losses.softmax_cross_entropy(logits_calc,onehot_labels)
	loss=tf.reduce_mean(cross_entropy,name='xentropy_mean')
	tf.summary.scalar('xentropy_mean',loss)
	return loss

def evaluation(logits_in,labels):
	logits_in=tf.squeeze(logits_in)
	correct=tf.nn.in_top_k(logits_in,labels,1)
	tf.summary.scalar('evaluation',tf.reduce_sum(tf.cast(correct,tf.int32)))
	return tf.reduce_sum(tf.cast(correct,tf.int32))



def inputs(train,batch_size,num_epochs):
	if not num_epochs:num_epochs=None
	if train=='train':
		filename=os.path.join(FLAGS.tfrecord_dir,gd.TRAIN_FILE)
	elif train=='val':
		filename=os.path.join(FLAGS.tfrecord_dir,gd.VALIDATION_FILE)
	# else:
	# 	filename=os.path.join(FLAGS.tfrecord_dir,gd.TEST_FILE)

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
		#with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
		with slim.arg_scope(resnet_v2.resnet_arg_scope()):
			
			images,labels=inputs(train='train',batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
			
			images_val,labels_val=inputs(train='val',batch_size=FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
			
			images=tf.reshape(images,[-1,gd.INPUT_SIZE,gd.INPUT_SIZE,3])
			# print("images:")
			# print(images)

			logits,description=resnet_v2.resnet_v2_101(images,4,is_training=True)
			#logits,description=alexnet.alexnet_v2(images,num_classes=4)
			# print('logits:')
			# print(logits)
			# print('labels:')
			# print(labels)

			tf.get_variable_scope().reuse_variables()

			images_val=tf.reshape(images_val,[-1,gd.INPUT_SIZE,gd.INPUT_SIZE,3])
			#loss=slim.losses.softmax_cross_entropy(logits, labels)
			logits_val,_=resnet_v2.resnet_v2_101(images_val,4,is_training=True)
			
			loss=calc_loss(logits,labels)
			
			optimizer=tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

			global_step=tf.Variable(0,name='global_step',trainable=False)

			train_op=optimizer.minimize(loss,global_step=global_step)			
			eval_correct=evaluation(logits,labels)

			eval_correct_eval=evaluation(logits_val,labels_val)
			#train_op=slim.learning.create_train_op(loss,optimizer)

			#logdir=FLAGS.log_dir

			# slim.learning.train(train_op,logdir,number_of_steps=1000,
			# 	save_summaries_secs=300,save_interval_secs=600)

			summary_op=tf.summary.merge_all()

		init_op=tf.initialize_all_variables()

		saver=tf.train.Saver()
		config=tf.ConfigProto()
		config.gpu_options.allow_growth=True

		with tf.Session(config=config) as sess:
			sess.run(init_op)
			summary_writer=tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
			coord=tf.train.Coordinator()

			threads=tf.train.start_queue_runners(sess=sess,coord=coord)

			try:
				step=0
				while not coord.should_stop():
					start_time=time.time()
					_,loss_value=sess.run([train_op,loss])
					if step%100 ==0:
						summary_str=sess.run(summary_op)
						summary_writer.add_summary(summary_str,step)
						# print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
						#                                  duration))
						print('step %d : loss = %.2f' %(step,loss_value))
						logfile=open(log_name,'a')
						logfile.write('Step %d: loss = %.2f \n' % (step, loss_value))
						logfile.close()

					if step%1000==0 or step == FLAGS.max_steps:
						logfile=open(log_name,'a')
						logfile.write('Train:\n')
						logfile.close()

						do_eval(sess,eval_correct,log_name)

						logfile=open(log_name,'a')
						logfile.write('Test:\n')
						logfile.close()

						precision_test=do_eval(sess,eval_correct_eval,log_name)
						summary_str=sess.run(summary_op)
						summary_writer.add_summary(summary_str,step)
				step+=1
			except tf.errors.OutOfRangeError:
				f.write('Done training for  epochs,steps.\n' )
			finally:
				coord.request_stop()

			coord.join(threads)

if __name__=="__main__":
	run_training()


