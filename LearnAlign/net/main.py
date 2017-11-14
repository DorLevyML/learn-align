'''
this file is based on the file: fully_connected_feed.py
from: tensorflow/examples/tutorials/mnist/
of the TensorFlow Mechanics 101 tutorial,
and was modified from it.
'''

# imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

import time
import datetime
import numpy as np
import tensorflow as tf
import os
import sys

this_file_path = os.path.realpath(__file__)
netDir  = os.path.dirname(this_file_path)
projDir = os.path.dirname(netDir)
repDir  = os.path.dirname(projDir)
sys.path.append(repDir)
from LearnAlign.net  import AlignmentModel
from LearnAlign.data import DataEngine, utils


# configuration flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# model architecture
flags.DEFINE_string('task', 'art', 'Task to train the model for (art/nat/tsp5/tsp10).')
flags.DEFINE_boolean('birnn', False, 'Whether RNN is bi-directional.')
flags.DEFINE_string('decoder', 'cnn', 'decoder to use (rnn/cnn)')
flags.DEFINE_string('rnn_dec_type', 'match_lstm', 'type of rnn decoder (local/ptr1/ptr2/match_lstm)')

# model hyper-parameters
flags.DEFINE_integer('num_layers', 1, 'number of lstm layers')
flags.DEFINE_integer('lstm_width', 128, 'number of units in one lstm layer')
flags.DEFINE_integer('batch_size', 32, 'Batch size [samples]')
flags.DEFINE_float('clip_gradients', -1, 'Gradient clipping (-1 indicates none).')
flags.DEFINE_float('rnn_dropout', -1, 'Dropout rate in RNN (unit fraction to DROP. -1 indicates none).')
flags.DEFINE_float('cnn_dropout', -1, 'Dropout rate in CNN (unit fraction to DROP. -1 indicates none).')
flags.DEFINE_integer('max_seq_len_art', -1, 'maximum sequence length for artificial data [steps]')
flags.DEFINE_integer('max_seq_len_nat', 450, 'maximum sequence length for natural data [steps]')
flags.DEFINE_integer('max_seq_len_ptr', -1, 'maximum sequence length for PtrNets data [steps]')
flags.DEFINE_integer('beam_size', 10, 'beam size for beam search algorithm (in TSP).')
flags.DEFINE_boolean('perm', True, 'Whether to randomly permute the nodes of every TSP sample in the train set.')

# train & eval options
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_boolean('train', True, 'Whether to train the model or evaluate it.')
flags.DEFINE_boolean('load_existing', False, 'Whether to load an existing model for training.')
flags.DEFINE_string('input_dir', projDir+'/output/saved_experiment', 'Directory of saved experiment.')

# periods
flags.DEFINE_integer('max_steps', 9999999999, 'Max number of training steps.')
flags.DEFINE_integer('summary_n_overview_period', 20, 'Number of steps to write the summaries and print an overview.')
flags.DEFINE_integer('checkpoint_period', 20, 'Number of steps to save a checkpoint.')
flags.DEFINE_integer('eval_period', 1000, 'Number of steps to evaluate the model.')

# formats
flags.DEFINE_string('train_dir_format', projDir+'/output/%s/', 'Directory format to put the training data.')
flags.DEFINE_string('eval_dir_format', 'evaluations/%s/', 'Directory format to put the evaluation data.')


# evaluations

def eval(sess, model, data_set, output_dir, global_step=None):

	if FLAGS.task in ['art','nat']:
		eval_epoch_alignment(sess, model, data_set, output_dir, global_step=global_step)
	elif FLAGS.task in ['tsp5', 'tsp10']:
		eval_epoch_tsp_route_len(sess, model, data_set, output_dir, global_step=global_step)


def eval_epoch_alignment(sess, model, data_set, output_dir, global_step=None):
	'''
	evaluation of alignment accuracy across the whole given data set.
	'''

	print('Evaluating alignment for %s data set...' % data_set.name)
	true_count_total = 0.0  # number of correct predictions
	line_count_total = 0.0  # total number of lines
	steps_per_epoch = data_set.num_samples // FLAGS.batch_size
	num_samples = steps_per_epoch * FLAGS.batch_size
	start_time = datetime.datetime.now()

	for _ in xrange(steps_per_epoch):

		batches, len_batches, _ = data_set.next_batch()
		feed_dict = model.fill_feed_dict(batches, len_batches)

		loss, true_count, line_count, accuracy = sess.run([model.loss, model.true_count, model.line_count, model.accuracy], feed_dict=feed_dict)
		true_count_total += true_count
		line_count_total += line_count

	duration = datetime.datetime.now() - start_time
	precision = true_count_total / line_count_total
	
	txt = []
	if global_step:
		txt.append('Step: %d' % global_step)
	txt.append('Alignment evaluation results for %s data set-' % data_set.name)
	txt.append('Duration: %s' % duration)
	txt.append('Num samples: %d' %	num_samples)
	txt.append('Precision (over the whole epoch): %0.04f' % precision)
	txt.append('Accuracy (over one batch): %0.04f' % accuracy)
	txt.append('Loss (over one batch): %.2f' % loss)
	txt = '\n'.join(txt)
	
	print(txt)
	
	file_path = os.path.join(output_dir, 'eval_ali_data_set_%s.txt' % data_set.name)
	utils.WriteFile(txt, file_path)


def eval_epoch_tsp_route_len(sess, model, data_set, output_dir, global_step=None):
	'''
	evaluation of predicted route length across the whole given data set.
	'''

	print('Evaluating TSP route length for %s data set...' % data_set.name)
	if global_step is None:
		global_step = -1
	route_len_total = 0.0  # Counts the number of correct predictions.
	steps_per_epoch = data_set.num_samples // FLAGS.batch_size
	num_samples = steps_per_epoch * FLAGS.batch_size
	start_time = datetime.datetime.now()

	for _ in xrange(steps_per_epoch):
		batches, len_batches, _ = data_set.next_batch()
		feed_dict = model.fill_feed_dict(batches, len_batches)

		sample_batch, sparse_label_batch, dense_label_batch = batches
		len_sample_batch, len_label_batch = len_batches

		loss, pred_softmax = sess.run([model.loss, model.pred_softmax], feed_dict=feed_dict)
		for member in xrange(FLAGS.batch_size):
			route_len = eval_tsp_route_len_beam(sample_batch[member], pred_softmax[member], len_sample_batch[member])
			route_len_total += route_len

	duration = datetime.datetime.now() - start_time
	average_route_len = route_len_total / num_samples

	txt = []
	txt.append('Step: %d' % global_step)
	txt.append('TSP route length evaluation results for %s data set-' % data_set.name)
	txt.append('Duration: %s' % duration)
	txt.append('Num samples: %d' % num_samples)
	txt.append('Average route length (over the whole epoch): %0.04f' % average_route_len)
	txt = '\n'.join(txt)

	print(txt)

	file_path = os.path.join(output_dir, 'eval_tsp_data_set_%s.txt' % data_set.name)
	utils.WriteFile(txt, file_path)


def eval_tsp_route_len_beam(sample, soft_pred, sample_len):
	'''
	infer valid TSP route.
	calculate route length.
	'''

	def indices_of_top_k(_arr, _k):
		_k = min(_arr.shape[0], _k)
		return np.argpartition(_arr, -_k)[-_k:]

	def score(_route):
		_prob = 1
		for _i in xrange(len(_route) - 1):
			_src = _route[_i]
			_dst = _route[_i + 1]
			_prob *= soft_pred[_src, _dst]
		return _prob

	def calc_route_len(_route):
		_route_len = 0
		for _i in xrange(sample_len):
			_src = _route[_i]
			_dst = _route[_i+1]
			_x = sample[_src, :]
			_y = sample[_dst, :]
			_route_len += np.linalg.norm(_x - _y)
		return _route_len

	sample_len = int(sample_len)
	legal_starts = range(sample_len)
	beam = [[node] for node in legal_starts]

	for step in xrange(sample_len):

		# empty saved
		saved = []

		for route in beam:

			if step < sample_len - 1:
				# find all legal dsts
				legal_dsts = range(sample_len)
				for node in route:
					legal_dsts.remove(node)
			else:
				# close every route (add the first city)
				legal_dsts = [route[0]]

			# make routes of all new dsts
			for dst in legal_dsts:
				saved.append(route + [dst])

		# calculate the score of every route
		scores = [score(route) for route in saved]

		# select the k best scores, where k is the beam size
		if step < sample_len - 1:
			k = FLAGS.beam_size
		else:
			k = 1
		inds = indices_of_top_k(np.array(scores), k)

		# add those routes to the beam
		beam = [saved[ind] for ind in inds]

	route = beam[0]

	# return length of selected route
	return calc_route_len(route)


# utils

def LoadData():

	if FLAGS.task in ['tsp5', 'tsp10']:
		ptr_data_dir = os.path.join(projDir, 'DataSets/PtrNets_datasets')
		if FLAGS.task == 'tsp5':
			file_list_ptr_train = [os.path.join(ptr_data_dir, 'tsp5.txt')]
			file_list_ptr_test = [os.path.join(ptr_data_dir, 'tsp5_test.txt')]
		elif FLAGS.task == 'tsp10':
			file_list_ptr_train = [os.path.join(ptr_data_dir, 'tsp_10_train_exact.txt')]
			file_list_ptr_test  = [os.path.join(ptr_data_dir, 'tsp_10_test_exact.txt')]

		# validation dataset is the test dataset only for code compatibility. it is not used for validation.
		data_sets = DataEngine.TspDatasetsCollection(FLAGS.batch_size)
		data_sets.LoadFromFile('%s_train' % FLAGS.task, file_list_ptr_train, max_seq_len=FLAGS.max_seq_len_ptr, perm=FLAGS.perm)
		data_sets.LoadFromFile('%s_valid' % FLAGS.task, file_list_ptr_test,  max_seq_len=FLAGS.max_seq_len_ptr)
		data_sets.LoadFromFile('%s_test'  % FLAGS.task, file_list_ptr_test,  max_seq_len=FLAGS.max_seq_len_ptr)
		data_sets.Init()

		return data_sets

	elif FLAGS.task in ['art', 'nat']:

		# artificial data
		art_data_dir = os.path.join(projDir, 'DataSets/artificial_functions_datasets')

		file_list_art_train = [os.path.join(art_data_dir, 'ds_O1_train_39977.txt'),
													 os.path.join(art_data_dir, 'ds_O2_train_39945.txt'),
													 os.path.join(art_data_dir, 'ds_O3_train_39811.txt')]

		file_list_art_valid = [os.path.join(art_data_dir, 'ds_O1_valid_5012.txt'),
													 os.path.join(art_data_dir, 'ds_O2_valid_5078.txt'),
													 os.path.join(art_data_dir, 'ds_O3_valid_5030.txt')]

		file_list_art_test  = [os.path.join(art_data_dir, 'ds_O1_test_5011.txt'),
													 os.path.join(art_data_dir, 'ds_O2_test_4977.txt'),
													 os.path.join(art_data_dir, 'ds_O3_test_5159.txt')]

		# natural data
		nat_data_dir = os.path.join(projDir, 'DataSets/natural_projects_datasets')

		file_list_nat_train = [os.path.join(nat_data_dir, 'ds_O1_len450_train_17629.txt'),
													 os.path.join(nat_data_dir, 'ds_O2_len450_train_16008.txt'),
													 os.path.join(nat_data_dir, 'ds_O3_len450_train_8754.txt')]

		file_list_nat_valid = [os.path.join(nat_data_dir, 'ds_O1_len450_valid_2186.txt'),
													 os.path.join(nat_data_dir, 'ds_O2_len450_valid_2137.txt'),
													 os.path.join(nat_data_dir, 'ds_O3_len450_valid_1151.txt')]

		file_list_nat_test  = [os.path.join(nat_data_dir, 'ds_O1_len450_test_2190.txt'),
													 os.path.join(nat_data_dir, 'ds_O2_len450_test_2004.txt'),
													 os.path.join(nat_data_dir, 'ds_O3_len450_test_1059.txt')]

		# load
		data_sets = DataEngine.AlignmentDatasetsCollection(FLAGS.batch_size)
		data_sets.LoadFromFile('art_train', file_list_art_train, max_seq_len=FLAGS.max_seq_len_art)
		data_sets.LoadFromFile('art_valid', file_list_art_valid, max_seq_len=FLAGS.max_seq_len_art)
		data_sets.LoadFromFile('art_test',  file_list_art_test,  max_seq_len=FLAGS.max_seq_len_art)
		data_sets.LoadFromFile('nat_train', file_list_nat_train, max_seq_len=FLAGS.max_seq_len_nat)
		data_sets.LoadFromFile('nat_valid', file_list_nat_valid, max_seq_len=FLAGS.max_seq_len_nat)
		data_sets.LoadFromFile('nat_test',  file_list_nat_test,  max_seq_len=FLAGS.max_seq_len_nat)
		data_sets.Init()

		return data_sets


# main

def main(_):
	
	# load data
	print('loading data sets...'); start_time = datetime.datetime.now()
	data_sets = LoadData()
	print('data sets loaded (%s)' % (datetime.datetime.now() - start_time))

	# create session
	with tf.Graph().as_default(), tf.Session() as sess:
		
		# create model
		print('building model...'); start_time = datetime.datetime.now()
		if FLAGS.train:
			with tf.name_scope("Train"):
				with tf.variable_scope("Model"):
					m_train = AlignmentModel.Model(FLAGS, data_sets, is_training=True)
			with tf.name_scope("Eval"):
				with tf.variable_scope("Model", reuse=True):
					m_eval  = AlignmentModel.Model(FLAGS, data_sets, is_training=False)
		else:
			with tf.name_scope("Eval"):
				with tf.variable_scope("Model", reuse=None):
					m_eval  = AlignmentModel.Model(FLAGS, data_sets, is_training=False)
		print('model built (%s)' % (datetime.datetime.now() - start_time))

		# calculate number of free parameters
		dim_list = [v.get_shape().as_list() for v in tf.trainable_variables()]
		dim_list = [np.prod(np.array(d)) for d in dim_list]
		num_params = np.sum(np.array(dim_list))
		print('num_params: {:,}'.format(num_params))
		
		# create a saver for writing/loading checkpoints
		saver = tf.train.Saver()

		# train/test
		if FLAGS.train:
			run_training(sess, m_train, m_eval, data_sets, saver)
		else:
			run_evaluation(sess, m_eval, data_sets, saver)
		
		
def run_training(sess, m_train, m_eval, data_sets, saver):

	# data to use
	ds_train = '%s_train' % FLAGS.task
	ds_valid = '%s_valid' % FLAGS.task
	ds_test  = '%s_test'  % FLAGS.task
	
	# initialize variables, either with their ops or with saved values
	if FLAGS.load_existing:
		# load variables
		print('loading variables from checkpoint...'); start_time = datetime.datetime.now()
		latest_checkpoint = tf.train.latest_checkpoint(FLAGS.input_dir)
		saver.restore(sess, latest_checkpoint)
		print('variables loaded (%s)' % (datetime.datetime.now() - start_time))
	else:
		# run the variables initializers
		print('initializing variables...'); start_time = datetime.datetime.now()
		init = tf.global_variables_initializer()
		sess.run(init)
		print('variables initialized (%s)' % (datetime.datetime.now() - start_time))

	# SummaryWriters
	print('initializing SummaryWriters...'); start_time = datetime.datetime.now()
	output_dir = FLAGS.train_dir_format % utils.NowStr()
	train_summary_writer = tf.summary.FileWriter(output_dir+'train', sess.graph)
	eval_train_summary_writer = tf.summary.FileWriter(output_dir+'eval_train')
	eval_valid_summary_writer = tf.summary.FileWriter(output_dir+'eval_valid')
	eval_test_summary_writer  = tf.summary.FileWriter(output_dir+'eval_test')
	print('SummaryWriters initialized (%s)' % (datetime.datetime.now() - start_time))

	# train
	print('begin training')
	for step in xrange(FLAGS.max_steps):
		start_time = time.time()
		
		# fill feed dict
		batches, len_batches, _ = data_sets.Get(ds_train).next_batch()
		feed_dict = m_train.fill_feed_dict(batches, len_batches)
		
		# train step
		_, loss_value, acc_val = sess.run([m_train.train_op, m_train.loss, m_train.accuracy], feed_dict=feed_dict)

		duration = time.time() - start_time

		# summaries and overview
		if step % FLAGS.summary_n_overview_period == 0:
			# print status
			print('Step %d: loss = %.2f, accuracy = %.2f (%.3f sec)' % (step, loss_value, acc_val, duration))
			# update the events file
			summary_str = sess.run(m_train.summaries, feed_dict=feed_dict)
			train_summary_writer.add_summary(summary_str, step)

			feed_dict = m_eval.fill_feed_dict(batches, len_batches)
			summary_str = sess.run(m_eval.summaries, feed_dict=feed_dict)
			eval_train_summary_writer.add_summary(summary_str, step)

			batches, len_batches, _ = data_sets.Get(ds_valid).next_batch()
			feed_dict = m_eval.fill_feed_dict(batches, len_batches)
			summary_str = sess.run(m_eval.summaries, feed_dict=feed_dict)
			eval_valid_summary_writer.add_summary(summary_str, step)

			batches, len_batches, _ = data_sets.Get(ds_test).next_batch()
			feed_dict = m_eval.fill_feed_dict(batches, len_batches)
			summary_str = sess.run(m_eval.summaries, feed_dict=feed_dict)
			eval_test_summary_writer.add_summary(summary_str, step)

		# save checkpoint
		if (step + 1) % FLAGS.checkpoint_period == 0 or (step + 1) == FLAGS.max_steps:
			print('saving checkpoint...'); start_time = datetime.datetime.now()
			saver.save(sess, output_dir, global_step=step)
			print('checkpoint Saved (%s)' % (datetime.datetime.now() - start_time))

		# evaluate the model
		if (step + 1) % FLAGS.eval_period == 0 or (step + 1) == FLAGS.max_steps:
			# eval(sess, m_eval, data_sets.Get(ds_train), output_dir, step)
			eval(sess, m_eval, data_sets.Get(ds_valid), output_dir, step)
			# eval(sess, m_eval, data_sets.Get(ds_test),  output_dir, step)

	print('training finished.')


def run_evaluation(sess, m, data_sets, saver):
	
	# load variables
	print('loading variables from checkpoint...'); start_time = datetime.datetime.now()
	latest_checkpoint = tf.train.latest_checkpoint(FLAGS.input_dir)
	saver.restore(sess, latest_checkpoint)
	print('variables loaded (%s)' % (datetime.datetime.now() - start_time))
	
	# create output dir
	output_dir = os.path.join(FLAGS.input_dir, FLAGS.eval_dir_format % utils.NowStr())
	os.makedirs(output_dir)
	
	# evaluate

	if FLAGS.task in ['art', 'nat']:

		art_data_dir = os.path.join(projDir, 'DataSets/artificial_functions_datasets')
		nat_data_dir = os.path.join(projDir, 'DataSets/natural_projects_datasets')

		file_list_art_test  = [os.path.join(art_data_dir, 'ds_O1_test_5011.txt'),
													 os.path.join(art_data_dir, 'ds_O2_test_4977.txt'),
													 os.path.join(art_data_dir, 'ds_O3_test_5159.txt')]

		file_list_nat_test  = [os.path.join(nat_data_dir, 'ds_O1_len450_test_2190.txt'),
													 os.path.join(nat_data_dir, 'ds_O2_len450_test_2004.txt'),
													 os.path.join(nat_data_dir, 'ds_O3_len450_test_1059.txt')]

		data_sets.LoadFromFile('art_test_O1', [file_list_art_test[0]], restrict_len=True)
		data_sets.LoadFromFile('art_test_O2', [file_list_art_test[1]], restrict_len=True)
		data_sets.LoadFromFile('art_test_O3', [file_list_art_test[2]], restrict_len=True)
		data_sets.LoadFromFile('art_test_all', file_list_art_test,     restrict_len=True)

		data_sets.LoadFromFile('nat_test_O1', [file_list_nat_test[0]], restrict_len=True)
		data_sets.LoadFromFile('nat_test_O2', [file_list_nat_test[1]], restrict_len=True)
		data_sets.LoadFromFile('nat_test_O3', [file_list_nat_test[2]], restrict_len=True)
		data_sets.LoadFromFile('nat_test_all', file_list_nat_test,     restrict_len=True)

		data_sets.Init()

		for opt in ['O1', 'O2', 'O3', 'all']:
			name = '%s_test_%s' % (FLAGS.task, opt)
			eval_epoch_alignment(sess, m, data_sets.Get(name), output_dir)

	elif FLAGS.task in ['tsp5', 'tsp10']:
		name = '%s_test' % FLAGS.task
		eval_epoch_tsp_route_len(sess, m, data_sets.Get(name), output_dir)


if __name__ == '__main__':
	tf.app.run()
