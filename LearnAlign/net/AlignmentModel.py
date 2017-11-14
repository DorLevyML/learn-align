# imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import array_ops

# model

class Model(object):
	
	def __init__(self, FLAGS, data_sets, is_training=True):

		# architecture configuration
		self.task = FLAGS.task

		# hyper parameters
		self.birnn         = FLAGS.birnn
		self.num_layers    = FLAGS.num_layers
		self.lstm_width    = FLAGS.lstm_width
		
		# training parameters
		self.is_training   = is_training
		self.batch_size    = FLAGS.batch_size
		self.learning_rate = FLAGS.learning_rate
		self.decoder       = FLAGS.decoder
		self.rnn_dec_type  = FLAGS.rnn_dec_type
		self.clip_gradients = FLAGS.clip_gradients
		if self.clip_gradients == -1:
			self.clip_gradients = None
		self.cnn_dropout = FLAGS.cnn_dropout
		self.rnn_dropout = FLAGS.rnn_dropout

		# for summaries
		self.grad_norm_scopes = []

		# general
		if self.task in ['tsp5', 'tsp10']:
			self.max_sample_len = data_sets.max_sample_len
			self.max_label_len  = data_sets.max_label_len
			self.max_obj_len    = data_sets.max_sample_len
			self.max_src_len    = data_sets.max_sample_len + 1
			self.max_obj_lines  = self.max_obj_len
			self.max_src_lines  = self.max_src_len
		elif self.task in ['art', 'nat']:
			self.obj_vocab_size = data_sets.obj_vocab_size
			self.src_vocab_size = data_sets.src_vocab_size
			self.max_obj_len    = data_sets.max_obj_len
			self.max_src_len    = data_sets.max_src_len
			self.max_obj_lines  = data_sets.max_obj_lines
			self.max_src_lines  = data_sets.max_src_lines
		self.optimizer = None
		
		# graph inputs
		if self.task in ['tsp5', 'tsp10']:
			self.input           = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_sample_len, 2])
			self.target          = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_label_len])
			self.dense_target    = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_sample_len, self.max_sample_len+1])

			self.len_input       = tf.placeholder(tf.int32, shape=[self.batch_size])
			self.len_target      = tf.placeholder(tf.int32, shape=[self.batch_size])

		elif self.task in ['art', 'nat']:
			self.input_obj       = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_obj_len])
			self.input_src       = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_src_len])
			self.input_lines_obj = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_obj_lines])
			self.input_lines_src = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_src_lines])
			self.target          = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_obj_lines])

			self.len_input_obj       = tf.placeholder(tf.int32, shape=[self.batch_size])
			self.len_input_src       = tf.placeholder(tf.int32, shape=[self.batch_size])
			self.len_input_lines_obj = tf.placeholder(tf.int32, shape=[self.batch_size])
			self.len_input_lines_src = tf.placeholder(tf.int32, shape=[self.batch_size])
		
		# graph definition
		if self.task in ['tsp5', 'tsp10']:
			self.input_obj, self.input_src, self.len_input_obj, self.len_input_src = self.make_tsp_inputs(self.input, self.len_input)
			self.pred = self.inference_tsp(self.input_obj, self.input_src, self.len_input_obj, self.len_input_src)
			self.loss, self.pred_softmax, self.hard_pred, self.dense_target, self.true_count, self.line_count, self.accuracy = self.calc_loss(self.target, self.pred, self.len_input_obj, self.len_input_src, dense_targets=self.dense_target)
		elif self.task in ['art', 'nat']:
			self.pred = self.inference(self.input_obj, self.input_src, self.input_lines_obj, self.input_lines_src, self.len_input_obj, self.len_input_src)
			self.loss, self.pred_softmax, self.hard_pred, self.dense_target, self.true_count, self.line_count, self.accuracy = self.calc_loss(self.target, self.pred, self.len_input_lines_obj, self.len_input_lines_src)

		if self.is_training:
			self.train_op = self.training(self.loss)

		# summary
		accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
		loss_summary     = tf.summary.scalar('loss', self.loss)
		summary_list = [accuracy_summary, loss_summary]
		if self.is_training:
			grad_summaries = self.perform_grad_norm_summaries()
			summary_list += grad_summaries
		self.summaries = tf.summary.merge(summary_list)

		
	@property
	def hidden_size(self):
		if self.birnn:
			return 2 * self.lstm_width
		else:
			return self.lstm_width


	def fill_feed_dict(self, batches, len_batches):

		if self.task in ['tsp5', 'tsp10']:
			sample_batch, sparse_label_batch, dense_label_batch = batches
			len_sample_batch, len_label_batch = len_batches

			feed_dict = {
				self.input        : sample_batch,
				self.target       : sparse_label_batch,
				self.dense_target : dense_label_batch,

				self.len_input    : len_sample_batch,
				self.len_target   : len_label_batch
				}

			return feed_dict

		elif self.task in ['art', 'nat']:
			obj, src, objLines, srcLines, ali = batches
			len_obj, len_src, len_objLines, len_srcLines = len_batches

			feed_dict = {
				self.input_obj       : obj,
				self.input_src       : src,
				self.input_lines_obj : objLines,
				self.input_lines_src : srcLines,
				self.target          : ali,

				self.len_input_obj       : len_obj,
				self.len_input_src       : len_src,
				self.len_input_lines_obj : len_objLines,
				self.len_input_lines_src : len_srcLines
				}

			return feed_dict


	def add_scope_to_grad_norm_summaries(self):
		scope = tf.get_variable_scope()
		self.grad_norm_scopes.append(scope.name)


	def perform_grad_norm_summaries(self):
		grad_norm_summaries = []
		for scope_name in self.grad_norm_scopes:
			var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name)
			grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list)
			grads = [grad for (grad, var) in grads_and_vars]
			grad_norm = tf.global_norm(grads)
			summary = tf.summary.scalar('grad_norm_%s' % scope_name, grad_norm)
			grad_norm_summaries.append(summary)

		return grad_norm_summaries

		
	def lstm_cell(self):
		def single_cell():
			return tf.contrib.rnn.BasicLSTMCell(self.lstm_width, forget_bias=1.0)
		if self.num_layers == 1:
			return single_cell()
		else:
			return tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
		
		
	def representation(self, inputs, len_inputs):
		'''
		input  shape: [batch, max_len, lstm_width]
		output shape: [batch, max_len, lstm_width]
		'''

		inputs = tf.transpose(inputs, [1,0,2])
		inputs = tf.unstack(inputs)

		if self.birnn:
			with tf.variable_scope("fw"):
				cell_fw = self.lstm_cell()
			with tf.variable_scope("bw"):
				cell_bw = self.lstm_cell()
			outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, sequence_length=len_inputs)
			state = tf.concat([state_fw, state_bw], 1)
		else:
			cell = self.lstm_cell()
			outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32, sequence_length=len_inputs)
			
		outputs = tf.stack(outputs)
		outputs = tf.transpose(outputs, [1,0,2])

		if self.rnn_dropout != -1:
			outputs = tf.layers.dropout(outputs,
																	rate=self.rnn_dropout,
																	noise_shape=[self.batch_size, 1, self.hidden_size],
																	training=self.is_training)

		return outputs, state
	
	
	def encoder(self, vocab_size, inputs, len_inputs, line_marker):
		'''
		inputs.shape = [batch, max_seq_len]
		
		we increase vocab_size by 1, due to two reasons:
		- embedding_lookup matches the symbols (in the range [1,vocab_size]) to the embedding_matrix
			indices (in the range [0, vocab_size-1]), so symbol vocab_size is out of range
		- the sequences are padded by the out-of-vocabulary symbol 0
		'''
		
		# embed
		with tf.device("/cpu:0"):
			embedding_matrix = tf.get_variable('embedding', [vocab_size+1, self.lstm_width], initializer=tf.random_uniform_initializer(-1.0, 1.0))
			inference = tf.nn.embedding_lookup(embedding_matrix, inputs)
		
		# represent sequence
		inference, state = self.representation(inference, len_inputs)
		
		# sample line representaions
		inference = self.sample_line_reps(inference, line_marker)

		return inference, state


	def sample_line_reps(self, seq, lineMarker):
		'''
		seq:
		shape = [batch, max_len, lstm_width]
		
		lineMarker:
		shape = [batch, lineNum]
		'''
		
		# split batch:
		seq = tf.unstack(seq) # len=batch, shape=[max_len,lstm_width]
		lineMarker = tf.unstack(lineMarker) # len=batch, shape=[lineNum]
		
		seq = [tf.gather(seq[member], lineMarker[member]) for member in xrange(self.batch_size)] # len=batch, shape=[lineNum,lstm_width]
		seq = tf.stack(seq) # shape = [batch, lineNum, lstm_width]
		
		return seq
		
		
	def shared_fc_net(self, input_vector_weighted):
		'''
		in:
			input_vector_weighted.shape = [batch_size, lstm_width]
		out:
			logit.shape = [batch_size, 1]
		'''
		
		# build net
		hidden = tf.tanh(input_vector_weighted)
		logit_w = tf.get_variable('logit_weights', [self.hidden_size, 1], initializer=tf.random_uniform_initializer(-1.0, 1.0))
		logit = tf.matmul(hidden, logit_w)
		
		# share this model
		if tf.get_variable_scope().reuse == False:
			tf.get_variable_scope().reuse_variables()
		
		return logit


	def rnn_decoder(self, h_obj, h_src, last_state_of_src_enc):
		'''
		h_obj.shape = [batch, obj, lstm_width]
		h_src.shape = [batch, src, lstm_width]
		last_state_of_src_enc.shape = [batch, lstm_width]
		'''
		
		with tf.variable_scope('decoder'):
			
			# first, pre calculate here what's possible
			h_src_w = tf.get_variable('h_src_weights', [self.hidden_size, self.hidden_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
			h_src_weighted = [tf.matmul(h_src[:,j,:], h_src_w) for j in xrange(self.max_src_lines)]

			if self.rnn_dec_type in ['local', 'ptr2', 'match_lstm']:
				h_obj_w = tf.get_variable('h_obj_weights', [self.hidden_size, self.hidden_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
				h_obj_weighted = [tf.matmul(h_obj[:,j,:], h_obj_w) for j in xrange(self.max_obj_lines)]

			# now continue as usual
			if self.rnn_dec_type in ['ptr1', 'ptr2', 'match_lstm']:
				cell_output_w = tf.get_variable('cell_output_weights', [self.hidden_size, self.hidden_size], initializer=tf.random_uniform_initializer(-1.0, 1.0))
				cell = self.lstm_cell()
				last_pointed_src = tf.zeros([self.batch_size, self.hidden_size])
				state = last_state_of_src_enc
			if self.rnn_dec_type in ['ptr2', 'match_lstm']:
				last_obj = tf.zeros([self.batch_size, self.hidden_size])
			if self.rnn_dec_type == 'match_lstm':
				last_attention_over_src = tf.zeros([self.batch_size, self.hidden_size])

			alignment_pred = []
			for i in xrange(self.max_obj_lines):
				
				# decoder
				if self.rnn_dec_type == 'ptr1':
					ecnoder_input = tf.concat([last_pointed_src, h_obj[:,i,:]], 1)
				elif self.rnn_dec_type == 'ptr2':
					ecnoder_input = tf.concat([last_pointed_src, last_obj], 1)
				elif self.rnn_dec_type == 'match_lstm':
					ecnoder_input = tf.concat([last_attention_over_src, last_obj], 1)

				if self.rnn_dec_type in ['ptr1', 'ptr2', 'match_lstm']:
					cell_output, state = cell(ecnoder_input, state)
					cell_output_weighted = tf.matmul(cell_output, cell_output_w)

				if self.rnn_dec_type == 'local':
					dec_out = h_obj_weighted[i]
				elif self.rnn_dec_type == 'ptr1':
					dec_out = cell_output_weighted
				elif self.rnn_dec_type == 'ptr2':
					dec_out = cell_output_weighted + h_obj_weighted[i]
				elif self.rnn_dec_type == 'match_lstm':
					dec_out = cell_output_weighted + h_obj_weighted[i]

				# alignment model
				align_scores = [self.shared_fc_net(dec_out+src) for src in h_src_weighted]
				align_scores = [tf.squeeze(score) for score in align_scores]
				align_scores = tf.stack(align_scores, axis=1) # shape=[batch,src]
				alignment_pred.append(align_scores)
				
				# pointed src
				if self.rnn_dec_type in ['ptr1', 'ptr2']:
					if self.is_training:
						# at training time, use teacher forcing
						pointed_src_ind = self.target[:,i]
					else:
						# at evaluation time, use own predictions
						pointed_src_ind = tf.argmax(align_scores, dimension=1) # shape=[batch,1]
						pointed_src_ind = tf.squeeze(tf.to_int32(pointed_src_ind)) # shape=[batch]
					last_pointed_src = self.gather_batch(h_src, pointed_src_ind) # shape=[batch,lstm_width]

				# attention over src
				if self.rnn_dec_type == 'match_lstm':
					# attention coefficients
					alpha = tf.nn.softmax(align_scores) # shape = [batch, src]
					alpha = tf.expand_dims(alpha, axis=1) # shape = [batch, 1, src]
					# attention weighted sum
					last_attention_over_src = tf.matmul(alpha, h_src)  # shape = [batch, 1, lstm_width]
					last_attention_over_src = tf.squeeze(last_attention_over_src)

				# last obj
				if self.rnn_dec_type in ['ptr2', 'match_lstm']:
					last_obj = h_obj[:, i, :]

			return tf.stack(alignment_pred, axis=1)  # shape=[batch,obj,src]


	def gather_batch(self, batch, sparse_indices):
		'''
		in:
			batch.shape = [batch_size, src, lstm_width]
			sparse_indices.shape = [batch_size]

		out:
			shape = [batch_size, lstm_width]
		'''

		lin = tf.range(0, self.batch_size, 1)
		dense_indices = tf.stack([lin, sparse_indices], axis=1)

		return tf.gather_nd(batch, dense_indices)


	def conv(self, inputs, kernel_shape):
		weights = tf.get_variable("weights", kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
		return tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')


	def conv_bias_relu(self, inputs, kernel_shape):
		num_filters = kernel_shape[3]
		biases = tf.get_variable("biases", [num_filters], initializer=tf.constant_initializer(0.0))
		conv = self.conv(inputs, kernel_shape)
		inference = tf.nn.relu(conv + biases)
		if self.cnn_dropout != -1:
			noise_shape = [self.batch_size, self.max_obj_lines, self.max_src_lines, num_filters]
			# for spatial dropout: noise_shape = [self.batch_size, 1, 1, num_filters]
			inference = tf.layers.dropout(inference,
																		rate=self.cnn_dropout,
																		noise_shape=noise_shape,
																		training=self.is_training)
		return inference


	def cnn_decoder(self, h_obj, h_src):
		'''
		h_obj.shape = [batch, obj, lstm_width]
		h_src.shape = [batch, src, lstm_width]
		last_state_of_src_enc.shape = [batch, lstm_width]
		'''

		with tf.variable_scope("cnn"):

			# make a grid out of h_obj, h_src:

			# [batch, obj, hidden_size] -> [batch, obj, 1, hidden_size]
			# [batch, src, hidden_size] -> [batch, 1, src, hidden_size]
			h_obj = tf.expand_dims(h_obj, axis=2)
			h_src = tf.expand_dims(h_src, axis=1)

			# [batch, obj, 1, hidden_size] -> [batch, obj, src, hidden_size]
			# [batch, 1, src, hidden_size] -> [batch, obj, src, hidden_size]
			h_obj = tf.tile(h_obj, multiples=[1, 1, self.max_src_lines, 1])
			h_src = tf.tile(h_src, multiples=[1, self.max_obj_lines, 1, 1])

			# [batch, obj, src, hidden_size] -> [batch, obj, src, 2*hidden_size]
			grid = tf.concat(values=[h_obj, h_src], axis=3)

			# apply convolutions:
			num_channels = 2 * self.hidden_size

			with tf.variable_scope("conv1"):
				conv = self.conv_bias_relu(grid, [5, 5, num_channels, 32])
			with tf.variable_scope("conv2"):
				conv = self.conv_bias_relu(conv, [5, 5, 32, 32])
			with tf.variable_scope("conv3"):
				conv = self.conv_bias_relu(conv, [5, 5, 32, 32])
			with tf.variable_scope("conv4"):
				conv = self.conv_bias_relu(conv, [5, 5, 32, 32])
			with tf.variable_scope("conv5"):
				conv = self.conv(conv, [5, 5, 32, 1])

			self.add_scope_to_grad_norm_summaries()

			return tf.squeeze(conv)


	def inference(self, obj, src, obj_lines, src_lines, obj_len, src_len):
		'''
		obj.shape       = [batch, objLen]
		obj_lines.shape = [batch, objLines]
		'''
		
		# encoders:
		with tf.variable_scope("obj"):
			h_obj, _ = self.encoder(self.obj_vocab_size, obj, obj_len, obj_lines)
		with tf.variable_scope("src"):
			h_src, last_state_of_src_enc = self.encoder(self.src_vocab_size, src, src_len, src_lines)
		
		# alignment:
		if self.decoder == 'rnn':
			align_pred = self.rnn_decoder(h_obj, h_src, last_state_of_src_enc)
		elif self.decoder == 'cnn':
			align_pred = self.cnn_decoder(h_obj, h_src)
		
		return align_pred


	def make_tsp_inputs(self, inputs, inputs_len):
		'''
		inputs.shape     = [batch, max_len, 2]
		inputs_len.shape = [batch]
		'''

		# no change for obj axis:
		input_obj = inputs
		len_input_obj = inputs_len

		# for src axis, pad inputs with NULL:
		null = -0.5*tf.ones([self.batch_size,1,2])
		input_src = tf.concat([inputs, null], 1)
		len_input_src = inputs_len+1

		return input_obj, input_src, len_input_obj, len_input_src


	def inference_tsp(self, obj, src, obj_len, src_len):
		'''
		obj.shape     = [batch, max_len, 2]
		obj_len.shape = [batch]
		'''

		# encoders:
		with tf.variable_scope("obj"):
			h_obj, _ = self.representation(obj, obj_len)
			self.add_scope_to_grad_norm_summaries()
		with tf.variable_scope("src"):
			h_src, last_state_of_src_enc = self.representation(src, src_len)
			self.add_scope_to_grad_norm_summaries()

		# alignment:
		if self.decoder == 'rnn':
			align_pred = self.rnn_decoder(h_obj, h_src, last_state_of_src_enc)
		elif self.decoder == 'cnn':
			align_pred = self.cnn_decoder(h_obj, h_src)

		return align_pred


	def calc_loss(self, target, pred, len_obj, len_src, dense_targets=None):
		'''
		Calculates the loss from the targets and the predictions.
		target.shape = [batch, obj]
		pred.shape   = [batch, obj, src]
		'''

		begin_logits = tf.constant([0,0], dtype=tf.int32)
		begin_labels = tf.constant([0],   dtype=tf.int32)
		
		losses          = []
		pred_softmax    = []
		if dense_targets is None: dense_targets = []
		hard_pred_dense = []
		true_count      = tf.zeros([1], dtype=tf.int32)
		line_count      = tf.zeros([1], dtype=tf.int32)
		correct_prediction = []
		
		for member in xrange(self.batch_size):
			seqLen      = len_obj[member]
			num_classes = len_src[member]

			size_logits = tf.stack([seqLen, num_classes])
			logits = pred[member,:,:]
			size_labels = tf.expand_dims(seqLen, 0)

			# slice pred logits
			logits = tf.slice(logits, begin_logits, size_logits)

			# slice target
			labels = tf.slice(target[member, :], begin_labels, size_labels)

			# dense target
			if self.task in ['tsp5', 'tsp10']:
				dense_target = dense_targets[member,:,:]
			elif self.task in ['art', 'nat']:
				dense_target = self.sparse_to_dense(labels, seqLen)
				dense_targets.append(dense_target)

			# slice dense target
			dense_target_sliced = tf.slice(dense_target, begin_logits, size_logits)

			# soft prediction
			soft_pred = logits

			# paddings
			pad_y = tf.expand_dims(self.max_obj_lines-seqLen, 0)
			pad_y = tf.concat([begin_labels, pad_y], 0)
			pad_x = tf.expand_dims(self.max_src_lines-num_classes, 0)
			pad_x = tf.concat([begin_labels, pad_x], 0)
			paddings = tf.stack([pad_y,pad_x])

			# soft pred as softmax
			softmax = tf.nn.softmax(soft_pred)
			softmax = tf.pad(softmax, paddings)
			pred_softmax.append(softmax)
			
			# hard pred
			hard_pred = tf.to_int32(tf.argmax(soft_pred, 1))
			hard_pred_dense.append(self.sparse_to_dense(hard_pred, seqLen))

			# num correct
			if self.task in ['tsp5', 'tsp10']:
				equal = tf.equal(hard_pred, tf.to_int32(tf.argmax(dense_target_sliced, 1)))
				equal = tf.reduce_all(equal, keep_dims=True)
				line_count += 1
			elif self.task in ['art', 'nat']:
				equal = tf.equal(hard_pred, labels)
				line_count += size_labels
			true_locations = tf.where(equal)
			true_count += array_ops.shape(true_locations)[0]
			correct_prediction.append(equal)

			# loss
			if self.task in ['tsp5', 'tsp10']:
				loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int64(dense_target_sliced))
			elif self.task in ['art', 'nat']:
				loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.to_int64(labels))

			losses.append(loss)
			
		# loss
		losses = tf.concat(losses, 0)
		loss = tf.reduce_mean(losses) # shape = [1]
		
		# softmax, hard pred, dense target
		pred_softmax    = tf.stack(pred_softmax)
		hard_pred_dense = tf.stack(hard_pred_dense)
		if not self.task in ['tsp5', 'tsp10']: dense_targets = tf.stack(dense_targets)
		
		# accuracy
		correct_prediction = tf.concat(correct_prediction, 0)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		return loss, pred_softmax, hard_pred_dense, dense_targets, true_count, line_count, accuracy


	def sparse_to_dense(self, sparse, sparse_len, sparse_values=1):
		'''
		a wrap for tf.sparse_to_dense()
		'''
		lin = tf.range(0, sparse_len, 1)
		sparse_indices = tf.stack([lin, sparse])
		sparse_indices = tf.transpose(sparse_indices, [1,0])
		output_shape = [self.max_obj_lines, self.max_src_lines]
		dense = tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, validate_indices=False)
		
		return dense


	def training(self, loss):

		# optimizer. Adam defaults:
		# tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		
		# global step variable
		global_step = tf.Variable(0, name='global_step', trainable=False)

		# train op
		train_op = tf.contrib.layers.optimize_loss(loss,
																							 global_step=global_step,
																							 learning_rate=None,
																							 optimizer=self.optimizer,
																							 clip_gradients=self.clip_gradients)

		return train_op




	
	
	
	
	
	
	






