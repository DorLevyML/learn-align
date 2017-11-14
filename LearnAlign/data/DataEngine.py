# imports

import os
import numpy as np
import random
import sys
import hashlib

this_file_path = os.path.realpath(__file__)
dataDir = os.path.dirname(this_file_path)
projDir = os.path.dirname(dataDir)
repDir = os.path.dirname(projDir)
sys.path.append(repDir)
from LearnAlign.data import utils

# constants

sep_in_sample  = '\n<<<sep_in_sample>>>\n'
sep_out_sample = '\n<<<sep_out_sample>>>\n'

rw_out_sep = '@#!#@'
rw_in_sep  = '!#@#!'

reserved_words = [
'if', 'else', 'switch', 'case', 'default',
'break',
'int', 'float', 'char', 'double', 'long',
'for', 'while', 'do',
'void',
'goto',
'auto', 'signed', 'const', 'extern', 'register', 'unsigned',
'return',
'continue',
'enum',
'sizeof',
'struct', 'typedef',
'union',
'volatile'
]

# data samples definitions

class AlignmentSample(object):
	
	def __init__(self):
		self.name      = None
		self.offset    = None
		self.obj       = None
		self.src       = None
		self.ali       = None
		self.obj_seq   = None
		self.src_seq   = None
		self.obj_lines = None
		self.src_lines = None
		self.ali_seq   = None
		self.hasAlignmentInfo = False


	# obj sequence properties

	@property
	def obj_len(self):
		return len(self.obj_seq)


	@property
	def obj_num_lines(self):
		return len(self.obj_lines)


	@property
	def obj_line_lens(self):
		a = [-1] + self.obj_lines
		return [a[l+1]-a[l]-1 for l in xrange(len(a)-1)]


	# src sequence properties

	@property
	def src_len(self):
		return len(self.src_seq)


	@property
	def src_num_lines(self):
		return len(self.src_lines)


	@property
	def src_line_lens(self):
		a = [-1] + self.src_lines
		return [a[l+1]-a[l]-1 for l in xrange(len(a)-1)]


	# other properties

	@property
	def has_minus_1_ali(self):
		for line in self.ali:
			if line[1] == -1:
				return True
		return False
		
		
	@property
	def all_minus_1_ali(self):
		for line in self.ali:
			if line[1] != -1:
				return False
		return True
	
	
	@property
	def has_invalid_ali(self):
		for line in self.ali:
			if line[1] < -1 or line[1] == 0:
				return True
		return False
		
		
	@staticmethod
	def samplesToFile(samples_list, file_path):
		txt = sep_out_sample.join([sample.toStr() for sample in samples_list])
		utils.WriteFile(txt, file_path)
		
		
	@staticmethod
	def samplesFromFile(file_path):
		txt = utils.ReadFile(file_path)
		return [AlignmentSample.fromStr(sampleStr) for sampleStr in txt.split(sep_out_sample)]
		
		
	def toStr(self):
		ali = '\n'.join(['%d %d' % (line[0], line[1]) for line in self.ali])
		return sep_in_sample.join([self.obj, self.src, ali])
	
	
	@staticmethod
	def fromStr(sampleStr):
		obj, src, ali = sampleStr.split(sep_in_sample)
		ali = [[int(digit) for digit in line.split(' ')] for line in ali.split('\n')]
		
		sample = AlignmentSample()
		sample.obj = obj
		sample.src = src
		sample.ali = ali
		
		return sample
	
	
	def hash_mod(self, base):
		'''
		'''
		d = hashlib.md5(self.toStr()).hexdigest()
		return int(d, 16) % base
		

class TspSample(object):

	def __init__(self):
		self.sample = None
		self.label  = None


	@property
	def sample_len(self):
		return self.sample.shape[0]

	@property
	def label_len(self):
		return self.label.shape[0]

	@property
	def tour_len(self):
		route_len = 0
		for i in xrange(self.label_len-1):
			src = self.label[i]
			dst = self.label[i + 1]
			x = self.sample[src, :]
			y = self.sample[dst, :]
			route_len += np.linalg.norm(x - y)
		return route_len


	@staticmethod
	def samplesFromFile(file_path):
		lines = utils.ReadFileLines(file_path)
		return [TspSample.fromStr(sampleStr) for sampleStr in lines]


	@staticmethod
	def fromStr(sampleStr):

		x, y = sampleStr.strip().split(' output ')
		x = [float(num) for num in x.split(' ')]
		y = [int(num)-1 for num in y.split(' ')] # PtrNet data labels are 1-based

		# create sample object and arrays
		sample = TspSample()
		sample.sample = np.asarray(x).reshape((-1, 2))
		sample.label  = np.asarray(y)

		return sample


	def rand_perm(self):
		'''
		random permutation over the nodes.
		'''

		# sample a random permutation and its inverse
		perm = np.random.permutation(self.sample_len)
		inv_perm = np.argsort(perm)

		# apply permutation
		self.sample = self.sample[inv_perm,:]
		self.label = perm[self.label]


# datasets definitions

class AlignmentDatasetsCollection(object):
	
	def __init__(self, batch_size):
		
		# data set properties
		self.batch_size = batch_size
		
		# init max lengths
		self.max_obj_len      = 0
		self.max_src_len      = 0
		self.max_obj_lines    = 0
		self.max_src_lines    = 0
		self.max_obj_line_len = 0
		self.max_src_line_len = 0
		
		# vocabularies
		self.vocab_obj = utils.Vocabulary()
		self.vocab_src = utils.Vocabulary()
		
		# data sets dictionary
		self.data_sets = {}
		
	# properties
	
	@property
	def obj_vocab_size(self):
		return self.vocab_obj.vocab_size
	
	
	@property
	def src_vocab_size(self):
		return self.vocab_src.vocab_size
	
	
	# utils

	def print_statistics(self):

		def print_stats(l):
			l = np.array(l)
			mean = np.mean(l)
			std = np.std(l)
			print('%.1f +/- %.1f' % (mean, std))
			print('----------------')

		data = []
		for name, _ in self.data_sets.iteritems():
			data += self.Get(name).data

		src_num_lines = [sample.src_num_lines for sample in data]
		src_line_lens = utils.flatten_list([sample.src_line_lens for sample in data])

		obj_num_lines = [sample.obj_num_lines for sample in data]
		obj_line_lens = utils.flatten_list([sample.obj_line_lens for sample in data])

		print('src_num_lines:')
		print_stats(src_num_lines)
		print('src_line_lens:')
		print_stats(src_line_lens)
		print('obj_num_lines:')
		print_stats(obj_num_lines)
		print('obj_line_lens:')
		print_stats(obj_line_lens)


	# data preparation
	
	def PrepareObjectCode(self, obj):
		
		lines = obj.split('\n')
		
		try:
			seq = []
			
			for line in lines:
				
				words = line.split(' ')
				opcode = words[0]
				seq.append(self.vocab_obj.get_symbol_idx(opcode))
				
				# if has operands:
				if len(words) >= 2:
					tmp = words[1].replace(':','|:|').replace('(','|(|').replace(')','|)|').replace(',','|')
					operands = [member for member in tmp.split('|') if member]
					
					for operand in operands:
						
						# if is register:
						if operand[0] == '%':
							seq.append(self.vocab_obj.get_symbol_idx(operand))
								
						# if is number:
						elif operand[0:3] == '$0x':
							num = utils.hex_str_to_dec_str(operand[3:])
							for c in num:
								seq.append(self.vocab_obj.get_symbol_idx(c))
							
						# if is something else, append char-wise:
						else:
							for c in operand:
								seq.append(self.vocab_obj.get_symbol_idx(c))
					
				# new line:
				seq.append(self.vocab_obj.get_symbol_idx('\n'))

		except:
			print 'got exception for:'
			print('\n'.join(lines))
			raise
		
		return seq
		
		
	def PrepareSourceCode(self, src):
		
		lines = src.split('\n')
		
		# trim lines
		txt = '\n'.join([line.strip() for line in lines])
		txt = txt+'\n'
		
		# saved words to atoms
		seq = []
		for word in reserved_words:
			txt = txt.replace(word, rw_out_sep+rw_in_sep+word+rw_out_sep)
		txt = txt.split(rw_out_sep)
		for mini_txt in txt:
			if mini_txt[:len(rw_in_sep)] == rw_in_sep:
				atom = mini_txt[len(rw_in_sep):]
				seq.append(atom)
			else:
				[seq.append(atom) for atom in mini_txt]
				
		# remove white space character
		seq = [atom for atom in seq if atom != ' ']
		
		# append to sequence
		seq = [self.vocab_src.get_symbol_idx(atom) for atom in seq]
		
		return seq
		
		
	def PrepareAlignment(self, ali):
		aliList = [srcLine-1 for _,srcLine in ali] # python indices begin at 0, line numbers begin at 1
		return aliList
		
		
	def PrepareForNN(self, sample, restrict_len=False, max_seq_len=-1):
		'''
		prepare data for neural net
		'''
		
		objSeq    = self.PrepareObjectCode(sample.obj)
		srcSeq    = self.PrepareSourceCode(sample.src)
		alignList = self.PrepareAlignment(sample.ali)
		
		# find places of new lines in the obj/src sequences
		self.vocab_obj.debugMode = False
		self.vocab_src.debugMode = False
		objLinesMarker = [i for i in xrange(len(objSeq)) if objSeq[i]==self.vocab_obj.get_symbol_idx('\n')]
		srcLinesMarker = [i for i in xrange(len(srcSeq)) if srcSeq[i]==self.vocab_src.get_symbol_idx('\n')]

		# update max lengths:
		if max_seq_len >= 0:
			if (len(objSeq)         > max_seq_len) or \
				 (len(srcSeq)         > max_seq_len) or \
				 (len(objLinesMarker) > max_seq_len) or \
				 (len(srcLinesMarker) > max_seq_len):
				return None

		if restrict_len:
			if (len(objSeq)         > self.max_obj_len)   or \
				 (len(srcSeq)         > self.max_src_len)   or \
				 (len(objLinesMarker) > self.max_obj_lines) or \
				 (len(srcLinesMarker) > self.max_src_lines):
				return None
		else:
			if (len(objSeq)         > self.max_obj_len):   self.max_obj_len   = len(objSeq)
			if (len(srcSeq)         > self.max_src_len):   self.max_src_len   = len(srcSeq)
			if (len(objLinesMarker) > self.max_obj_lines): self.max_obj_lines = len(objLinesMarker)
			if (len(srcLinesMarker) > self.max_src_lines): self.max_src_lines = len(srcLinesMarker)

		sample.obj_seq = objSeq
		sample.src_seq = srcSeq
		sample.obj_lines = objLinesMarker
		sample.src_lines = srcLinesMarker
		sample.ali_seq = alignList
		return sample
		
		
	# API
	
	def LoadFromFile(self, name, file_list, restrict_len=False, max_seq_len=-1, shuffle=True):
		'''
		'''
		
		# load from file
		samples = [AlignmentSample.samplesFromFile(file_path) for file_path in file_list]
		samples = utils.flatten_list(samples)
		
		# prepare for nn and clean samples that were discarded
		data = [self.PrepareForNN(sample, restrict_len=restrict_len, max_seq_len=max_seq_len) for sample in samples]
		data = [sample for sample in data if sample]
		
		# create data set
		self.data_sets[name] = AlignmentDataset(name, self.batch_size, data, shuffle=shuffle)
		
		
	def Init(self):
		'''
		after loading data sets (train, valid, test), run
		this function to set the maximal lengths of all AlignmentDataset objects
		'''
		for name, _ in self.data_sets.iteritems():
			max_lens = self.max_obj_len, self.max_src_len, self.max_obj_lines, self.max_src_lines, self.max_obj_line_len, self.max_src_line_len
			self.Get(name).set_max_lens(max_lens)
	
	
	def Get(self, name):
		'''
		return AlignmentDataset object by name
		'''
		return self.data_sets[name]
	

class AlignmentDataset(object):
	
	def __init__(self, name, batch_size, data, shuffle=True):
		self.name = name
		self.batch_size = batch_size
		self.data = data
		if shuffle:
			self.shuffle()
		
		self.curr_sample = 0
	
	
	@property
	def num_samples(self):
		return len(self.data)
	
	
	def set_max_lens(self, max_lens):
		self.max_obj_len, self.max_src_len, self.max_obj_lines, self.max_src_lines, self.max_obj_line_len, self.max_src_line_len = max_lens
	
	
	def shuffle(self):
		random.shuffle(self.data)
	
	
	def next_sample(self):
		sample = self.data[self.curr_sample]
		self.curr_sample += 1
		if(self.curr_sample == len(self.data)):
			self.curr_sample = 0
			self.shuffle()
		
		return sample
	
	
	def next_batch(self):
		
		# init batches
		obj_batch      = np.zeros([self.batch_size, self.max_obj_len])
		src_batch      = np.zeros([self.batch_size, self.max_src_len])
		objLines_batch = np.zeros([self.batch_size, self.max_obj_lines])
		srcLines_batch = np.zeros([self.batch_size, self.max_src_lines])
		ali_batch      = np.zeros([self.batch_size, self.max_obj_lines])
		
		# init batches lengths
		len_obj_batch      = np.zeros([self.batch_size])
		len_src_batch      = np.zeros([self.batch_size])
		len_objLines_batch = np.zeros([self.batch_size])
		len_srcLines_batch = np.zeros([self.batch_size])
		
		# init batches texts
		obj_text_batch = []
		src_text_batch = []
		
		# fill a batch:
		for i in xrange(self.batch_size):
			
			# extract one sample:
			sample = self.next_sample()
			objText  = sample.obj
			srcText  = sample.src
			obj      = sample.obj_seq
			src      = sample.src_seq
			objLines = sample.obj_lines
			srcLines = sample.src_lines
			ali      = sample.ali_seq
			
			# add the sample to batch:
			obj_batch[i,0:len(obj)]           = obj
			src_batch[i,0:len(src)]           = src
			objLines_batch[i,0:len(objLines)] = objLines
			srcLines_batch[i,0:len(srcLines)] = srcLines
			ali_batch[i,0:len(ali)]           = ali
			
			# add to length vectors:
			len_obj_batch[i]      = len(obj)
			len_src_batch[i]      = len(src)
			len_objLines_batch[i] = len(objLines)
			len_srcLines_batch[i] = len(srcLines)
			
			# add to text batch:
			obj_text_batch.append(objText)
			src_text_batch.append(srcText)
			
		batches     = (obj_batch, src_batch, objLines_batch, srcLines_batch, ali_batch)
		len_batches = (len_obj_batch, len_src_batch, len_objLines_batch, len_srcLines_batch)
		textBatches = (obj_text_batch, src_text_batch)
		
		return (batches, len_batches, textBatches)
		
		
class TspDatasetsCollection(object):

	def __init__(self, batch_size):

		# data set properties
		self.batch_size = batch_size

		# init max length
		self.max_sample_len = 0
		self.max_label_len  = 0

		# data sets dictionary
		self.data_sets = {}

	# data preparation

	def PrepareForNN(self, sample, restrict_len=False, max_seq_len=-1):
		'''
		prepare data for neural net
		'''

		# update max lengths:
		if max_seq_len >= 0:
			if (sample.sample_len > max_seq_len) or \
			   (sample.label_len  > max_seq_len):
				return None

		if restrict_len:
			if (sample.sample_len > self.max_sample_len) or \
				 (sample.label_len  > self.max_label_len):
				return None
		else:
			if sample.sample_len > self.max_sample_len: self.max_sample_len = sample.sample_len
			if sample.label_len  > self.max_label_len:  self.max_label_len  = sample.label_len

		return sample

	# API

	def LoadFromFile(self, name, file_list, restrict_len=False, max_seq_len=-1, shuffle=True, perm=False):

		# load from file
		samples = [TspSample.samplesFromFile(file_path) for file_path in file_list]
		samples = utils.flatten_list(samples)

		# prepare for nn and clean samples that were discarded
		data = [self.PrepareForNN(sample, restrict_len=restrict_len, max_seq_len=max_seq_len) for sample in samples]
		data = [sample for sample in data if sample]

		# create data set
		self.data_sets[name] = TspDataset(name, self.batch_size, data, shuffle=shuffle, perm=perm)


	def Init(self):
		'''
		after loading data sets (train, valid, test), run
		this function to set the maximal lengths of all TspDataset objects
		'''
		max_lens = self.max_sample_len, self.max_label_len
		for name, _ in self.data_sets.iteritems():
			self.Get(name).set_max_lens(max_lens)


	def Get(self, name):
		'''
		this function returns TspDataset object by name
		'''
		return self.data_sets[name]


class TspDataset(object):

	def __init__(self, name, batch_size, data, shuffle=True, perm=False):
		self.name = name
		self.batch_size = batch_size
		self.data = data
		if shuffle:
			self.shuffle()
		if perm:
			self.perm()

		self.curr_sample = 0

	@property
	def num_samples(self):
		return len(self.data)


	def set_max_lens(self, max_lens):
		self.max_sample_len, self.max_label_len = max_lens


	def shuffle(self):
		random.shuffle(self.data)


	def perm(self):
		for sample in self.data:
			sample.rand_perm()


	def next_sample(self):
		sample = self.data[self.curr_sample]
		self.curr_sample += 1
		if self.curr_sample == len(self.data):
			self.curr_sample = 0
			self.shuffle()
			self.perm()

		return sample


	def next_batch(self):

		# init batches
		sample_batch        = np.zeros([self.batch_size, self.max_sample_len, 2])
		sparse_label_batch  = np.zeros([self.batch_size, self.max_label_len])
		dense_label_batch   = np.zeros([self.batch_size, self.max_sample_len, self.max_sample_len+1])

		# init batches lengths
		len_sample_batch = np.zeros([self.batch_size])
		len_label_batch  = np.zeros([self.batch_size])

		# fill a batch:
		for i in xrange(self.batch_size):
			# extract one sample:
			sample = self.next_sample()
			# add the sample to batch:
			sample_batch[i, 0:sample.sample_len, :] = sample.sample
			sparse_label_batch[i, 0:sample.label_len] = sample.label
			# make dense target:
			dense_label_batch[i, :, sample.sample_len] = np.ones([self.max_sample_len])
			for j in xrange(sample.label_len-1):
				# calc indices
				row = sample.label[j]
				col = sample.label[j+1]
				# assign one-hot
				dense_label_batch[i, row, col] = 1
				# delete NULL class one-hot
				dense_label_batch[i, row,  sample.sample_len] = 0

			# add to length vectors:
			len_sample_batch[i] = sample.sample_len
			len_label_batch[i]  = sample.label_len

		batches = (sample_batch, sparse_label_batch, dense_label_batch)
		len_batches = (len_sample_batch, len_label_batch)

		return batches, len_batches, None

