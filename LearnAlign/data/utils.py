# imports

import datetime
import math
import itertools

# code

class Vocabulary(object):
	
	def __init__(self, debugMode=False):
		
		# two dictionaries for transforming between symbols and their indices:
		self.symbol2idx = {}
		self.idx2symbol = {}
		
		# the size of the class's vocabulary, i.e., the total number of symbols
		# that's been seen:
		self.vocab_size = 0
		
		# debug mode
		self.debugMode = debugMode
	
	
	def get_symbol_idx(self, symbol):

		if self.debugMode:
			print(symbol)
		
		# if this is a new symbol for the class, update vocabulary
		# size and the two dictionaries:
		if not self.symbol2idx.has_key(symbol):
			self.vocab_size = self.vocab_size + 1
			self.symbol2idx[symbol] = self.vocab_size
			self.idx2symbol[self.vocab_size] = symbol
		
		# return the index of the given symbol:
		return self.symbol2idx[symbol]
		
		
def ReadFileLines(file_path):
	f = open(file_path, 'r')
	lines = f.readlines()
	f.close()
	return lines
	
	
def ReadFile(file_path):
	f = open(file_path, 'r')
	txt = f.read()
	f.close()
	return txt
	
	
def WriteFile(txt, file_path):
	f = open(file_path, 'w')
	f.write(txt)
	f.close()
	
	
def NowStr():
	return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def hex_str_to_dec_str(hex_str):
	'''
	in: hex string without '0x'
	out: decimal string of the same value
	'''
	num = int(hex_str, 16)
	if hex_str[0]== 'f' and len(hex_str) in [8, 16]:
		num -= long(math.pow(16, len(hex_str)))

	dec_str = str(int(num))
	return dec_str


def flatten_list(lists):
	return list(itertools.chain.from_iterable(lists))



