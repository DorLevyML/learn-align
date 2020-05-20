'''
end-to-end data set generation, for both artificial and natural data.
'''

# imports

import os
import datetime
import math
import sys
import regex as re
import subprocess
import shutil
import tarfile

this_file_path = os.path.realpath(__file__)
dataGenDir = os.path.dirname(this_file_path)
projDir = os.path.dirname(dataGenDir)
repDir = os.path.dirname(projDir)
sys.path.append(repDir)
from LearnAlign.data import utils
from LearnAlign.data.DataEngine import AlignmentSample, AlignmentDatasetsCollection

# conf

class Conf(object):
	def __init__(self):

		self.gen_opt_lvls = [
			'O1',
			'O2',
			'O3'
		]

		self.data_to_generate = [
			'art',
			'nat'
		]

		self.num_samples_art = 50000
		self.global_max_seq_len = 450
		self.part_train = 80
		self.part_valid = 10
		self.part_test  = 10
		self.divide_samples_by_hash = True

		self.pyfuzz_script_path = os.path.join(dataGenDir, 'pyfuzz/pyfuzz_wrap.py')
		self.art_gen_source_file_path = os.path.join(dataGenDir, 'workDir/source.c')
		self.art_gen_output_file_path = os.path.join(dataGenDir, 'workDir/source.o')

		self.natural_projects_dir = os.path.join(dataGenDir, 'natural_projects')
		self.target_dir_ds_art    = os.path.join(dataGenDir, 'generated_datasets_art')
		self.target_dir_ds_nat    = os.path.join(dataGenDir, 'generated_datasets_nat')

conf = Conf()

# utils

def exec_sub_proc(exec_line_list):
	'''
	execute in subprocess.
	return result.
	in case of an error, raise an exception.
	'''
	res = subprocess.check_output(
		exec_line_list,
		stdin=None,
		stderr=None,
		shell=False,
		universal_newlines=False
	)
	return res

class cd:
	'''
	Context manager for changing the current working directory.
	'''

	def __init__(self, new_path):
		self.new_path = os.path.expanduser(new_path)

	def __enter__(self):
		self.saved_path = os.getcwd()
		os.chdir(self.new_path)

	def __exit__(self, etype, value, traceback):
		os.chdir(self.saved_path)

def tik():
	return datetime.datetime.now()

def tok(t):
	return datetime.datetime.now() - t

# data generation

class CompiledCodeFunction(AlignmentSample):

	func_rec = re.compile(r'(\w+)\(\):')
	source_code_line_rec = re.compile(r'^(.+):(\d+)(?: \(discriminator \d+\))?$')
	object_code_line_rec = re.compile(r'^\s*[\da-fA-F]+:\s*(.+)$')

	def __init__(self, name, c_file_path):
		super(CompiledCodeFunction, self).__init__()
		self.name = name
		self.c_file_path = c_file_path
		self.objdump_lines = []
		self.ali = []

	def extract_obj_code_and_alignment(self):

		object_code_lines = []
		source_code_line_num = -1
		object_code_line_num = 0
		own_code = False

		for line in self.objdump_lines:

			# function name
			m = self.func_rec.search(line)
			if m:
				own_code = (m.group(1) == self.name)
				continue

			# source file + line
			m = self.source_code_line_rec.search(line)
			if m:
				self.hasAlignmentInfo = True
				own_file = (m.group(1) == self.c_file_path)
				if own_code and own_file:
					source_code_line_num = int(m.group(2))
				else:
					source_code_line_num = -1
				continue

			# object code line
			m = self.object_code_line_rec.search(line)
			if m:
				object_code_line_num += 1
				object_code_line = m.group(1)
				object_code_lines.append(object_code_line)
				self.ali.append((object_code_line_num, source_code_line_num))
				continue

		self.obj = '\n'.join(object_code_lines)

	def add_objdump_line(self, line):
		self.objdump_lines.append(line)

	def in_range(self, start_line, end_line):
		for _, src_line in self.ali:
			if src_line < start_line or src_line > end_line:
				return False
		return True

	def normalize_alignment(self):
		'''
		correct alignment by offset
		'''
		self.ali = [(a, b - self.offset) if b != -1 else (a, b) for (a, b) in self.ali]

class CompiledCodeFile(object):
	'''
	bug:
	- happens for balanced quotations of unbalanced parentheses outside of a function definition
	- e.g. static char c = "{"
	- in some cases it makes the regex engine get stuck
	- currently it's left open and sources for which it happens should be edited manually before
		being processed
	'''

	# source code regex
	func_sig_re = r'(\w+)\s*\([^()]*\)\s*'
	func_arg_re = r'(\s*(\w+\s+)?\w+(\s+|\s+\*|\*\s+|\s+\*\s+)\w+\s*;)*\s*'
	func_sig_rec = re.compile(func_sig_re + func_arg_re + r'$')
	parentheses_rec = re.compile(r'\{(?:\'\{\'|\'\}\'|".*?"|/\*(.|\n)*?\*/|//.*?\n|//.*?$|[^{}]|(?R))*\}')

	# object code regex
	func_header_rec = re.compile(r'^\w+ <(\w+)>:$')

	# i file regex
	empty_line_rec = re.compile(r'^\s*$')
	comment_rec = re.compile(r'^\s*#')

	def __init__(self, o_file_path, print_log=True):

		self.print_log = print_log

		self.o_file_path = o_file_path
		self.c_file_path = self.change_ext_to('c')
		self.i_file_path = self.change_ext_to('i')
		self.s_file_path = self.change_ext_to('s')

		self.candidates = []
		self.samples = []

	# properties

	@property
	def has_c_file(self):
		return os.path.isfile(self.c_file_path)

	@property
	def has_i_file(self):
		return os.path.isfile(self.i_file_path)

	# utils

	def change_ext_to(self, x):
		return os.path.splitext(self.o_file_path)[0] + '.%s' % x

	def _print(self, s):
		if self.print_log:
			print(s)

	# data processing

	def process_objdump_output(self, objdump_output):

		objdump_lines = objdump_output.split('\n')
		curr_func = None

		for line in objdump_lines:
			if curr_func is None:
				m = self.func_header_rec.search(line)
				if m:
					curr_func = CompiledCodeFunction(
						name=m.group(1),
						c_file_path=self.c_file_path)
			else:
				if line == '':
					curr_func.extract_obj_code_and_alignment()
					self.candidates.append(curr_func)
					curr_func = None
				else:
					curr_func.add_objdump_line(line)

	def process_source(self):

		self._print('searching functions in source code...')

		# read source file
		txt = utils.ReadFile(self.c_file_path)

		# extract all function definitions and then search for the wanted function names
		for m in self.parentheses_rec.finditer(txt):
			n = self.func_sig_rec.search(txt, 0, m.start())
			if n is None:
				continue
			func_def = n.group(0) + m.group(0)
			func_name = n.group(1)

			for ind, sample in enumerate(self.candidates):

				if sample.name == func_name and sample.hasAlignmentInfo:
					line_num_start = txt[:m.start()].count('\n') + 1
					line_num_end = txt[:m.end()].count('\n') + 1

					if sample.in_range(line_num_start, line_num_end):
						self._print('found function: %s' % func_name)
						sample.src = func_def
						sample.offset = txt[:n.start()].count('\n')
						self.samples.append(sample)
						self.candidates.pop(ind)
						break

	def correct_samples(self):
		'''
		1) remove samples that:
		- have no source code
		- have no alignment info
		- have alignment that is all -1 (those samples may originate
			in code that hasn't been preprocessed for some reason)
		2) correct alignment by offset
		'''

		# mark samples to be removed
		for ind, sample in enumerate(self.samples):
			if sample.src is None or \
			   not sample.hasAlignmentInfo or \
			   sample.all_minus_1_ali:
				self.samples[ind] = None

		# remove samples
		self.samples = [sample for sample in self.samples if sample]

		# normalize alignment
		for sample in self.samples:
			sample.normalize_alignment()

	# API

	def extract_data_from_file(self):

		exec_line = 'objdump -d -l --no-show-raw-insn %s' % self.o_file_path
		res = exec_sub_proc(exec_line.split())

		self.process_objdump_output(res)
		self.process_source()
		self.correct_samples()

	def clean_source_and_temps(self):
		'''
		1) cleans i file source code:
		- remove empty lines
		- remove lines that begin with '#'
		2) writes the clean source code to c file
		3) deletes temps (i & s files). this is mandatory for the second compilation
		'''

		# read i
		i_txt = utils.ReadFile(self.i_file_path)

		# clean i source
		i_txt = i_txt.split('\n')
		i_txt = [line for line in i_txt if not self.empty_line_rec.search(line)]
		i_txt = [line for line in i_txt if not self.comment_rec.search(line)]
		i_txt = '\n'.join(i_txt)

		# write to c
		utils.WriteFile(i_txt, self.c_file_path)

		# delete temps
		os.remove(self.i_file_path)
		os.remove(self.s_file_path)

	@staticmethod
	def is_o_file(file_name):
		return os.path.splitext(file_name)[1] == '.o'

class GnuProject(object):

	pseudo_comments_rec = re.compile(r'\/\*|\/\/')

	def __init__(self, root_dir, sub_dir, opt_lvl):
		self.root_dir = root_dir
		self.name = sub_dir
		self.opt_lvl = opt_lvl
		self.path = os.path.join(root_dir, sub_dir)
		self.preprocessed_files = []
		self.samples = []

	# properties

	@property
	def contains_c_files(self):
		for _, _, file_names in os.walk(self.path):
			for file_name in file_names:
				if os.path.splitext(file_name)[1] == '.c':
					return True

		return False

	@property
	def num_samples(self):
		return len(self.samples)

	# utils

	def build_gnu_project(self, cflags, clean):
		# cd to project's directory
		with cd(self.path):
			if clean:
				res = exec_sub_proc(['make', 'clean'])
				res = exec_sub_proc(['make', 'distclean'])
			res = exec_sub_proc(['./configure', 'CC=gcc', 'CFLAGS=%s' % cflags])
			res = exec_sub_proc(['make'])

	def clean_project_sources(self):

		for root_dir, _, file_names in os.walk(self.path):
			for file_name in file_names:

				# only .o files...
				if not CompiledCodeFile.is_o_file(file_name):
					continue

				# ...who have .c and .i files
				o_file_path = os.path.join(root_dir, file_name)
				compiled_code_file = CompiledCodeFile(o_file_path)
				if not compiled_code_file.has_c_file or \
					not compiled_code_file.has_i_file:
					continue

				try:
					compiled_code_file.clean_source_and_temps()
				except Exception as e:
					print('caught exception:')
					print(e)
					print('skipping file:')
					print(compiled_code_file.c_file_path)
					continue

				# append file to list
				self.preprocessed_files.append(compiled_code_file.c_file_path)

	def preprocess_project(self):

		print('executing 1st build...')
		try:
			self.build_gnu_project('-O0 -save-temps', clean=False)  # no optimization (for speed)
		except:
			print('1st build failed. continuing.')

		print('cleaning sources...')
		self.clean_project_sources()

		print('executing 2nd build with clean sources...')
		try:
			self.build_gnu_project('-g -%s' % self.opt_lvl, clean=True)  # for alignment
		except:
			print('2nd build failed. continuing.')

	def extract_data_from_project(self):

		# for every file in directory tree
		for root_dir, _, file_names in os.walk(self.path):
			for file_name in file_names:

				if not CompiledCodeFile.is_o_file(file_name):
					continue

				o_file_path = os.path.join(root_dir, file_name)
				compiled_code_file = CompiledCodeFile(o_file_path)
				if not compiled_code_file.has_c_file:
					continue

				print('file %s' % compiled_code_file.c_file_path)
				if compiled_code_file.c_file_path not in self.preprocessed_files:
					print('has not been preprocessed. skipping...')
					continue

				try:
					compiled_code_file.extract_data_from_file()
					self.samples += compiled_code_file.samples
				except KeyboardInterrupt as e:
					# Notes:
					# - KeyboardInterrupt inherits from BaseException, not Exception.
					# - due to this except block, ctrl+c won't stop the program. to stop
					#   it use ctrl+z instead.
					print('caught KeyboardInterrupt. skipping...')
					continue
				except Exception as e:
					print('caught exception:')
					print(e)
					print('skipping...')
					continue

	def check_data(self):
		'''
		checks if data is not OK:
		1) invalid alignment (not a natural number nor -1)
		2) sources still contain comments
		one problematic sample is enough to flag a project as problematic.
		There's no need to examine all samples, because usually a problem in
		one sample originates in a bigger problem that affects other samples as well.
		so it's best to first fix the problem and only then examine the project samples again.
		'''

		print('checking data...')

		for sample in self.samples:
			if sample.has_invalid_ali:
				print('found a sample with invalid alignment: %s' % sample.name)
				break

		for sample in self.samples:
			if self.pseudo_comments_rec.search(sample.src):
				print('found a sample that might contain comments: %s' % sample.name)
				break

	def clean_data(self):

		print('cleaning data...')

		initial_count = self.num_samples

		# mark samples to be removed
		for ind, sample in enumerate(self.samples):
			if sample.has_invalid_ali or \
			   sample.has_minus_1_ali:
				self.samples[ind] = None

		# remove marked samples
		self.samples = [sample for sample in self.samples if sample]
		print('removed %d out of %d samples' % (initial_count - self.num_samples, initial_count))

	# API

	def process_project(self):
		self.preprocess_project()
		self.extract_data_from_project()
		self.check_data()
		self.clean_data()

	def write_dataset_file(self, data_sets_dir):
		data_file_format = os.path.join(data_sets_dir, 'ds_%s_%s.txt')
		AlignmentSample.samplesToFile(self.samples, data_file_format % (self.opt_lvl, self.name))

class NaturalDataGeneration(object):

	def __init__(self):
		self.c_projects = None
		self.none_c = None
		self.c_projects_with_data = None
		self.c_projects_no_data = None

	@property
	def num_samples(self):
		return sum([proj.num_samples for proj in self.c_projects_with_data])

	def prepare_all_projects(self):
		'''
		- resets processed projects lists
		- deletes project directories (present from previous builds).
		- unpacks project archives for the next build.
		'''

		# reset lists
		self.c_projects = []
		self.none_c = []
		self.c_projects_with_data = []
		self.c_projects_no_data = []

		root_dir, sub_dirs, file_names = os.walk(conf.natural_projects_dir).next()

		# remove all sub dirs
		for sub_dir in sub_dirs:
			sub_dir_path = os.path.join(root_dir, sub_dir)
			shutil.rmtree(sub_dir_path)

		# unzip all archive files
		for file_name in file_names:
			file_path = os.path.join(root_dir, file_name)
			if not tarfile.is_tarfile(file_path):
				continue
			tf = tarfile.open(file_path)
			tf.extractall(conf.natural_projects_dir)
			tf.close()

	def extract_data_from_c_projects(self, opt_lvl):

		# separate projects to c and none-c
		_, sub_dirs, _ = os.walk(conf.natural_projects_dir).next()
		for sub_dir in sub_dirs:
			proj = GnuProject(conf.natural_projects_dir, sub_dir, opt_lvl)
			if proj.contains_c_files:
				self.c_projects.append(proj)
			else:
				self.none_c.append(proj)

		# preprocess & extract data from all c projects
		for proj in self.c_projects:

			print('-----------------------')
			print('project: %s' % proj.name)
			proj.process_project()

			print('%s: %d samples' % (proj.name, proj.num_samples))
			if proj.num_samples == 0:
				self.c_projects_no_data.append(proj)
			else:
				self.c_projects_with_data.append(proj)

		# show summary
		print('none c projects:')
		for proj in self.none_c:
			print('\t%s' % proj.name)
		print('c projects with no data:')
		for proj in self.c_projects_no_data:
			print('\t%s' % proj.name)
		print('c projects with data:')
		for proj in self.c_projects_with_data:
			print('\t%s (%d samples)' % (proj.name, proj.num_samples))
		print('total: %d samples' % self.num_samples)

	def write_dataset_files(self, data_sets_dir, opt_lvl):

		# for each individual project
		for proj in self.c_projects_with_data:
			proj.write_dataset_file(data_sets_dir)

		# for all projects combined
		total_samples = [proj.samples for proj in self.c_projects_with_data]
		total_samples = utils.flatten_list(total_samples)
		ds_file = os.path.join(data_sets_dir, 'ds_%s.txt' % opt_lvl)
		AlignmentSample.samplesToFile(total_samples, ds_file)
		print('wrote %d samples to file:' % len(total_samples))
		print(ds_file)

	# API

	def run(self, opt_lvl, target_dir, ds_specs):
		self.prepare_all_projects()
		self.extract_data_from_c_projects(opt_lvl)
		self.write_dataset_files(target_dir, opt_lvl)

class ArtificialDataGeneration(object):

	@staticmethod
	def generate_sample(opt_lvl, rand_code):
		'''
		args:
		opt_lvl - compiler optimization level
		rand_code - whether to generate random code or use an existing source file.
		'''

		if rand_code:
			# generate random source code
			exec_line = 'python %s %s' % (conf.pyfuzz_script_path, conf.art_gen_source_file_path)
			res = exec_sub_proc(exec_line.split())

		# run compiler with optimizations and output debug info
		exec_line = 'gcc -c -g -%s %s -o %s' % (opt_lvl, conf.art_gen_source_file_path, conf.art_gen_output_file_path)
		res = exec_sub_proc(exec_line.split())

		# extract code & alignment
		compiled_code_file = CompiledCodeFile(conf.art_gen_output_file_path, print_log=False)
		compiled_code_file.extract_data_from_file()
		sample = compiled_code_file.samples[0]  # a list of one sample

		return sample

	# API

	def compile_one_with_multiple_opt_levels(self):
		'''
		generates a data sample for one source code
		function (should be given in workDir/source.c).
		'''
		opt_lvls = ['O0', 'O1', 'O2', 'O3']
		samples = [self.generate_sample(opt_lvl, rand_code=False) for opt_lvl in opt_lvls]
		output_file_path = os.path.join(conf.target_dir_ds_art, 'safe.txt')
		AlignmentSample.samplesToFile(samples, output_file_path)

	def run(self, opt_lvl, target_dir, ds_specs):
		samples = [self.generate_sample(opt_lvl, rand_code=True) for _ in xrange(ds_specs.num_samples)]
		target_file = os.path.join(target_dir, 'ds_%s.txt' % opt_lvl)
		AlignmentSample.samplesToFile(samples, target_file)

# dataset generation

class DatasetFileList(object):

	ds_file_re_format = r'^ds_[\w-]+_%s_\d+\.txt$'

	def __init__(self, name):
		self.name = name
		self.file_list = []
		self.ds_file_rec = re.compile(self.ds_file_re_format % self.name)

	def file_belongs_to_list(self, filename):
		m = self.ds_file_rec.search(filename)
		return m is not None

	def add_file(self, file_name):
		self.file_list.append(file_name)

	def sort(self):
		self.file_list.sort()

class DatasetSpecs(object):
	def __init__(self, name, data_dir, max_seq_len, num_samples):
		self.name = name
		self.data_dir = data_dir
		self.max_seq_len = max_seq_len
		self.num_samples = num_samples

class DatasetGenerator(object):
	'''
	generates data set files.
	'''

	def __init__(self, data_gen, dataset_specs):
		self.data_gen = data_gen
		self.dataset_specs = dataset_specs

	def create_dataset(self):

		for opt_lvl in conf.gen_opt_lvls:

			target_dir = os.path.join(self.dataset_specs.data_dir, '%s' % opt_lvl)
			if not os.path.exists(target_dir):
				os.mkdir(target_dir)

			t = tik()
			self.data_gen.run(opt_lvl, target_dir, self.dataset_specs)
			print('dataset created (%s)' % tok(t))

	@staticmethod
	def take_data_part(data_all, range_start, range_end):
		base = conf.part_train + conf.part_valid + conf.part_test
		if conf.divide_samples_by_hash:
			range_ = range(range_start, range_end)
			return [sample for sample in data_all if sample.hash_mod(base) in range_]
		else:
			start = int(math.floor((float(range_start) / base) * len(data_all)))
			end =   int(math.floor((float(range_end)   / base) * len(data_all)))
			return data_all[start:end]

	def limit_length_and_divide_datasets(self):

		print('limiting sample lengths and dividing datasets...')

		ds_file_format     = os.path.join(self.dataset_specs.data_dir, '%s/ds_%s.txt')
		target_file_format = os.path.join(self.dataset_specs.data_dir, '%s/ds_%s_len%d_%s_%d.txt')

		# load data set and limit sample length
		t = tik()
		data_sets = AlignmentDatasetsCollection(batch_size=1)
		for opt_lvl in conf.gen_opt_lvls:
			data_sets.LoadFromFile(
				opt_lvl,
				[ds_file_format % (opt_lvl, opt_lvl)],
				max_seq_len=self.dataset_specs.max_seq_len
			)
		data_sets.Init()
		print('data sets loaded (%s)' % tok(t))

		for opt_lvl in conf.gen_opt_lvls:

			# take all data
			data_all = data_sets.Get(opt_lvl).data

			# divide
			ind1 = conf.part_train
			ind2 = conf.part_train + conf.part_valid
			base = conf.part_train + conf.part_valid + conf.part_test
			data_train = self.take_data_part(data_all, 0, ind1)
			data_valid = self.take_data_part(data_all, ind1, ind2)
			data_test  = self.take_data_part(data_all, ind2, base)

			# show num samples
			print('num samples for opt level %s:' % opt_lvl)
			print('total: %d' % len(data_all))
			print('train: %d' % len(data_train))
			print('valid: %d' % len(data_valid))
			print('test:  %d' % len(data_test))

			# save to files
			AlignmentSample.samplesToFile(
				data_train,
				target_file_format % (opt_lvl, opt_lvl, self.dataset_specs.max_seq_len, 'train', len(data_train))
			)
			AlignmentSample.samplesToFile(
				data_valid,
				target_file_format % (opt_lvl, opt_lvl, self.dataset_specs.max_seq_len, 'valid', len(data_valid))
			)
			AlignmentSample.samplesToFile(
				data_test,
				target_file_format % (opt_lvl, opt_lvl, self.dataset_specs.max_seq_len, 'test',  len(data_test))
			)

	def find_my_files(self):
		file_lists = [
			DatasetFileList('train'),
			DatasetFileList('valid'),
			DatasetFileList('test')
		]
		for root_dir, _, file_names in os.walk(self.dataset_specs.data_dir):
			# add each data file to relevant file list
			for file_name in file_names:
				for file_list in file_lists:
					if file_list.file_belongs_to_list(file_name):
						file_path = os.path.join(root_dir, file_name)
						file_list.add_file(file_path)
		# sort files for consistency
		for file_list in file_lists:
			file_list.sort()
		return file_lists

	def load_my_files_to_data_sets_collection(self, data_sets_object):
		file_lists = self.find_my_files()
		print('%s: loading data from:' % self.dataset_specs.name)
		print(self.dataset_specs.data_dir)
		for file_list in file_lists:
			print('loading %s data from:' % file_list.name)
			for f in file_list.file_list:
				print(f)
			data_sets_object.LoadFromFile(
				'%s_%s' % (self.dataset_specs.name, file_list.name),
				file_list.file_list,
				max_seq_len=self.dataset_specs.max_seq_len)

	# API

	def run(self):
		self.create_dataset()
		self.limit_length_and_divide_datasets()

	@staticmethod
	def show_datasets_statistics(ds_gen_list):

		print('calculating statistics...')

		# load
		data_sets_collection = AlignmentDatasetsCollection(batch_size=1)
		for ds_gen in ds_gen_list:
			ds_gen.load_my_files_to_data_sets_collection(data_sets_collection)
		data_sets_collection.Init()

		# show statistics
		print('statistics:')
		data_sets_collection.print_statistics()

		print('max lengths:')
		print(data_sets_collection.max_obj_len)
		print(data_sets_collection.max_src_len)
		print(data_sets_collection.max_obj_lines)
		print(data_sets_collection.max_src_lines)

		print('vocabulary sizes:')
		print(data_sets_collection.obj_vocab_size)
		print(data_sets_collection.src_vocab_size)

		print('vocabularies:')
		print(data_sets_collection.vocab_obj.symbol2idx)
		print(data_sets_collection.vocab_src.symbol2idx)

def main():

	ds_gen_art = DatasetGenerator(
		data_gen=ArtificialDataGeneration(),
		dataset_specs=DatasetSpecs(
			name='art',
			data_dir=conf.target_dir_ds_art,
			max_seq_len=conf.global_max_seq_len,
			num_samples=conf.num_samples_art
		)
	)

	ds_gen_nat = DatasetGenerator(
		data_gen=NaturalDataGeneration(),
		dataset_specs=DatasetSpecs(
			name='nat',
			data_dir=conf.target_dir_ds_nat,
			max_seq_len=conf.global_max_seq_len,
			num_samples=-1
		)
	)

	ds_gen_list = []
	if 'art' in conf.data_to_generate:
		ds_gen_list.append(ds_gen_art)
	if 'nat' in conf.data_to_generate:
		ds_gen_list.append(ds_gen_nat)

	# generate data
	for ds_gen in ds_gen_list:
		ds_gen.run()

	# show statistics
	DatasetGenerator.show_datasets_statistics(ds_gen_list)

if __name__ == "__main__":
	main()

