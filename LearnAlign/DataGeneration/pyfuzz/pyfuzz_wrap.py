'''
this wrap is based on the example code in pgen-example.py
'''

import random
import sys

from pgen import pgen_opts, ProgGenerator
from pygen.cgen import CodeGenerator

if __name__ == "__main__":
	
	sourceFilePath = sys.argv[1]
	
	pgen = ProgGenerator(pgen_opts, random.Random())
	m = pgen.generate()
	cgen = CodeGenerator()
	txt = cgen.generate(m)

	f = open(sourceFilePath, 'w')
	f.write(txt)
	f.close()
