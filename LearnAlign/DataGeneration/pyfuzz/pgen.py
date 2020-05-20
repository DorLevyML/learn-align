from codegen.arithgen import IntegerGen, gen_max_int_gen

pgen_opts = {
    "module": {
		"children": [
			(1.0, "arith_integer"),
			(0.0, "arith_float")
		],
	   "mainloop": 25000,
	   "prog_size": 1,
	   "module_size": 1,
    },
	
    "arith_integer": {
		"children": [
			(1.0, "arith_integer"),
			(1.0, ("arith_integer", "local")),
			(2.0, "loop_integer"),
			(1.0, "change_global"),
			(1.0, "integer_closure"),
			(1.0, "tail_recursion"),
			(1.0, "classes"),
		],
        "max_children": 0,
        "numbers": [IntegerGen(0, 100), IntegerGen(0, 100)],
        "type": [(1.0, "thin"), (1.0, "fat")],
        "if": 0.10,
		
		"args_num" : 5,
		"ops_num": 2,
		"statements_num": {
			"min": 2,
			"max": 4
		},
		"statements": [
			(1.0, "arith"),
			(1.0, "if"),
			(1.0, "loop")
		],
		
		"loop_integer": {
			"numbers": [IntegerGen(0, 10)],
			"if": 0.40,
		}
    },
	
    "loop_integer": {
		"numbers": [IntegerGen(-10, 10)],
		"if": 0.10,
    },
	
    "change_global": {
		"numbers": [IntegerGen(-10, 10)],
    },
	
    "integer_closure": {
		"numbers": [IntegerGen(-10, 10)]
    },
	
    "tail_recursion": {
		"numbers": [IntegerGen(-10, 10)],
		"type": [(1.0, "standard"), (1.0, "closure"), (0.5, "fcall")],
    },

}

from pygen.cgen import Assignment, CallStatement, ForLoop, Module

from utils import eval_branches

from codegen.integergen import ArithIntegerGenerator


class ProgGenerator(object):

    def __init__(self, opts, rng):
        self.opts = opts
        self.module = None
        self.rng = rng

    def next_variable(self):
        nr = self.arg_number
        self.arg_number += 1
        return "var%d" % (nr, )

    def generate(self):
        """Instantiates a new module and fills it randomly."""
        self.module = Module(main=True)
        self.func_number = 1
        self.arg_number = 1
        lopts = self.opts["module"]

        self.prog_size = lopts["prog_size"]
        self.module_size = lopts["module_size"] - self.prog_size

        f = self.arith_integer(self.opts["arith_integer"])
        return f

        while self.module_size > 0 or self.prog_size > 0:
            main = []

            loop = ForLoop('i', ['%d' % (lopts["mainloop"],)], main)

            if "children" in lopts:
                branch = eval_branches(self.rng, lopts["children"])
                if branch == "arith_integer":
                    main.append(Assignment('x', '=', ['5']))
                    f = self.arith_integer(self.opts[branch], 2)
                    main.append(
                        Assignment('x',
                                   '=',
                                   [CallStatement(f,
                                                  ['x',
                                                   'i'])]))
                    main.append("print(x, end='')")

                    self.module.content.insert(0, f)
                if branch == "arith_float":
                    main.append(Assignment('x', '=', ['5.0']))
                    main.append("print(x, end='')")

            self.module.main_body.append("print('prog_size: %d')" %
                                        (lopts["prog_size"] - self.prog_size,))
            self.module.main_body.append(
                "print('func_number: %d')" %
                (self.func_number,))
            self.module.main_body.append(
                "print('arg_number: %d')" %
                (self.arg_number,))
            self.module.main_body.append(loop)

            created_size = lopts["prog_size"] - self.prog_size
            refill = min(created_size, self.module_size)

            self.module_size -= refill
            self.prog_size += refill
            break
			
        self.module.content.insert(0, 'from __future__ import print_function')

        return self.module

    def arith_integer(self, opts, globals=[]):
        """Insert a new arithmetic function using only integers."""
        gen = ArithIntegerGenerator(self.module, self, self.opts, self.rng)
        f = gen.generate(opts, globals)

        return f
