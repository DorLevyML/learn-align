from pygen.cgen import Declaration, MultipleDeclaration, Assignment, CallStatement, IfStatement, ForLoop
from .arithgen import ArithGen, AllSumGen

from utils import eval_branches, FunctionGenerator
from .globalsgen import ChangeGlobalGenerator
from .recursion import TailRecursionGenerator


class ArithIntegerGenerator(FunctionGenerator):

    def __init__(self, module, stats, opts, rng):
        super(ArithIntegerGenerator, self).__init__()
        self.opts = opts
        self.module = module
        self.rng = rng
        self.stats = stats

    def get_iterable(self, opts, literals):
        from . import iterables
        iter_gen = iterables.IterableGenerator(
            self.module,
            self.stats,
            self.opts,
            self.rng)
        return iter_gen.get_iterable(literals, self.args)
		
    def generate_statement(self, opts, f, gen, literals, numbers):
        statement = eval_branches(self.rng, opts['statements'])
        
        if statement == 'if':
            result = self.next_variable()
            literals.add(result)

            exp1 = gen.generate(self.args)
            exp2 = gen.generate(self.args)

            clause = self.rng.choice(list(literals)) + ' < ' + self.rng.choice(self.args)

            i = IfStatement(clause,
                            [Assignment(result, '=', [exp1])],
                            [Assignment(result, '=', [exp2])])
            f.content.append(i)
			
        if statement == 'arith':
            result = self.next_variable()

            exp = gen.generate(self.args)
            f.content.append(Assignment(result, '=', [exp]))
            literals.add(result)

        if statement == 'loop':
		
            loop_opts = opts['loop_integer']
            
            loop_numbers = [n.set_rng(self.rng) for n in loop_opts['numbers']]
            
            result = self.next_variable()
            literals.add(result)
            
            iter = self.get_iterable(loop_opts, literals)
            
            loop_var = self.next_variable()
            literals.add(loop_var)
            
            l = ForLoop(loop_var, iter)
            
            if loop_opts['if'] > self.rng.random():
                exp1 = ArithGen(1, self.rng).generate(self.args)
                exp2 = ArithGen(1, self.rng).generate(self.args)
                
                clause = ' '.join([self.rng.choice(list(literals)),'<',self.rng.choice(self.args)])
                
                i = IfStatement(clause,
                                [Assignment(result, '+=', [exp1])],
                                [Assignment(result, '+=', [exp2])])
                l.content.append(i)

            else:
                exp = ArithGen(1, self.rng).generate(self.args)
                l.content.append(Assignment(result, '+=', [exp]))

            f.content.append(Assignment(result, '=', ['0']))
            f.content.append(l)

		
    def generate_child(self, opts, f, literals):
        branch = eval_branches(self.rng, opts['children'])
        if branch == 'arith_integer':
            gen = ArithIntegerGenerator(
                self.module,
                self.stats,
                self.opts,
                self.rng)
            c = gen.generate(opts, 2)
            self.module.content.insert(0, c)

            args = self.rng.sample(list(literals), 2)
            result = self.next_variable()

            call = Assignment(result, '=', [CallStatement(c, args)])
            f.content.append(call)
            literals.add(result)

        if branch == ('arith_integer', 'local'):
            gen = ArithIntegerGenerator(
                self.module,
                self.stats,
                self.opts,
                self.rng)
            c = gen.generate(opts, 2, list(literals))

            f.content.append(c)

            args = self.rng.sample(list(literals), 2)
            result = self.next_variable()

            call = Assignment(result, '=', [CallStatement(c, args)])
            f.content.append(call)
            literals.add(result)

        if branch == 'loop_integer':
            gen = LoopIntegerGenerator(
                self.module,
                self.stats,
                self.opts,
                self.rng)

            c = gen.generate(self.opts['loop_integer'], 2, [])
            self.module.content.insert(0, c)

            args = self.rng.sample(list(literals), 2)
            result = self.next_variable()

            call = Assignment(result, '=', [CallStatement(c, args)])
            f.content.append(call)
            literals.add(result)

        if branch == 'change_global':
            gen = ChangeGlobalGenerator(
                self.module,
                self.stats,
                self.opts,
                self.rng)

            c = gen.generate(self.opts['change_global'], 0, [])
            self.module.content.insert(0, c)

            result = self.next_variable()

            call = Assignment(result, '=', [CallStatement(c, [])])
            f.content.append(call)
            literals.add(result)

        if branch == 'integer_closure':
            gen = IntegerClosureGenerator(
                self.module,
                self.stats,
                self.opts,
                self.rng)
            func = gen.generate(self.opts['integer_closure'], 2, [])

            args = self.rng.sample(list(literals), 2)
            result = self.next_variable()

            call = Assignment(result, '=', [CallStatement(func, args)])
            f.content.append(call)
            literals.add(result)

        if branch == 'tail_recursion':
            gen = TailRecursionGenerator(
                self.module,
                self.stats,
                self.opts,
                self.rng)
            func = gen.generate(self.opts['tail_recursion'], 2, [])

            args = self.rng.sample(list(literals), 2)
            result = self.next_variable()

            call = Assignment(result, '=', [CallStatement(func, args)])
            f.content.append(call)
            literals.add(result)

        if branch == 'classes':
            from . import classes
            gen = classes.ClassGenerator(
                self.module,
                self.stats,
                self.opts,
                self.rng)
            result = gen.generate_inline(literals)

            f.content.extend(result)

    def generate(self, opts, globals=[]):
        """Insert a new arithmetic function using only integers."""
        self.generate_arguments(opts["args_num"])

        f = self.create_function(self.args)

        literals = set(self.args) | set(globals)

        children = min(
            self.rng.randint(0,
                             opts['max_children']),
            self.stats.prog_size)
        if children > 0:
            self.stats.prog_size -= children
            for i in range(children):
                self.generate_child(opts, f, literals)

        numbers = [n.set_rng(self.rng) for n in opts['numbers']]
        
        ops_num        = opts['ops_num']
        statements_num = opts['statements_num']
        min_statements = statements_num['min']
        max_statements = statements_num['max']
        gen = ArithGen(ops_num, self.rng)
        for i in range(self.rng.randint(min_statements, max_statements)):
            self.generate_statement(opts, f, gen, literals, numbers)

        exp = AllSumGen().generate(list(literals))
        f.content.append('return %s;' % exp)

        f.content.insert(0, MultipleDeclaration(self.variables_list))
        
        return f


class LoopIntegerGenerator(FunctionGenerator):

    def __init__(self, module, stats, opts, rng):
        self.opts = opts
        self.module = module
        self.rng = rng
        self.stats = stats

    def get_iterable(self, opts, literals):
        from . import iterables
        iter_gen = iterables.IterableGenerator(
            self.module,
            self.stats,
            self.opts,
            self.rng)
        return iter_gen.get_iterable(literals)

    def generate(self, opts, args_num, globals):
        """Insert a new function with a loop containing some integer
        operations."""
        args = self.generate_arguments(args_num)

        literals = set(args) | set(globals)
        numbers = [n.set_rng(self.rng) for n in opts['numbers']]

        f = self.create_function(args)

        result = self.next_variable()
        literals.add(result)

        iter = self.get_iterable(opts, literals)

        loop_var = self.next_variable()
        literals.add(loop_var)

        l = ForLoop(loop_var, iter)

        if opts['if'] > self.rng.random():
            exp1 = ArithGen(1, self.rng).generate(list(literals) + numbers)
            exp2 = ArithGen(1, self.rng).generate(list(literals) + numbers)

            clause = ' '.join(
                [self.rng.choice(list(literals)),
                             '<',
                             self.rng.choice(list(literals))])

            i = IfStatement(clause,
                            [Assignment(result, '+=', [exp1])],
                            [Assignment(result, '+=', [exp2])])
            l.content.append(i)

        else:
            exp = ArithGen(1, self.rng).generate(list(literals) + numbers)
            l.content.append(Assignment(result, '+=', [exp]))

        f.content.append(Assignment(result, '=', ['0']))
        f.content.append(l)
        f.content.append('return ' + result)

        return f


class IntegerClosureGenerator(FunctionGenerator):

    def __init__(self, module, stats, opts, rng):
        self.opts = opts
        self.module = module
        self.rng = rng
        self.stats = stats

    def generate(self, opts, args_num, globals=[]):
        """Insert a new arithmetic function using only integers."""
        args = self.generate_arguments(args_num)

        closure = self.create_function(args)

        gen = self.create_function([])

        if opts['numbers']:
            number_gen = self.rng.choice(opts['numbers'])
            number_gen.set_rng(self.rng)
            number = number_gen()
        else:
            number = 0

        gen.content.extend(
            [
                'closure = [%s]' % (number, ),
                closure,
                Assignment('func', '=', [closure.name]),
                'return func',
            ]
        )

        c_var = self.next_variable()

        self.module.content.insert(0, gen)
        self.module.content.insert(
            1,
            Assignment(c_var,
                       '=',
                       [CallStatement(gen,
                                      [])]))

        gen_ai = ArithIntegerGenerator(
            self.module,
            self.stats,
            self.opts,
            self.rng)
        f = gen_ai.generate(self.opts['arith_integer'], args_num, [])

        self.module.content.insert(0, f)

        closure.content.append(
            Assignment('closure[0]',
                       '+=',
                       [CallStatement(f,
                                      args)]))
        closure.content.append('return closure[0]')

        return c_var
