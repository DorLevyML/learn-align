from pygen.cgen import *


def eval_branches(rng, branches):
    total = sum((chance for chance, _ in branches))
    val = rng.random() * total

    for chance, result in branches:
        if val < chance:
            return result
        else:
            val -= chance


class FunctionGenerator(object):

    def __init__(self):

        self.variables_list = []
        self.args = None

    def generate_arguments(self, args_num):
        self.args = [
            "a" + str(
                i) for i in range(
            self.stats.arg_number,
                    self.stats.arg_number + args_num)]
        self.stats.arg_number += args_num

    def next_variable(self):
        nr = self.stats.arg_number
        self.stats.arg_number += 1
        var = "v%d" % (nr, )
        
        # add to list of variables so we'll declare them later
        self.variables_list.append(var)
        
        return var

    def create_function(self, args):
        f = Function("func%d" % (self.stats.func_number,), args, [])
        self.stats.func_number += 1
        return f

    def create_class(self):
        c = Class("class%d" % (self.stats.func_number,))
        self.stats.func_number += 1
        return c

    def create_method(self, args):
        m = Method("func%d" % (self.stats.func_number,), args)
        self.stats.func_number += 1
        return m
