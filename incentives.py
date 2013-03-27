import math
import numpy
from math_helpers import multiply_vectors, normalize, dot_product

def linear_fitness_landscape(m, beta=None, self_interaction=False):
    """Computes a fitness landscape from a game matrix given by m and a population vector (i,j) summing to N."""
    # m = array of rows
    def f(pop):
        N = sum(pop)
        if self_interaction:
            div = N
        else:
            div = N-1
        #pop = [x / float(div) for x in pop]
        fitness = []
        for i in range(len(pop)):
            # - m[i][i] if individuals do not interact with themselves.
            f = dot_product(m[i], pop)
            if not self_interaction:
                f -= m[i][i]
            fitness.append(f / float(div))
        return fitness
    return f

def replicator(fitness, q=1):
    if q == 1:
        def f(x):
            return multiply_vectors(x, fitness(x))
        return f
    def g(x):
        y = []
        for i in range(len(x)):
            y.append(math.pow(x[i], q))
        y = numpy.array(y)
        return y * fitness(x)
    return g
    

#def replicator_incentive_power(fitness, q):
    #def g(x):
        #y = []
        #for i in range(len(x)):
            #y.append(math.pow(x[i], q))
        #y = numpy.array(y)
        #return y * fitness(x)
    #return g

def best_reply(fitness):
    """Compute best reply to fitness landscape at state."""
    def g(state):
        f = fitness(state)
        try:
            dim = state.size
        except AttributeError:
            state = numpy.array(state)
            dim = state.size
        replies = []
        for i in range(dim):
            x = numpy.zeros(dim)
            x[i] = 1
            replies.append(numpy.dot(x, f))
        replies = numpy.array(replies)
        i = numpy.argmax(replies)
        x = numpy.zeros(dim)
        x[i] = 1
        return x
    return g

def logit(fitness, eta):
    def f(x):
        return numpy.exp(numpy.array(fitness(x)) / eta)
    return f

def fermi(fitness, eta):
    def f(x):
        return multiply_vectors(x, numpy.exp(numpy.array(fitness(x)) / eta))
    return f    
    