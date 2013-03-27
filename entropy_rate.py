from itertools import izip
from math import exp, log
import multiprocessing

import numpy
from numpy.linalg import matrix_power
from matplotlib import pyplot
from scipy.misc import comb

from mpsim.math_helpers import normalize, multiply_vectors, dot_product
from mpsim.incentives import replicator_incentive, replicator_incentive_power, best_reply_incentive, logit_incentive, fermi_incentive
from mpsim.moran import linear_fitness_landscape, fermi_transform

## Multiprocessing support functions for multi-core processors ##

# Use generators for function params 

def constant_generator(x):
    while True:
        yield x
    
def params_gen(*args):
   return izip(*args)

def compute_entropy_rate_multiprocessing(args):
    """Simple function wrapper for pool.map"""
    e = single_test(*args)
    return (args[:-1], e)
   
def run_batches(args, processes=None, func=compute_entropy_rate_multiprocessing):
    """Runs calculations on multiple processing cores."""
    if not processes:
        processes = multiprocessing.cpu_count()
    params = params_gen(*args)
    pool = multiprocessing.Pool(processes=processes)
    try:
        results = pool.map(func, params)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print 'Control-C received. Exiting.'
        pool.terminate()
        exit()
    return results

## Mutations ##

def boundary_mutations(mu_ab=0.001, mu_ba=0.001):
    """Only mutations on the boundary states."""
    def f(N, i):
        if i == 0:
            return (mu_ab, 0.)
        elif i == N:
            return (0., mu_ba)
        return (0., 0.)
    return f

def uniform_mutations(mu_ab=0.001, mu_ba=0.001):
    """Mutations range over all states."""
    def f(N, i):
        return (mu_ab, mu_ba)
    return f
    
### Markov process calculations ##
            
def tridiagonal_entropy_rate(d, s):
    """Computes entropy rate from transition probabilities d and stationary distribution s."""
    e = 0.
    for k in d.keys():
        i = k[0]
        t = d[k]
        si = s[i]
        if (t == 0) or (si == 0):
            continue
        e += s[i] * t * log(t)
    return -e    

def approximate_entropy_rate(d, s):
    """Computes entropy rate from transition probabilities d and stationary distribution s."""
    entropies = []
    for i in range(len(s)):
        e = 0.
        for j in range(len(s)):
            k = d[i][j]
            try:
                e += -k * log(k)
            except:
                continue
        entropies.append(e)
    return dot_product(s, entropies)

def tridiagonal_stationary(d, N):
    """Computes the stationary distribution of a tridiagonal process from the transition probabilities."""
    ## log space version
    a = [0]
    b = []
    for i in range(0, N):
        b.append(log(d[(i, i+1)]) - log(d[(i+1, i)]))
        a.append(a[-1] + b[-1])
    p = [-1. * numpy.logaddexp.reduce(a)]
    for i in range(1, N+1):
        p.append(p[0] + a[i])
    return numpy.exp(p)

def approximate_stationary(N, d, iterations=None, power_multiple=15):
    """Approximate the stationary distribution by computing a large power of 
the transition matrix, using successive squaring."""
    # Could probably get away with a much smaller number.
    power = int(log(N) + power_multiple)
    # Use successive squaring to a large power
    m = matrix_power(d, 2**power)
    # Each row is the stationary distribution
    return m[0]
    
def single_test(*args):
    test_func = args[-1]
    args = args[:-1]
    if not test_func:
        test_func = moran_test
    return test_func(*args)

def test_prep(N, m, incentive_func=None, eta=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001):
    fitness_landscape = linear_fitness_landscape(m)
    if not incentive_func:
        incentive_func = replicator_incentive
    if not eta:
        incentive = incentive_func(fitness_landscape)
    else:
        incentive = incentive_func(fitness_landscape, eta)
    if not mutation_func:
        mutation_func = boundary_mutations
    mutations = mutation_func(mu_ab=mu_ab, mu_ba=mu_ba)
    return (fitness_landscape, incentive, mutations)
    
## Moran Process   
    
def moran_transitions(N, fitness_landscape=None, incentive=None, mutations=None, w=None):
    """Computes transition probabilities for the Markov process. Since the transition matrix is tri-diagonal (i.e. sparse), use a dictionary."""
    if not mutations:
        mutations = boundary_mutations()
    if not w:
        w = 1.
    d = dict()
    mu, nu = mutations(N, 0)
    d[(0,1)] = mu
    d[(0,0)] = 1. - mu
    mu, nu = mutations(N, N)
    d[(N, N-1)] = nu
    d[(N, N)] = 1. - nu
    for a in range(1, N):
        b = N - a
        if incentive:
            i = normalize(incentive([a,b]))
        else:
            i = normalize(multiply_vectors([a, b], fitness_landscape([a,b])))
        i[0] = 1. - w + w * i[0]
        i[1] = 1. - w + w * i[1]
        mu, nu = mutations(N, a)
        up = ((i[0] * (1. - mu) + i[1]*nu)) * float(b) / (a + b)
        down = ((i[0] * mu + i[1]*(1. - nu))) * float(a) / (a + b)
        d[(a, a+1)] = up
        d[(a, a-1)] = down
        d[(a, a)] = 1. - up - down
    return d

def moran_test(N, m, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001, verbose=False, report=False):
    """Compute the entropy rate for a single Markov process defined by the parameters."""
    (fitness_landscape, incentive, mutations) = test_prep(N, m, incentive_func=incentive_func, eta=eta, mutation_func=mutation_func, mu_ab=mu_ab, mu_ba=mu_ba)
    d = moran_transitions(N, incentive=incentive, mutations=mutations, w=w)
    s = tridiagonal_stationary(d, N)
    e = tridiagonal_entropy_rate(d, s)
    if verbose:
        print s
        for k in sorted(d.keys()):
            print k, d[k]
    if not report:
        return e
    return (e, s, d)

## n-fold Moran

def n_fold_moran_transitions(N, fitness_landscape=None, incentive=None, mutations=None, w=None, n=None):
    if not n:
        n = N
    d = moran_transitions(N, fitness_landscape=fitness_landscape, incentive=incentive, mutations=mutations, w=w)
    # Convert to matrix
    m = numpy.zeros((N+1, N+1))
    for k in d.keys():
        i, j = k
        m[i][j] = d[k]
    # Raise to n-th power
    transitions = matrix_power(m, n)
    return transitions

def n_fold_moran_test(N, m, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001, verbose=False, report=False):
    """Thin wrapper for approximate_test for code readability."""
    (fitness_landscape, incentive, mutations) = test_prep(N, m, incentive_func=incentive_func, eta=eta, mutation_func=mutation_func, mu_ab=mu_ab, mu_ba=mu_ba)
    transitions = moran_transitions(N, incentive=incentive, mutations=mutations, w=w)
    s = tridiagonal_stationary(transitions, N)
    d = n_fold_moran_transitions(N, incentive=incentive, mutations=mutations, w=w)
    e = approximate_entropy_rate(d,s)
    if verbose:
        print s
        #for k in sorted(d.keys()):
            #print k, d[k]
    if not report:
        return e
    return (e, s, d)       

## Wright Fisher Process  

def wright_fisher_transitions(N, fitness_landscape=None, incentive=None, mutations=None, w=None):
    """Computes transition probabilities for the Markov process. Since the transition matrix is tri-diagonal (i.e. sparse), use a dictionary."""
    if not mutations:
        mutations = boundary_mutations()
    if not w:
        w = 1.
    d = dict()
    d = numpy.zeros((N+1, N+1))
    mu, nu = mutations(N, 0)
    d[0][1] = mu
    d[0][0] = 1- mu
    mu, nu = mutations(N, N)
    d[N][N-1] = nu
    d[N][N] = 1 - nu
    for a in range(1, N):
        b = N - a
        if incentive:
            i = normalize(incentive([a,b]))
        else:
            i = normalize(multiply_vectors([a, b], fitness_landscape([a,b])))
        i[0] = 1. - w + w * i[0]
        i[1] = 1. - w + w * i[1]
        mu, nu = mutations(N, a)
        up = ((i[0] * (1. - mu) + i[1]*nu))
        down = ((i[0] * mu + i[1]*(1. - nu)))
        for j in range(0, N+1):
            if (j == 0) or (j == N):
                d[a][j] = exp(j * log(up) + (N-j) * log(down))
            else:
                d[a][j] = comb(N,j, exact=True) * exp(j * log(up) + (N-j) * log(down))
    return d
    
def wright_fisher_test(N, m, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001, verbose=False, report=False):
#def wright_fisher_test(*args, **kwargs):
    """Thin wrapper for approximate_test for code readability."""
    (fitness_landscape, incentive, mutations) = test_prep(N, m, incentive_func=incentive_func, eta=eta, mutation_func=mutation_func, mu_ab=mu_ab, mu_ba=mu_ba)
    d = wright_fisher_transitions(N, incentive=incentive, mutations=mutations, w=w)    
    s = approximate_stationary(N, d)
    e = approximate_entropy_rate(d,s)
    if verbose:
        print s
        #for k in sorted(d.keys()):
            #print k, d[k]
    if not report:
        return e
    return (e, s, d)    
    
if __name__ == '__main__':
    m = [[1,1],[1,1]]
    N = 1000
    mu = 0.01
    (e,s,d) = approximate_entropy_test(N, m, report=True, mu_ab=mu, mu_ba=mu)
    print e



    