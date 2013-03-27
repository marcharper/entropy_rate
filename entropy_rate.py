from itertools import izip
from math import exp, log
import multiprocessing

import numpy
from numpy.linalg import matrix_power
from matplotlib import pyplot
from scipy.misc import comb

from math_helpers import normalize, dot_product
from incentives import linear_fitness_landscape, replicator, best_reply, fermi, logit

## Multiprocessing support functions for multi-core processors ##

# Use generators for function params 

def constant_generator(x):
    while True:
        yield x

def params_gen(constant_parameters, variable_parameters):
    """Manage parameters for multiprocessing. Functional parameters cannot be pickled, hence this workaround."""
    parameters = []
    #N, process, incentive_name, m, q, eta, w, mutations_name, mu_ab, mu_ba, n = args
    for p, default in [("N", 10), ("process", "moran"), ("incentive", "replicator"), ("m", [[1,1],[1,1]]), ("q", 1), ("eta", 1), ("w", 1), ("mutations", "uniform"), ("mu_ab", 0.01), ("mu_ba", 0.01), ("n", None)]:
        try:
            value = constant_generator(constant_parameters[p])
        except KeyError:
            try:
                value = variable_parameters[p]
            except KeyError:
                value = constant_generator(default)
        parameters.append(value)
    return izip(*parameters)

def compute_entropy_rate_multiprocessing(args):
    """Simple function wrapper for pool.map"""
    e = compute_entropy_rate(*args)
    return (args, e)
   
def run_batches(constant_parameters, variable_parameters, num_processes=None, func=compute_entropy_rate_multiprocessing):
    """Runs calculations on multiple processing cores."""
    if not num_processes:
        num_processes = multiprocessing.cpu_count()
    params = params_gen(constant_parameters, variable_parameters)
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        results = pool.map(func, params)
        #result = pool.apply_async(func, list(params))
        #results = result.get()
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
            
def tridiagonal_entropy_rate(transitions, stationary):
    """Computes entropy rate from transition probabilities d and stationary distribution s."""
    e = 0.
    for k in transitions.keys():
        i = k[0]
        t = transitions[k]
        si = stationary[i]
        if (t == 0) or (si == 0):
            continue
        e += stationary[i] * t * log(t)
    return -e    

def approximate_entropy_rate(transitions, stationary):
    """Computes entropy rate from transition probabilities d and stationary distribution s."""
    entropies = []
    for i in range(len(stationary)):
        e = 0.
        for j in range(len(stationary)):
            k = transitions[i][j]
            try:
                e += -k * log(k)
            except:
                continue
        entropies.append(e)
    return dot_product(stationary, entropies)

def tridiagonal_stationary(transitions, N):
    """Computes the stationary distribution of a tridiagonal process from the transition probabilities."""
    ## log space version
    a = [0]
    b = []
    for i in range(0, N):
        b.append(log(transitions[(i, i+1)]) - log(transitions[(i+1, i)]))
        a.append(a[-1] + b[-1])
    p = [-1. * numpy.logaddexp.reduce(a)]
    for i in range(1, N+1):
        p.append(p[0] + a[i])
    return numpy.exp(p)

def approximate_stationary(N, transitions, power_multiple=15):
    """Approximate the stationary distribution by computing a large power of the transition matrix, using successive squaring."""
    # Could probably get away with a much smaller power.
    power = int(log(N) + power_multiple)
    # Use successive squaring to a large power
    m = matrix_power(transitions, 2**power)
    # Each row is the stationary distribution
    return m[0]
    
## Moran Process   
    
def moran_transitions(N, incentive, mutations, w=None):
    """Computes transition probabilities for the Markov process. Since the transition matrix is tri-diagonal (i.e. sparse), use a dictionary."""
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
        i = normalize(incentive([a,b]))
        i[0] = 1. - w + w * i[0]
        i[1] = 1. - w + w * i[1]
        mu, nu = mutations(N, a)
        up = ((i[0] * (1. - mu) + i[1]*nu)) * float(b) / (a + b)
        down = ((i[0] * mu + i[1]*(1. - nu))) * float(a) / (a + b)
        d[(a, a+1)] = up
        d[(a, a-1)] = down
        d[(a, a)] = 1. - up - down
    return d

def moran_entropy_rate(N, incentive, mutations, w=None, verbose=False, report=False):
    """Compute the entropy rate for a single Markov process defined by the parameters."""
    transitions = moran_transitions(N, incentive=incentive, mutations=mutations, w=w)
    stationary = tridiagonal_stationary(transitions, N)
    e = tridiagonal_entropy_rate(transitions, stationary)
    if verbose:
        print stationary
        for k in sorted(d.keys()):
            print k, transitions[k]
    if not report:
        return e
    return (e, stationary, transitions)

## n-fold Moran

def n_fold_moran_transitions(N, incentive, mutations, w=None, n=None):
    if not n:
        n = N
    d = moran_transitions(N, incentive, mutations, w=w)
    # Convert to matrix
    m = numpy.zeros((N+1, N+1))
    for k in d.keys():
        i, j = k
        m[i][j] = d[k]
    # Raise to n-th power
    transitions = matrix_power(m, n)
    return transitions

def n_fold_moran_entropy_rate(N, incentive, mutations, w=None, verbose=False, report=False, n=None):
    """Thin wrapper for approximate_test for code readability."""
    transitions = moran_transitions(N, incentive=incentive, mutations=mutations, w=w)
    stationary = tridiagonal_stationary(transitions, N)
    transitions = n_fold_moran_transitions(N, incentive, mutations, w=w, n=n)
    e = approximate_entropy_rate(transitions, stationary)
    if verbose:
        print s
    if not report:
        return e
    return (e, stationary, transitions)

## Wright Fisher Process  

def wright_fisher_transitions(N, incentive, mutations, w=None):
    """Computes transition probabilities for the Markov process. Since the transition matrix is tri-diagonal (i.e. sparse), use a dictionary."""
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
        #i = normalize(multiply_vectors([a, b], fitness_landscape([a,b])))
        i = normalize(incentive([a,b]))
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
    
def wright_fisher_entropy_rate(N, incentive, mutations, w=None, verbose=False, report=False):
    """Thin wrapper for approximate_test for code readability."""
    transitions = wright_fisher_transitions(N, incentive=incentive, mutations=mutations, w=w)    
    stationary = approximate_stationary(N, transitions)
    e = approximate_entropy_rate(transitions, stationary)
    if verbose:
        print s
    if not report:
        return e
    return (e, stationary, transitions)    

def compute_entropy_rate(N, process="moran", incentive="replicator", m=[[1,1],[1,1]], q=1, eta=None, w=None, mutations=None, mu_ab=None, mu_ba=None, n=None, verbose=False, report=False):
    """Convenience function for unifying call procedure for the various processes."""
    #N, process, incentive_name, m, q, eta, w, mutations_name, mu_ab, mu_ba, n = args
    fitness_landscape = linear_fitness_landscape(m)
    incentive_name = incentive
    if incentive_name == "replicator":
        incentive = replicator(fitness_landscape, q=q)
    elif incentive_name == "best_reply":
        incentive = best_reply(fitness_landscape)
    elif incentive_name == "logit":
        incentive = logit(fitness_landscape, eta)
    elif incentive_name == "fermi":
        incentive = best_reply(fitness_landscape, eta)
    else:
        raise ValueError, "Argument 'incentive' must be 'replicator', 'best_reply', 'fermi', or 'logit'"
    mutations_name = mutations
    if mutations_name == "uniform":
        mutations = uniform_mutations(mu_ab, mu_ba)
    elif mutations_name == "boundary":
        mutations = boundary_mutations(mu_ab, mu_ba)
    else:
        raise ValueError, "Argument 'mutations' must be 'uniform', or 'boundary'"        
    process = process.lower()
    if process == "moran":
        r = moran_entropy_rate(N, incentive, mutations, w=w, verbose=verbose, report=report)
    elif process == "wright-fisher":
        r = wright_fisher_entropy_rate(N, incentive, mutations, w=w, verbose=verbose, report=report)
    elif process == "n-fold-moran":
        r = n_fold_moran_entropy_rate(N, incentive, mutations, w=w, verbose=verbose, report=report, n=n)
    else:
        raise ValueError, "Argument 'process' must be 'moran', 'n-fold-moran', or 'wright-fisher'"
    return r    

if __name__ == '__main__':
    # Simple example for each process process.
    N = 50
    mutations = uniform_mutations(mu_ab=0.001, mu_ba=0.001)
    m = [[1,2],[2,1]]
    fitness_landscape = linear_fitness_landscape(m)
    incentive = replicator_incentive(fitness_landscape, q=1.15)
    for process in ["moran", "n-fold-moran", "wright-fisher"]:
        (entropy_rate, stationary_distribution, transitions) = compute_entropy_rate(N, incentive, mutations, process=process, report=True)
        print process, entropy_rate

    