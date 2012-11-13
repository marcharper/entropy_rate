from itertools import izip
import inspect
import multiprocessing
import sys
from math import exp, log, isnan
from mpsim.incentives import replicator_incentive, replicator_incentive_power, best_reply_incentive, logit_incentive, fermi_incentive

import numpy
from matplotlib import pyplot

## Multiprocessing ##

#def default_parameters(N=3, m=[[1,1],[1,1]], mutation_func=boundary_mutations, mu_ab=0.001, mu_ba=0.001, eta=None, w=1., incentive_func=replicator_incentive, eta=None):
    #d = dict()
    #d['N'] = N
    #pass

#def process_parameters():
    #fitness_landscape = linear_fitness_landscape(m)
    #if not eta:
        #incentive = incentive_func(fitness_landscape)
    #else:
        #incentive = incentive_func(fitness_landscape, eta)
    #mutations = mutation_func(mu_ab, mu_ba)

def constant_generator(x):
    while True:
        yield x
    
def params_gen(*args):
    #if not inspect.isgenerator(ms):
        #ms = constant_generator(ms)
   return izip(*args)
   
def compute_entropy_rate_multiprocessing(args):
    e = single_test(*args)
    return (args, e)
   
def run_batches(args, processes=None, func=compute_entropy_rate_multiprocessing):
    """Runs simulations on multiple processing cores in batch sizes dictated by iters_gen, posting data to callbacks to reduce memory footprint."""
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
    
## Math helpers ##

#vector
def normalize(x):
    s = float(sum(x))
    for j in range(len(x)):
        x[j] /= s
    return x

##dictionary
#def normalize_dictionary(x):
    #s = float(sum(x.values()))
    #for k in x.keys():
        #x[k] /= s
    #return x

def multiply_vectors(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i]*b[i])
    return c    

def dot_product(a, b):
    c = 0
    for i in range(len(a)):
        c += a[i] * b[i]
    return c    

def arange(a, b, steps=100):
    """Similar to numpy.arange"""
    delta = (b - a) / float(steps)
    xs = []
    for i in range(steps):
        x = a + delta * i
        xs.append(x)
    return xs

## Landscapes ##

def fermi_transform(landscape, beta=1.0):
    def f(x):
        fitness = list(map(lambda z: exp(beta*z), landscape(x)))
        return fitness
    return f    
    
def linear_fitness_landscape(m, beta=None, self_interaction=False):
    """Computes a fitness landscape for a three player game from a matrix given by m and a population vector (i,j,...,k) summing to N."""
    # m = array of rows
    def f(pop):
        fitness = []
        for i in range(len(pop)):
            # - m[i][i] because we assume that individuals do not interact with themselves.
            f = dot_product(m[i], pop)
            if not self_interaction:
                f -= m[i][i] 
            fitness.append(f)
        return fitness
    if beta:
        f = fermi_transform(f, beta)
    return f

## Mutations ##

def boundary_mutations(mu_ab=0.001, mu_ba=0.001):
    def f(N, i):
        if i == 0:
            return (mu_ab, 0.)
        elif i == N:
            return (0., mu_ba)
        return (0., 0.)
    return f

def uniform_mutations(mu_ab=0.001, mu_ba=0.001):
    def f(N, i):
        return (mu_ab, mu_ba)
    return f
    
## Markov processors ##
    
#def transitions(N, fitness_landscape, incentive=None, mu=0.001, mu_ab=0., mu_ba=0.):
    #d = dict()
    #d[(0,1)] = mu
    #d[(N, N-1)] = mu
    #for a in range(1, N):
        #b = float(N - a)
        #if incentive:
            #birth = normalize(incentive(normalize([a,b])))
        #else:
            #birth = normalize(multiply_vectors([a, b], fitness_landscape([a,b])))
        ##up = log(birth[0]) + log(b) - log(a + b)
        ##down = log(birth[1]) + log(a) - log(a + b)
        #up = birth[0] * b / (a + b)
        #down = birth[1] * a / (a + b)
        #d[(a, a+1)] = up
        #d[(a, a-1)] = down
    #return d

    
def transitions(N, fitness_landscape=None, incentive=None, mutations=None, w=None):
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

#def lognormalize(x):
    #a = numpy.logaddexp.reduce(x)
    #return numpy.exp(x - a)    
    
def stationary(d, N):
    ## log space version
    # Could find N from d.keys()
    a = [0]
    b = []
    for i in range(0, N):
        b.append(log(d[(i, i+1)]) - log(d[(i+1, i)]))
        #b.append(d[(i, i+1)] - d[(i+1, i)])
        a.append(a[-1] + b[-1])
    p = [-1. * numpy.logaddexp.reduce(a)]
    for i in range(1, N+1):
        p.append(p[0] + a[i])
    return numpy.exp(p)
    
def entropy_rate(d, s):
    e = 0.
    for k in d.keys():
        i = k[0]
        e += s[i] * d[k] * log(d[k])
    return -e

def single_test(N, m, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001):
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
    d = transitions(N, incentive=incentive, mutations=mutations, w=w)
    s = stationary(d, N)
    #print s[0], s[-1], s[len(s) // 2]
    e = entropy_rate(d, s)
    return e    
    
## Plots ##

def static_plot(Ns, m, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001):
    es = []
    #for N in range(start, end+1):
        #fitness_landscape = fitness_static(r)
        #d = transitions(N, fitness_landscape, mu=mu)
        #s = stationary(d, N)
        #es.append(entropy_rate(d, s))
    #Ns = range(start, end+1)
    params = [Ns]
    for p in [m, incentive_func, eta, w, mutation_func, mu_ab, mu_ba]:
        params.append(constant_generator(p))
    es = run_batches(params)
    print es[-1]
    pyplot.plot(Ns, [x[-1] for x in es])
    return es

def compute_heatmap_data(rs, Ns, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001):
    outfilename = "entropy_rate_r_vs_n.csv"
    import csv
    writer = csv.writer(open(outfilename,'w'))
    for r in rs:
        print r
        m = [[r,r],[1,1]]
        params = [Ns]
        for p in [m, incentive_func, eta, w, mutation_func, mu_ab, mu_ba]:
            params.append(constant_generator(p))
        es = run_batches(params)
        for (args, e) in es:
            # N, r, mu, e
            N, m, incentive_func, eta, w, mutation_func, mu_ab, mu_ba = args
            writer.writerow([N, r, mu_ab, e])
            
def heatmap_images(mutation_func=uniform_mutations):
    import heatmap
    step = 0.1
    stop = 2
    rs = numpy.arange(0.6,1.4,0.01)
    Ns = range(3, 150)
    mus = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0001, 0.00001]
    #mus = [0.1, 0.01,]
    for i in range(len(mus)):
        print i
        mu = mus[i]
        compute_heatmap_data(rs, Ns, mutation_func=mutation_func, mu_ab=mu, mu_ba=mu)
        pyplot.clf()
        heatmap.main("entropy_rate_r_vs_n.csv")
        pyplot.savefig("heatmap_" + str(i) + ".png", dpi=240)
        
def max_population(rs, Ns, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001):
    """Calculates the population size N that produces the maximum entropy rate for a landscape [[r,r],[1,1]] by brute force."""
    maximums = []
    for r in rs:
        m = [[r,r],[1,1]]
        params = [Ns]
        for p in [m, incentive_func, eta, w, mutation_func, mu_ab, mu_ba]:
            params.append(constant_generator(p))
        es = run_batches(params)
        for i in range(len(es)-1):
            if es[i+1][-1] < es[i][-1]:
                maximums.append((r, i, es[i][-1]))
                if r == 1:
                    print r, i, es[i][-1]
                break
    return maximums        
    
def figures():
    m = [[1,1],[1,1]]
    Ns = range(3, 1000)
    for mu in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]:
        static_plot(Ns, m, mu_ab=mu, mu_ba=mu, mutation_func=uniform_mutations)
    pyplot.show()
    
    #mus = [0.05, 0.01, 0.005, 0.003, 0.001]
    #Nmax = [30,105,205,350,1050]
    #for i in range(len(mus)):
        #mu = mus[i]
        #print i, mu
        #rs = numpy.arange(0.9,1.1,0.005)
        #Ns = range(3, Nmax[i]+1)
        ## no maxima for uniform_mutations
        #results = max_population(rs, Ns, mu_ab=mu, mu_ba=mu, mutation_func=boundary_mutations)
        #pyplot.plot([x[0] for x in results], [x[1] for x in results])
    #pyplot.show()     
    
    #heatmap_images(mutation_func=boundary_mutations)
    #heatmap_images(mutation_func=uniform_mutations)

def binary_entropy(p):
    return -p*log(p) - (1-p) * log(1-p)
    
if __name__ == '__main__':
    #print binary_entropy(1./6)
    #exit()
    #m = [[1,1],[1,1]]
    #Ns = range(3, 10000, 10)
    #mutation_func=uniform_mutations
    #print single_test(100000, m, mutation_func=mutation_func, mu_ab=.000001, mu_ba=.000001)
    #print single_test(100000, m, mutation_func=mutation_func, mu_ab=.000001, mu_ba=.000005)
    ##static_plot(Ns, m, mutation_func=uniform_mutations, mu_ab=.001, mu_ba=.001)
    ##static_plot(Ns, m, mutation_func=uniform_mutations, mu_ab=.001, mu_ba=.005)
    ##pyplot.show()
    #exit()

    #m = [[1,1],[1,1]]
    #Ns = range(3, 5000)
    #for mu in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]:
        #static_plot(Ns, m, mu_ab=mu, mu_ba=mu)
    #pyplot.show()
  
    figures()

    
    ## run to 50000 to see if converges
    #for mu in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]:
        #neutral_plot(3, 5000, mu=mu)
    #pyplot.show()
    
    #rs = [0.1, 0.5, 1., 1.1, 2.]
    ##rs = numpy.arange(0.1, 2.1, 0.1)
    ##rs = [0.9, 0.99, 0.999, 1., 1.001, 1.01, 1.1]
    #Ns = range(3, 1000)
    #for r in rs:        
        #print r
        #m = [[r,r],[1,1]]
        #static_plot(Ns, m, mu_ab=0.01, mu_ba=0.01)
    #pyplot.show()
    
    #rs = [1.1]
    #for r in rs:        
        #print r
        #m = [[r,r],[1,1]]
        #static_plot(m, end=500, mu=0.001, incentive_name="replicator_incentive_power", eta=3.)
    #pyplot.show()
    #exit()

    #r = 1.1
    #etas = [10, 100, 1000]
    #for eta in etas:        
        #print eta
        #m = [[r,r],[1,1]]
        #static_plot(m, end=1500, mu=0.001, incentive_name="fermi_incentive", eta=eta)
    #pyplot.show()
    #exit()    
    
    ## Standard 2x2 games
    #ms = []
    ##ms.append([[1,1],[1,1]])
    #ms.append([[1,2],[1,3]])
    ##ms.append([[0,1],[1,0]])
    #ms.append([[1,2],[2,1]])
    #ms.append([[2,1],[1,2]])
    ##ms.append([[1,2],[5,1]])
    #ms.append([[3,1],[5,2]])
    #start=3
    #end=1000
    #for m in ms:
        #static_plot(m, start=start, end=end, mu=0.001)
        ##pyplot.plot(range(start, end), 
    #pyplot.show()
    
    pass

    