import csv
import math

from matplotlib import pyplot
import numpy

import heatmap
from entropy_rate import uniform_mutations, boundary_mutations, moran_test, wright_fisher_test, constant_generator, run_batches

## Plots ##

def ensure_digits(total, s):
    num_digits = len(str(s))
    total_digits = len(str(total))
    if num_digits < total_digits:
        return "0"*(total_digits - num_digits) + str(s)
    return s

def savefig(filename, dpi=400):
    pyplot.tight_layout()
    pyplot.savefig(filename, dpi=dpi, bbox_inches='tight')
    pyplot.clf()

def plot_transitions(N, m, mu=0.01, mutation_func=None):
    (e, s, d) = moran_test(N, m, incentive_func=None, eta=None, w=None, mutation_func=mutation_func, mu_ab=mu, mu_ba=5*mu, verbose=False, report=True)
    pyplot.figure()
    for i in range(1, N):
        up = d[(i,i+1)]
        down = d[(i, i-1)]
        null = d[(i,i)]
        pyplot.bar(i, down, bottom=0., color='r')
        pyplot.bar(i, null, bottom=down, color='g')
        pyplot.bar(i, up, bottom=down+null, color='b')
    pyplot.figure()
    pyplot.bar(range(len(s)), s)    

def static_plot(Ns, m, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001, test_func=None):
    """Plot entropy rate over a range of population sizes."""
    params = [Ns]
    for p in [m, incentive_func, eta, w, mutation_func, mu_ab, mu_ba, test_func]:
        params.append(constant_generator(p))
    es = run_batches(params)
    print es[-1]
    pyplot.plot(Ns, [x[-1] for x in es])
    return es

def mu_plot(N, m, incentive_func=None, eta=None, w=None, mutation_func=None, mus=None, test_func=None):
    """Plot entropy rate over a range of mutation probabilities."""
    params = [constant_generator(N)]
    for p in [m, incentive_func, eta, w, mutation_func]:
        params.append(constant_generator(p))
    params.append(mus)
    params.append(mus)
    params.append(constant_generator(test_func))
    es = run_batches(params)
    print es[-1]
    pyplot.plot([-math.log(mu) for mu in mus], [x[-1] for x in es])
    return es

    
def rs_plot(N, rs, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001, test_func=None):
    """Plot entropy rate over a range of fitness values."""
    ms = [ [[r,r],[1,1]] for r in rs]
    params = [constant_generator(N), ms]
    for p in [incentive_func, eta, w, mutation_func, mu_ab, mu_ba, test_func]:
        params.append(constant_generator(p))
    es = run_batches(params)
    print es[-1]
    pyplot.plot(rs, [x[-1] for x in es])
    return es    

### Heatmaps ##    
    
def compute_N_r_heatmap_data(Ns, rs, incentive_func=None, eta=None, w=None, mutation_func=None, mu_ab=0.001, mu_ba=0.001, test_func=None):
    data = []
    for r in rs:
        print r
        m = [[r,r],[1,1]]
        params = [Ns]
        for p in [m, incentive_func, eta, w, mutation_func, mu_ab, mu_ba, test_func]:
            params.append(constant_generator(p))
        es = run_batches(params)
        for (args, e) in es:
            N, m, incentive_func, eta, w, mutation_func, mu_ab, mu_ba = args
            data.append([N, r, mu_ab, mu_ba, e])
    return data
            
def heatmap_N_r_images(Ns, rs, mus, mutation_func=uniform_mutations, test_func=None):
    for i in range(len(mus)):
        print i
        mu = mus[i]
        data = compute_N_r_heatmap_data(Ns, rs, mutation_func=mutation_func, mu_ab=mu, mu_ba=mu, test_func=test_func)
        pyplot.clf()
        heatmap.main(data=data, xindex=1, yindex=0, cindex=-1, xfunc=float, yfunc=int, cfunc=float)
        index = ensure_digits(len(mus),i)
        pyplot.savefig("heatmap_%s.png" % index, dpi=400)

def compute_N_mu_heatmap_data(Ns, mus, r, incentive_func=None, eta=None, w=None, mutation_func=None, test_func=None):
    data = []
    m = [[r,r],[1,1]]
    for mu in mus:
        print mu
        params = [Ns]
        for p in [m, incentive_func, eta, w, mutation_func, mu, mu, test_func]:
            params.append(constant_generator(p))
        es = run_batches(params)
        for (args, e) in es:
            N, m, incentive_func, eta, w, mutation_func, mu_ab, mu_ba = args
            data.append([N, r, mu, -math.log(e)])
    return data
            
def heatmap_N_mu_images(r=1., mutation_func=boundary_mutations, test_func=None):
    #Ns = range(3, 1000,10)
    Ns = [int(3 + math.pow(1.5, x)) for x in range(0, 30)]
    mus = [math.pow(.4, x) for x in range(3, 22)]
    #mus = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0001, 0.00001]
    #mus = [0.1, 0.01,]
    data = compute_N_mu_heatmap_data(Ns, mus, r, mutation_func=mutation_func, test_func=test_func)
    pyplot.clf()
    heatmap.main(data=data, xindex=2, yindex=0, cindex=-1, xfunc=float, yfunc=int, cfunc=float)
    pyplot.savefig("heatmap_N_m" +  ".png", dpi=400)

def entropy_test(N=50):
    from mpsim.math_helpers import shannon_entropy, normalize
    ts = []
    for i in range(0, N/2):
        t = [(N-i)*(N-i), (N-i)*(N-i), i*(N-2*i)]
        #t = [float(x) / (N*(2*N-3*i)) for x in t]
        ts.append(normalize(t))
    es = [shannon_entropy(t) for t in ts]
    print max(es)
    pyplot.plot(range(0, N/2), es)
    pyplot.show()
    
    
if __name__ == '__main__':
    ##rs = numpy.arange(0.6,1.4,0.01)
    ##Ns = range(3, 100)
    ##mus = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0001, 0.00001]
    #rs = numpy.arange(0.6,1.4,0.01)
    #Ns = range(2, 100)
    #mus = [0.2, 0.1, 0.05, 0.04]
    #heatmap_N_r_images(Ns, rs, mus, mutation_func=boundary_mutations, test_func=moran_test)
    #heatmap_N_mu_images(r=.1, mutation_func=boundary_mutations)
    #heatmap_N_mu_images(r=1., mutation_func=uniform_mutations)
    
    pass
