import csv
import math
import os

from matplotlib import pyplot
import numpy

import heatmap
from entropy_rate import uniform_mutations, boundary_mutations, compute_entropy_rate, constant_generator, run_batches

import matplotlib
font = {'size': 22}
matplotlib.rc('font', **font)

## Plots ##

def ensure_directory(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

def ensure_digits(total, s):
    """Make sure filenames are sequential."""
    num_digits = len(str(s))
    total_digits = len(str(total))
    if num_digits < total_digits:
        return "0"*(total_digits - num_digits) + str(s)
    return s

def savefig(filename, dpi=400, directory=None, verbose=True):
    if directory:
        ensure_directory(directory)
        filename = os.path.join(directory, filename)
    #pyplot.tight_layout()
    if verbose:
        print "saving", filename
    pyplot.savefig(filename, dpi=dpi, bbox_inches='tight')
    pyplot.clf()

def entropy_rate_vs_N(Ns, process, m, incentive=None, eta=None, w=None, mutations=None, mu_ab=0.001, mu_ba=0.001, div_log=False, linewidth=1.5):    
    """Plot entropy rate over a range of population sizes."""
    variable_parameters = dict(N=Ns)
    constant_parameters = dict(m=m, incentive=incentive, eta=eta, w=w, mutations=mutations, mu_ab=mu_ab, mu_ba=mu_ba, process=process)
    es = run_batches(constant_parameters, variable_parameters)
    if div_log:
        pyplot.plot(Ns, [(es[i][-1])/(math.log(Ns[i])) for i in range(len(es))], linewidth=linewidth)
    else:
        pyplot.plot(Ns, [es[i][-1] for i in range(len(es))], linewidth=linewidth)
    print mu_ab, es[-1][-1]
    return es

### Heatmaps ##    
    
def compute_N_r_heatmap_data(Ns, rs, incentive=None, eta=None, w=None, mutations=None, mu_ab=0.001, mu_ba=0.001, process=None):
    data = []
    for r in rs:
        print r
        m = [[r,r],[1,1]]
        variable_parameters = dict(N=Ns)
        constant_parameters = dict(m=m, incentive=incentive, eta=eta, w=w, mutations=mutations, mu_ab=mu_ab, mu_ba=mu_ba, process=process)
        es = run_batches(constant_parameters, variable_parameters)
        for (args, e) in es:
            N, process, incentive_name, m, q, eta, w, mutations_name, mu_ab, mu_ba, n = args
            data.append([N, r, mu_ab, mu_ba, e])
    return data
            
def heatmap_N_r_images(Ns, rs, mus, mutations=None, process=None, incentive=None, directory=None):
    for i in range(len(mus)):
        print i
        mu = mus[i]
        data = compute_N_r_heatmap_data(Ns, rs, mutations=mutations, mu_ab=mu, mu_ba=mu, process=process, incentive=incentive)
        pyplot.clf()
        heatmap.main(data=data, xindex=1, yindex=0, cindex=-1, xfunc=float, yfunc=int, cfunc=float)
        index = ensure_digits(len(mus),i)
        savefig("heatmap_%s.png" % index, directory=directory)

