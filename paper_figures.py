import numpy
from matplotlib import pyplot

import heatmap
from math_helpers import shannon_entropy, kl_divergence
from entropy_rate import compute_entropy_rate, run_batches
from plots import entropy_rate_vs_N, heatmap_N_r_images, savefig

# Font config for plots
import matplotlib
font = {'size': 22}
matplotlib.rc('font', **font)

# Figure 1 in text
def distributions_figure(N, m, mu=0.01, k=1., mutations=None):
    """Plot transition entropies and stationary distributions."""
    (e, s, d) = compute_entropy_rate(N, process="moran", incentive="replicator", m=m, mutations=mutations, mu_ab=mu, mu_ba=k*mu, verbose=False, report=True)
    print e
    dist = (d[(0,0)], d[(0,1)])
    transitions = [dist]
    transition_entropies = [shannon_entropy(dist)]
    for i in range(1, N):
        up = d[(i,i+1)]
        down = d[(i, i-1)]
        null = d[(i,i)]
        dist = (down, null, up)
        transitions.append(dist)
        transition_entropies.append(shannon_entropy(dist))
    dist = (d[(N,N-1)], d[(N,N)])
    transitions.append(dist)
    transition_entropies.append(shannon_entropy(dist))
    print transition_entropies[0], transition_entropies[-1], max(transition_entropies)

    ## Plots
    # Transition distributions
    ax1 = pyplot.subplot(311, axisbg = 'w')
    pyplot.xlim(0, N)
    pyplot.ylim(0, 1)
    width = 1.0
    down = 0
    null, up = transitions[0]
    pyplot.bar(0, down, bottom=0., color='r', width=width)
    pyplot.bar(0, null, bottom=down, color='g', width=width)
    pyplot.bar(0, up, bottom=down+null, color='b', width=width)
    for i in range(1, N):
        down, null, up = transitions[i]
        pyplot.bar(i, down, bottom=0., color='r', width=width)
        pyplot.bar(i, null, bottom=down, color='g', width=width)
        pyplot.bar(i, up, bottom=down+null, color='b', width=width)
    up = 0
    down, null = transitions[-1]
    pyplot.bar(N, down, bottom=0., color='r', width=width)
    pyplot.bar(N, null, bottom=down, color='g', width=width)
    pyplot.bar(N, up, bottom=down+null, color='b', width=width)
    pyplot.ylabel("Transition\n Probabilities")
    # Transition Entropies
    pyplot.subplot(312, sharex = ax1)    
    pyplot.bar(range(0, N+1), transition_entropies, width=width)
    pyplot.ylabel("Transition\n Entropies")
    # Stationary Distribution
    pyplot.subplot(313, sharex = ax1)
    pyplot.bar(range(len(s)), s, width=width)
    pyplot.xlim(0, N+1)
    pyplot.ylabel("Stationary\n Probabilities")
    pyplot.xlabel("Population Size")

# Figures 2, 4, and 8 in text
def figure_2(process=None, incentive="replicator", mutations=None, Ns=range(3, 100), r=1., div_log=False):
    """Plot entropy rate over a range of population sizes and for a small set of mutation rates."""
    m = [[r,r],[1,1]]
    for mu in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.0000001]:#, 1e-8, 1e-12, 1e-20]:
        # Run in parallel and plot
        es = entropy_rate_vs_N(Ns, process, m, incentive="replicator", eta=None, w=None, mutations=mutations, mu_ab=mu, mu_ba=mu, div_log=False, linewidth=1.5)    

# Figure 3 in text
def kl_mutations(Ns, mus, m, process="moran"):
    """Computes the KL-divergence between processes with different mutation regimes."""
    data = []
    for N in Ns:
        for mu in mus:
            (e, s1, d) = compute_entropy_rate(N, process="moran", incentive="replicator", m=m, mutations="boundary", mu_ab=mu, mu_ba=mu, verbose=False, report=True)
            (e, s2, d) = compute_entropy_rate(N, process="moran", incentive="replicator", m=m, mutations="uniform", mu_ab=mu, mu_ba=mu, verbose=False, report=True)            
            c = kl_divergence(s1, s2)
            data.append([N, mu, c])
    heatmap.main(data=data, xindex=1, yindex=0, cindex=-1, xfunc=float, yfunc=int, cfunc=float)
    pyplot.savefig("kl_boundary_uniform.png")

# Helper for figure 6
def max_population(rs, Ns, incentive="replicator", eta=None, w=None, mutations="boundary", mu_ab=0.001, mu_ba=0.001, process="moran"):
    """Calculates the population size N that produces the maximum entropy rate for a landscape [[r,r],[1,1]] by brute force."""
    maximums = []
    for r in rs:
        m = [[r,r],[1,1]]
        # Run in parallel
        variable_parameters = dict(N=Ns)
        constant_parameters = dict(m=m, incentive=incentive, eta=eta, w=w, mutations=mutations, mu_ab=mu_ab, mu_ba=mu_ba, process=process)
        es = run_batches(constant_parameters, variable_parameters)
        
        for i in range(len(es)-1):
            if es[i+1][-1] < es[i][-1]:
                maximums.append((r, i, es[i][-1]))
                if r == 1:
                    print r, i, es[i][-1]
                break
    return maximums

# Figure 6 in text
def figure_6():
    """Plots the value of N for which the maximum entropy rate occurs for various combinations of r and mu."""
    mus = [0.05, 0.01, 0.005, 0.003, 0.001]
    Nmax = [30,105,205,350,1050]
    for i in range(len(mus)):
        mu = mus[i]
        print i, mu
        rs = numpy.arange(0.9,1.1,0.005)
        Ns = range(3, Nmax[i]+1)
        # FYI no maxima for uniform
        results = max_population(rs, Ns, mu_ab=mu, mu_ba=mu, mutations="boundary", process="moran")
        pyplot.plot([x[0] for x in results], [x[1] for x in results], linewidth=1.5)
        pyplot.xticks([0.9, 0.95, 1.00, 1.05, 1.10], [0.9, 0.95, 1.00, 1.05, 1.10])
        pyplot.yticks([10, 100, 200, 400, 600, 800, 1000], [10, 100, 200, 400, 600, 800, 1000])
        pyplot.ylim(-10, 1010)

def generate_all_figures():
    """Generates all figures in the paper."""

    ## Figure 1, example plot of stationary distribution and entropies 
    N = 100
    m = [[2,4],[3,2]]
    distributions_figure(N, m, mu=0.001, k=10, mutations="uniform")
    savefig("figure_0.png")

    ## Figures 2,4, and 8: Entropy Rate vs N for each process.
    for mutations in ["boundary", "uniform"]:
        for process, Ns, div_log in [("moran", range(3, 1000), False), ("wright-fisher", range(3, 500), True), ('n-fold-moran', range(3, 500), True)]:
            print process, mutations
            figure_2(process=process, mutations=mutations, Ns=Ns, r=1., div_log=div_log)
            savefig("%s_neutral_%s.png" % (process, mutations))    
    
    ##Figure 3: KL divergence of mutation regimes figure
    Ns = range(3, 100)
    mus = [0.001 * k for k in range(1, 100)]
    m = [[2,2],[1,1]]
    kl_mutations(Ns, mus, m, process="moran")
    pyplot.ylabel("Population Size")
    pyplot.xlabel("Mutation Rate")
    pyplot.show()

    #Figure 6: Max entropy rate figure
    figure_6()
    pyplot.xlabel("Relative Fitness r")
    pyplot.ylabel("Population Size")
    pyplot.show()
    exit()
    savefig("figure_6.png")
    
    ## Figures 3, 5, and 7
    ## Variable fitness heatmaps 
    mus = [0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.001, 0.0001, 0.00001]
    for mutations in ["boundary", "uniform"]:
        for process, Ns, rs in [("moran", range(3, 151), numpy.arange(0.6,1.4,0.01)), ("wright-fisher", range(3, 101), numpy.arange(0.2,1.8,0.02)), ("n-fold-moran", range(3, 101), numpy.arange(0.2,1.8,0.02))]:
            print process, mutations
            directory = "_".join([process, mutations])
            heatmap_N_r_images(Ns, rs, mus, mutations=mutations, process=process, incentive="replicator", directory=directory)

if __name__ == '__main__':
    generate_all_figures()
    exit()
    


