import csv
import sys
import numpy

from matplotlib import pyplot


def get_cmap(cmap_name=None):
    if not cmap_name:
        cmap = pyplot.get_cmap('jet')
    else:
        cmap = pyplot.get_cmap(cmap_name)
    return cmap

### http://matplotlib.sourceforge.net/examples/pylab_examples/multi_image.html    
##class ImageFollower:
    ##'update image in response to changes in clim or cmap on another image'
    ##def __init__(self, follower):
        ##self.follower = follower
    ##def __call__(self, leader):
        ##self.follower.set_cmap(leader.get_cmap())
        ##self.follower.set_clim(leader.get_clim())       
        
def heatmap(rs, Ns, C, cmap=None):
    if not cmap:
        cmap = get_cmap()
    plot_obj = pyplot.pcolor(C, cmap=cmap)
    #plot_obj_2.callbacksSM.connect('changed', ImageFollower(color_obj_2))
    pyplot.colorbar()
    pyplot.yticks([x + 0.5 for x in range(len(rs))][::5], rs[::5])
    pyplot.xticks([x + 0.5 for x in range(len(Ns))[::5]], Ns[::5])
    return plot_obj
        
#row = [r, N, state, rr_mu, rr_s, rr_count, inf_mu, inf_mu_s, inf_mode, inf_mode_s]
def prepare_heatmap_data(filename):
    handle = open(filename)
    reader = csv.reader(handle)
    data = [row for row in reader]
    # Grab horizontal and vertical coordinates.
    rs = list(sorted(set([float(x[0]) for x in data])))
    Ns = list(sorted(set([float(x[1]) for x in data])))
    # Prepare to map to a grid.
    r_d = dict(zip(rs, range(len(rs))))
    N_d = dict(zip(Ns, range(len(Ns))))
    C = numpy.zeros(shape=(len(rs), len(Ns)))
    # Extract relaveent data and populate color matrix.
    for row in data:
        r = float(row[0])
        N = float(row[1])
        v = float(row[-1])
        C[r_d[r]][N_d[N]] = v
    return rs, Ns, C
    
def main(filename):
    rs, Ns, C = prepare_heatmap_data(filename)
    heatmap(rs, Ns, C)
    
if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)
    pyplot.show()    
   