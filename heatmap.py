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

def load_csv(filename):
    with open(filename) as handle:
        reader = csv.reader(handle)
        data = [row for row in reader]
    return data

def prepare_heatmap_data(data, xindex=0, yindex=1, cindex=-1, xfunc=float, yfunc=float, cfunc=float):
    # Grab horizontal and vertical coordinates.
    xs = list(sorted(set([xfunc(z[xindex]) for z in data])))
    ys = list(sorted(set([yfunc(z[yindex]) for z in data])))
    # Prepare to map to a grid.
    x_d = dict(zip(xs, range(len(xs))))
    y_d = dict(zip(ys, range(len(ys))))
    cs = numpy.zeros(shape=(len(ys), len(xs)))
    # Extract relevant data and populate color matrix, mapping to proper indicies.
    for row in data:
        x = xfunc(row[xindex])
        y = yfunc(row[yindex])
        c = cfunc(row[cindex])
        #cs[x_d[x]][y_d[y]] = c
        cs[y_d[y]][x_d[x]] = c
    return xs, ys, cs

def heatmap(xs, ys, cs, cmap=None, sep=10, offset=0.):
    if not cmap:
        cmap = get_cmap()
    plot_obj = pyplot.pcolor(cs, cmap=cmap)
    #plot_obj_2.callbacksSM.connect('changed', ImageFollower(color_obj_2))
    pyplot.colorbar()
    pyplot.xticks([x + offset for x in range(len(xs))][::sep], xs[::sep])
    pyplot.yticks([y + offset for y in range(len(ys))[sep::sep]], ys[sep::sep])

    return plot_obj    
    
def main(data=None, filename=None, xindex=0, yindex=1, cindex=-1, xfunc=float, yfunc=float, cfunc=float):
    if filename:
        data = load_csv(filename)
    if (not filename) and (not data):
        sys.stderr.write('Data or filename is required for heatmap.\n')
    xs, ys, cs = prepare_heatmap_data(data, xindex=xindex, yindex=yindex, cindex=cindex, xfunc=xfunc, yfunc=yfunc, cfunc=cfunc)
    heatmap(xs, ys, cs)
    
if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)
    pyplot.show()
    