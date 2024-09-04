from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def figsize(scale,t=None):
    fig_width_pt = 426.79135
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    if t == 1:
        return [fig_height,fig_height]
    elif t == 2:
        golden_mean = (np.sqrt(5.0)-1.0)/3.0            # Aesthetic ratio (you could change this)
        fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def apply_settings():
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc('text', usetex=True);
    plt.rc('font', family='serif');
#    plt.rc('legend',frameon=False);
    plt.rc('pgf',texsystem='pdflatex')
#    plt.set_cmap('Pastel1');
#    plt.rc('image', cmap='Pastel1');
    plt.rc('savefig',dpi=300)

def newfig(width,t=None):
    apply_settings();
    plt.clf();
    if t==0:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize(width,t));
    ax = fig.add_subplot(111);
    return fig, ax

def savefig(fig,filename):
    fig.savefig('{}.pgf'.format(filename),bbox_inches='tight')
    fig.savefig('{}.pdf'.format(filename),bbox_inches='tight')


#args = {'cmap':'Pastel1','rot':0}
args = {'cmap':'Set3','rot':0}
