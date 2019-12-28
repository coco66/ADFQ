"""
Util functions
"""
import numpy as np
from scipy.stats import norm

COLORS = ['g','k','r','b','c','m','y','burlywood','chartreuse','0.8','0.6', '0.4', '0.2']
#MARKER = ['-','x-', '-.','+-','*-','d-','o-','x-','s-','2-','3-']
MARKER = ['*-', '+-', 'o-', '-.']
T_chain = 5000
T_loop = 5000
T_grid5 = 10000
T_grid10 = 20000
T_minimaze = 30000
T_maze = 40000
T_movingmaze = 200

DTYPE = np.float64

EVAL_RUNS = 10
EVAL_NUM = 100
EVAL_STEPS = 50
EVAL_EPS = 0.05
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def discrete_phi(state, action, dim, anum):
    """Linear function approximation for tabular case
    """
    phi = np.zeros(dim, dtype=np.float)
    phi[state*anum+action] = 1.0
    return phi

def img_preprocess(org_img):
    """Atari games image preprocessing similar to DQN paper
    """
    imgGray = cv2.cvtColor( org_img, cv2.COLOR_RGB2GRAY )
    resizedImg = cv2.resize(np.reshape(imgGray, org_img.shape[:-1]), (84, 110))
    cropped = resizedImg[18:102,:]
    cropped = cropped.astype(np.float32)
    cropped *= (1.0/255.0)
    return cropped

def rbf(state, action, dim, const=1.0):
    """Radial Basis Function used in KTD paper (https://www.jair.org/index.php/jair/article/view/10675/25513)
    """
    n = dim
    c1 = np.reshape(np.array([-np.pi/4.0, 0.0, np.pi/4.0]),(3,1)) # For inverted pendulum
    c2 = np.reshape(np.array([-1.0,0.0,1.0]), (1,3)) # For inverted pendulum
    basis = np.exp(-0.5*(c1-state[0])**2)*np.exp(-0.5*(c2-state[1])**2)
    basis = np.append(basis.flatten(), const)
    phi = np.zeros(3*n, dtype=np.float32)
    phi[action*n:(action+1)*n] = basis

    return phi


def plot_IQR(T, data, labels, x_label='x', y_label='y', x_vals = None, title=None, save=False, legend=(True, 'upper right'),
    shadow=True, pic_name = None, colors=None, smoothed=False):
    import matplotlib.pyplot as plt
    """Plot with interquartile
        T : True finite-time horizon
        data : data to plot
        labels : labels to display in legend
        x_label : x-axis label
        y_label : y-axis label
        x_vals : x values. If not given (None), it is determined with T.
        title : title name to plot
        save : True to save the plot rather than display
        shadow : fill between 25% and 75%
        legend : Tuple with - legend[0] is True if you want to display a legend. legend[1] = a tuple for anchor location.
        pic_name : a name of an image file of the plot
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plot_err = []
    f, ax = plt.subplots()
    N = data.shape[-1]
    if len(data.shape) == 2:
        data = data[np.newaxis,:]
    if x_vals == None:
        x_vals = range(0,T, int(T/N))
        if N != x_vals:
            x_vals = x_vals[:(N-len(x_vals))]
    for (i,y) in enumerate(data):
        if smoothed:
            tmp_y = []
            for yi in y:
                tmp_y.append(smoothing(yi,4))
            y = np.array(tmp_y)

        m, ids25, ids75  = iqr(y)
        tmp, = ax.plot(x_vals, m, MARKER[i+2], color=colors[i], markeredgecolor = colors[i], markerfacecolor='None', label=labels[i], linewidth=2.0)
        plot_err.append(tmp)
        if shadow:
            ax.fill_between(x_vals, list(ids75), list(ids25), facecolor=colors[i], alpha=0.3)

    if legend[0]:
        ax.legend(plot_err, labels ,loc=legend[1], shadow=False, fancybox=True, framealpha=0.5,
                    prop={'family':'Times New Roman', 'size':16})#, bbox_to_anchor=legend[1])
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.set_xlabel(x_label,fontsize=14, fontname="Times New Roman")
    ax.set_ylabel(y_label,fontsize=14, fontname="Times New Roman")
    ax.grid()
    if title:
        ax.set_title(title,fontsize=15,fontname="Times New Roman")
    if save:
        f.savefig(pic_name,  bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        return f, ax

def plot_sd(T, data, labels, x_label='x', y_label='y', x_vals = None, title=None, save=False, legend=(True, 'upper right', (1.0, 1.0)),
    shadow=True, pic_name = None, colors=None, smoothed=False, figure=None):
    import matplotlib.pyplot as plt
    """Plot with interquartile
        T : True finite-time horizon
        data : data to plot
        labels : labels to display in legend
        x_label : x-axis label
        y_label : y-axis label
        x_vals : x values. If not given (None), it is determined with T.
        title : title name to plot
        save : True to save the plot rather than display
        shadow : fill between 25% and 75%
        legend : Tuple with - legend[0] is True if you want to display a legend. legend[1] = a tuple for anchor location.
        pic_name : a name of an image file of the plot
    """
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    #if not(colors):
    #    colors = ['g','k','c','b','m','r','y','burlywood','chartreuse','0.8','--', '-.', ':']
    if figure:
        f, ax, plegend = figure
    else:
        f, ax = plt.subplots()
        plegend = []
    N = data.shape[-1]
    if len(data.shape) == 2:
        data = data[np.newaxis,:]
    if x_vals is None:
        x_vals = range(0,T, int(T/N))
        if N != len(x_vals):
            x_vals = x_vals[:(N-len(x_vals))]
    for (i,y) in enumerate(data):
        if smoothed:
            tmp_y = []
            for yi in y:
                tmp_y.append(smoothing(yi,4))
            y = np.array(tmp_y)
        m, interval_l, interval_h = mstd(y)
        tmp, = ax.plot(x_vals, m, MARKER[i], color=colors[i], markersize = 7,
                        markeredgecolor = colors[i], markerfacecolor='None', label=labels[i],
                        linewidth=2.0)
        plegend.append(tmp)
        if shadow:
            ax.fill_between(x_vals, interval_h, interval_l, facecolor=colors[i], alpha=0.3)

    if legend[0]:
        ax.legend(plegend, labels ,loc=legend[1], shadow=False, fancybox=True, framealpha=0.5,
                    prop={'family':'Times New Roman', 'size':20}, bbox_to_anchor=legend[2])
    ax.tick_params(axis='both',which='major',labelsize=14)
    ax.set_xlabel(x_label,fontsize=18, fontname="Times New Roman")
    ax.set_ylabel(y_label,fontsize=18, fontname="Times New Roman")
    ax.grid()
    if title:
        ax.set_title(title,fontsize=20,fontname="Times New Roman")
    if save:
        f.savefig(pic_name,  bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
        return f, ax

def plot_smoothing(T, data, labels, x_label='x', y_label='y', title=None, save=False, legend=(True,(1.0,0.5)), pic_name=None, window=4, colors=None):
    import matplotlib.pyplot as plt
    """Plot with smoothed by a moving average
        T : True finite-time horizon
        data : data to plot
        labels : labels to display in legend
        x_label : x-axis label
        y_label : y-axis label
        x_vals : x values. If not given (None), it is determined with T.
        title : title name to plot
        save : True to save the plot rather than display
        legend : Tuple with - legend[0] is True if you want to display a legend. legend[1] = a tuple for anchor location.
        pic_name : a name of an image file of the plot
        window : moving average window size
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if not(colors):
        colors = COLORS
    ts = range(0, T, int(T/(data.shape[-1])))
    f, ax = plt.subplots()
    if len(data.shape)==3:
        data = np.mean(data, axis=1)
    plot_list = []
    for i in range(data.shape[0]):
        if MARKER[i] in [':','-','--']:
            lw = 1.5
        else:
            lw = 1.5

        tmp, =ax.plot(ts, smoothing(data[i], window),  MARKER[i], linewidth=lw, color = colors[i], markeredgecolor = colors[i], markerfacecolor='None')
        plot_list.append(tmp)
    for i in range(data.shape[0]):
        tmp, = ax.plot(ts, data[i], colors[i], alpha = 0.2)
    if legend[0]:
        ax.legend(plot_list, labels ,loc=legend[1], shadow=True,
                    prop={'family':'Times New Roman', 'size':13})#, bbox_to_anchor=legend[1])
    ax.tick_params(axis='both',which='major',labelsize=13)
    ax.set_xlabel(x_label,fontsize=13, fontname="Times New Roman")
    ax.set_ylabel(y_label,fontsize=13, fontname="Times New Roman")
    ax.set_xlim((0,ts[-1]))
    if title:
        ax.set_title(title,fontsize=15,fontname="Times New Roman")
    if save:
        f.savefig(pic_name,  bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def iqr(x):
    """Interquantiles
    x has to be a 2D np array. The interquantiles are computed along with the axis 1
    """
    x=x.T
    ids25=[]
    ids75=[]
    m = []
    for y in x:
        ids25.append(np.percentile(y, 25))
        ids75.append(np.percentile(y, 75))
        m.append(np.median(y))
    return m, ids25, ids75

def mstd(x):
    m = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return m, m-0.5*std, m+0.5*std


def smoothing(x, window):
    """Smoothing by a moving average
    Parameters
    ----------
        x : an array of data
        window : smoothing window size
    """
    if window == 0:
        return x
    interval = int(np.floor(window*0.5))
    smoothed = list(x[:interval])
    for i in range(len(x)-window):
        smoothed.append(np.mean(x[i:i+window+1]))
    smoothed.extend(x[-interval:])
    return smoothed

def value_plot(Q_tab, obj, isMaze = True, arrow = True):
    import matplotlib.pyplot as plt
    """Display value and policy in a grid/maze domain
    Parameters
    ----------
    Q_tab : tabular Q values
    env : environment object
    isMaze : True if env is a maze.
    arrow : True if you want to present policy arrows.
    """
    direction={0:(0,-0.4),1:(0,0.4),2:(-0.4,0),3:(0.4,0)} #(x,y) cooridnate
    V = np.max(Q_tab,axis=1)
    best_action = np.argmax(Q_tab,axis=1)
    if isMaze:
        idx2cell = obj.idx2cell
        for i in range(8):
            f,ax = plt.subplots()
            y_mat = np.zeros(obj.dim)
            for j in range(len(idx2cell)):
                pos = idx2cell[j]
                y_mat[pos[0], pos[1]] = V[8*j+i]
                if arrow:
                    a = best_action[8*j+i]
                    ax.arrow(pos[1], pos[0], direction[a][0], direction[a][1],
                        head_width=0.05, head_length=0.1, fc='r', ec='r')
            y_mat[obj.goal_pos] = max(V)+0.1
            ax.imshow(y_mat,cmap='gray')
    else:
        n = int(np.sqrt(len(V)))
        tab = np.zeros((n,n))
        for r in range(n):
            for c in range(n):
                if not(r==(n-1)and c==(n-1)):
                    tab[r,c] = V[n*c+r]
                    if arrow:
                        d = direction[best_action[n*c+r]]
                        plt.arrow(c,r,d[0],d[1], head_width=0.05, head_length=0.1, fc='r', ec='r')
        tab[obj.goal_pos] = max(V[:-1])+0.1
        plt.imshow(tab,cmap='gray')
    plt.show()

def max_Gaussian(mean_vector, sd_vector):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    N = len(mean_vector)
    min_mean_id = np.argmin(mean_vector)
    max_mean_id = np.argmax(mean_vector)
    x = np.arange(mean_vector[min_mean_id]-5*sd_vector[min_mean_id],
            mean_vector[max_mean_id]+5*sd_vector[max_mean_id], 0.01)
    cdfs = [norm.cdf(x, mean_vector[i], sd_vector[i]) for i in range(N)]
    pdfs = [norm.pdf(x, mean_vector[i], sd_vector[i]) for i in range(N)]
    term_in_sum = []
    for i in range(N):
        prod_cdfs = np.prod([cdfs[j] for j in range(N) if j!=i], axis=0)
        term_in_sum.append(pdfs[i] * prod_cdfs)
        plt.plot(x, norm.pdf(x, mean_vector[i], sd_vector[i]))
    prob = np.sum(term_in_sum, axis=0)
    prob = prob/ (np.sum(prob)*0.01)
    max_mean = np.sum(x*prob*0.01)
    plt.plot(x, prob)
    plt.axvline(x=max_mean, color='k')
    plt.axvline(x=mean_vector[max_mean_id], color='k', linestyle = '--')
    labels = [str(i) for i in range(N)]
    labels.extend(['max', 'max of mean', 'mean of max'])
    plt.legend(labels)
    plt.show()
