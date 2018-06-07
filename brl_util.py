""" 
Author: Heejin Chloe Jeong (chloe.hjeong@gmail.com)
Affiliation: University of Pennsylvania
"""

#import sys
#sys.path.insert(0,'/usr/local/lib/python2.7/site-packages')
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from scipy.misc import logsumexp
import pdb
from operator import itemgetter
from scipy.interpolate import spline
import copy
import time

COLORS = ['g','k','r','b','c','m','y','burlywood','chartreuse','0.8','0.6', '0.4', '0.2']
MARKER = ['-','--', '*-', '+-','d-','o-','x-','s-','2-','3-']
T_chain = 200
T_loop = 10000
T_grid5 = 15000
T_grid10 = 20000
T_minimaze = 30000
T_maze = 40000

EVAL_RUNS = 10
EVAL_NUM = 100
EVAL_STEPS = 50
EVAL_EPS = 0.0
DTYPE = np.float64
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
    #basis = 1/np.sqrt(np.exp((c1-state[0])**2)*np.exp((c2-state[1])**2))
    basis = np.exp(-0.5*(c1-state[0])**2)*np.exp(-0.5*(c2-state[1])**2)
    basis = np.append(basis.flatten(), const)
    phi = np.zeros(3*n, dtype=np.float32)
    phi[action*n:(action+1)*n] = basis

    return phi

def posterior_adf_numeric(n_means, n_vars, c_mean, c_var, reward, discount, terminal, varTH = 1e-20,
        scale_factor=1.0, num_interval=2000, width=10.0, noise=0.0, batch=False, REW_VAR = 1e-5):
    """ADFQ - Numeric posterior update
    Parameters
    ----------
        n_means : a numpy array of mean values for the next state (s'). shape = (anum,) or (batch_size, anum)
        n_vars : a numpy array of variance values for the next state (s'). shape = (anum,) or (batch_size, anum)
        c_mean : a mean value for the current state-action pair. shape = () or (batch_size,)
        c_var : a variance value for the current state-action pair. shape = () or (batch_size,)
        reward : the current observed reward. shape = () or (batch_size,)
        discount : the discount factor. Scalar.
        terminal : (=done) True if the environment is episodic and the current episode is done. shape = () or (batch_size,)
        varTH : variance threshold.
        scale_factor : scale factor.
        num_interval : The number of intervals for the samples (x-values).
        width : determines a range of samples. width*standard_deviation around mean.
        noise : noise added to c_var for the update in stochastic environment
        batch : True if you are using a batch update
        REW_VAR : when terminal == True, TD target = reward. p(r) = Gaussian with mean=0 and variance = REW_VAR.
    """
    c_var = c_var + noise    
    if batch:
        batch_size = n_means.shape[0]
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_var = np.reshape(c_var, (batch_size,1))
        reward = np.reshape(reward, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))
    else:
        if terminal:
            var_new = 1./(1./c_var + scale_factor/REW_VAR)
            sd_new = np.sqrt(var_new, dtype=DTYPE)
            mean_new = var_new*(c_mean/c_var + scale_factor*reward/REW_VAR)
            try:
                x = np.arange(mean_new-0.5*width*sd_new, mean_new+0.5*width*sd_new, width*sd_new/num_interval)
            except:
                print(mean_new, sd_new)
                pdb.set_trace()
            return mean_new, var_new, (x, norm.pdf(x, mean_new, sd_new))

    target_means = reward + discount*np.array(n_means, dtype = DTYPE)
    target_vars = discount*discount*np.array(n_vars, dtype = DTYPE)        
    
    bar_vars = 1./(1/c_var + 1./target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    if (target_vars < 0.0).any() or (bar_vars < 0.0).any():
        pdb.set_trace()
    sd_range = np.sqrt(np.concatenate((target_vars, bar_vars) ,axis=-1), dtype=DTYPE)
    mean_range = np.concatenate((target_means, bar_means),axis=-1)
    x_max = np.max(mean_range + 0.5*width*sd_range, axis=-1)
    x_min = np.min(mean_range - 0.5*width*sd_range, axis=-1)
    interval = (x_max-x_min)/num_interval
    # Need a better way for batch
    if batch : 
        mean_new = []
        var_new = []
        for j in range(batch_size):
            m, v, _ = posterior_numeric_helper(target_means[j], target_vars[j], c_mean[j], c_var[j], 
                bar_means[j], bar_vars[j], x_max[j], x_min[j], interval[j])
            mean_new.append(m)
            var_new.append(v)

        var_new  = (1.-terminal)*np.reshape(var_new, (batch_size,1)) + terminal*1./(1./c_var + scale_factor/REW_VAR)
        mean_new = (1.-terminal)*np.reshape(mean_new, (batch_size,1))+ terminal*var_new*(c_mean/c_var + scale_factor*reward/REW_VAR)
        return mean_new, np.maximum(varTH, var_new), None
    else:
        mean_new, var_new, (x, prob) = posterior_numeric_helper(target_means, target_vars, c_mean, c_var, 
            bar_means, bar_vars,  x_max, x_min, interval)
        if np.isnan(var_new).any():
            print("Variance is NaN")
        return mean_new, np.maximum(varTH, var_new), (x, prob)

def posterior_numeric_helper(target_means, target_vars, c_mean, c_var, bar_means, bar_vars,  x_max, x_min, interval):
    """ADFQ - Numeric posterior update helper function
    """
    anum = target_means.shape[-1]
    add_vars = c_var+target_vars
    x = np.append(np.arange(x_min, x_max, interval), x_max)
    cdfs = np.array([norm.cdf(x, target_means[i], np.sqrt(target_vars[i])) for i in range(anum)])
    nonzero_ids = []
    for cdf in cdfs:
        for (i,v) in enumerate(cdf):
            if v > 0.0:
                nonzero_ids.append(i)
                break
    if len(nonzero_ids) != anum:
        raise ValueError('CDF peak is outside of the range')
    log_probs = []
    min_id = len(x)
    log_max_prob = -1.e+100
    for b in range(anum):
        min_id = min(min_id, nonzero_ids[b])
        idx = max([nonzero_ids[c] for c in range(anum) if c!= b])
        tmp = - np.log(2*np.pi) -0.5*( np.log(add_vars[b]) + np.log(bar_vars[b]) \
                + (c_mean-target_means[b])**2/add_vars[b]+ (x[idx:] - bar_means[b])**2/bar_vars[b]) \
                + np.sum([np.log(cdfs[c, idx:]) for c in range(anum) if c!=b], axis=0)
        log_max_prob = max(log_max_prob, max(tmp))
        log_probs.append(tmp)

    probs = [np.exp(l - log_max_prob) for l in log_probs]
    probs_l = [np.concatenate((np.zeros(len(x)-min_id-len(p),),p)) for p in probs]
    prob_tot = np.sum(np.array(probs_l),axis=0)
    Z = np.sum(prob_tot)
    if ( Z== 0.0).any():
        pdb.set_trace()
        raise ValueError('All probabilities are 0.0')
    prob_tot = prob_tot/Z/interval
    x = x[min_id:]
    m = interval*np.inner(x, prob_tot)

    return m, interval*np.inner((x-m)**2, prob_tot), (x, prob_tot)
    
def posterior_adf(n_means, n_vars, c_mean, c_var, reward, discount, terminal, varTH=1e-20, REW_VAR = 1e-5, 
        scale_factor = 1.0, asymptotic = False, asymptotic_trigger = 1e-20, noise = 0.0, batch=False):
    """ADFQ posterior update
    Parameters
    ----------
        n_means : a numpy array of mean values for the next state (s'). shape = (anum,) or (batch_size, anum)
        n_vars : a numpy array of variance values for the next state (s'). shape = (anum,) or (batch_size, anum)
        c_mean : a mean value for the current state-action pair. shape = () or (batch_size,)
        c_var : a variance value for the current state-action pair. shape = () or (batch_size,)
        reward : the current observed reward. shape = () or (batch_size,)
        discount : the discount factor. Scalar.
        terminal : (=done) True if the environment is episodic and the current episode is done. shape = () or (batch_size,)
        varTH : variance threshold.
        scale_factor : scale factor.
        asymptotic : True to use the asymptotic update 
        asymptotic_trigger : a value to decide when to start the asymptotic update if "asymptotic==True"
        noise : noise added to c_var for the update in stochastic environment
        batch : True if you are using a batch update
        REW_VAR : when terminal == True, TD target = reward. p(r) = Gaussian with mean=0 and variance = REW_VAR.
    """      
    c_var = c_var + noise
    target_vars = discount*discount*np.array(n_vars, dtype=DTYPE)
    t = asymptotic_trigger/scale_factor
    if batch:
        batch_size = n_means.shape[0]
        target_means = np.reshape(reward , (batch_size,1))+ discount*np.array(n_means, dtype=DTYPE)
        stats = posterior_adf_batch_helper(target_means, target_vars, c_mean, c_var, discount,
            scale_factor=scale_factor, asymptotic=asymptotic)
        reward = np.array(reward, dtype=DTYPE)
        terminal = np.array(terminal, dtype=int)
        if asymptotic :
            is_asymptotic = np.prod((n_vars <= t), axis=-1) * np.prod((c_var <= t), axis=-1)
    else:
        target_means = reward + discount*np.array(n_means, dtype=DTYPE)
        stats = posterior_adf_helper(target_means, target_vars, c_mean, c_var, discount,
            scale_factor=scale_factor, asymptotic=asymptotic)
        stats = stats[np.newaxis,:]
        if asymptotic and (n_vars <= t).all() and (c_var <= t):
            b_rep = np.argmin(stats[:,:,2], axis=-1)
            weights = np.zeros((len(stats),))
            weights[b_rep] = 1.0
            return stats[b_rep, 0], np.maximum(stats[b_rep,1], varTH), (stats[:,0], stats[:,1], weights)
    
    logk = stats[:,:,2] - np.max(stats[:,:,2],axis=-1, keepdims=batch)
    weights = np.exp(logk - logsumexp(logk, axis=-1, keepdims=batch), dtype=DTYPE)
    v = weights*stats[:,:,0]
    mean_new = np.sum(v, axis=-1, keepdims=batch)
    var_new = np.sum(weights*stats[:,:,1], axis=-1) \
                + np.sum(v*(stats[:,:,0] - mean_new),axis=-1)/scale_factor
    var_new  = (1.-terminal)*var_new + terminal*1./(1./c_var + scale_factor/REW_VAR)
    mean_new = (1.-terminal)*np.squeeze(mean_new) + terminal*var_new*(c_mean/c_var + scale_factor*reward/REW_VAR)
    if np.isnan(mean_new).any() or np.isnan(var_new).any():
        pdb.set_trace()
    if batch:
        return np.squeeze(mean_new), np.maximum(varTH, np.squeeze(var_new)), (np.squeeze(stats[:,:,0]), np.squeeze(stats[:,:,1]), np.squeeze(weights))
    else:
        return mean_new[0], np.maximum(varTH, var_new[0]), (np.squeeze(stats[:,:,0]), np.squeeze(stats[:,:,1]), np.squeeze(weights))

def posterior_adf_batch_helper(target_means, target_vars, c_mean, c_var, discount, scale_factor, asymptotic=False):
    batch_stats = []
    for k in range(target_means.shape[0]):
        stats = posterior_adf_helper(target_means[k], target_vars[k], c_mean[k], c_var[k], discount, scale_factor, asymptotic=False)
        batch_stats.append(stats)
    return np.array(batch_stats)

def posterior_adf_helper(target_means, target_vars, c_mean, c_var, discount, scale_factor, asymptotic=False):
    anum = target_means.shape[-1]

    dis2 = discount*discount
    rho_vars = c_var/target_vars*dis2 
    sorted_idx = np.flip(np.argsort(target_means), axis=-1)
    bar_vars = 1./(1./c_var + 1./target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    add_vars = c_var + target_vars
    stats = {}
    # Search a range for mu_star
    for (j,b) in enumerate(sorted_idx):
        b_primes = [c for c in sorted_idx if c!=b]
        upper = 1e+20
        vals = []
        for i in range(anum):
            lower = -1e+20 if i==(anum-1) else target_means[b_primes[i]]
            mu_star = np.float64( (bar_means[b] + sum(target_means[b_primes[:i]]*rho_vars[b_primes[:i]])/(dis2+rho_vars[b]))\
                /(1+sum(rho_vars[b_primes[:i]])/(dis2+rho_vars[b])))
            vals.append((lower,mu_star,upper))
            if (np.float16(mu_star) >= np.float16(lower)) and (np.float16(mu_star) <= np.float16(upper)):
                var_star = 1./(1./bar_vars[b] + sum(1./target_vars[b_primes[:i]]))
                if asymptotic : 
                    logk = (target_means[b]-c_mean)**2/add_vars[b] + (mu_star-bar_means[b])**2/bar_vars[b] \
                        + sum( (target_means[b_primes[:i]]-mu_star)**2/target_vars[b_primes[:i]] )
                else:
                    logk = 0.5*(np.log(var_star) - np.log(bar_vars[b]) - np.log(2*np.pi) - np.log(add_vars[b])) \
                        - 0.5/scale_factor*( (target_means[b]-c_mean)**2/add_vars[b] + (mu_star-bar_means[b])**2/bar_vars[b] \
                            + sum( (target_means[b_primes[:i]]-mu_star)**2/target_vars[b_primes[:i]] ))
                stats[b]= (mu_star, var_star, logk)
                break
            upper = lower
        if len(stats) <= j:
            print("Could not find a maching mu star")
            pdb.set_trace()
    return np.array([stats[i] for i in range(anum)], dtype=DTYPE)

def posterior_adf_v2(n_means, n_vars, c_mean, c_var, reward, discount, terminal, varTH=1e-20, REW_VAR = 1e-5, 
        logEps=-1e+20,  scale_factor = 1.0, asymptotic=False, asymptotic_trigger = 1e-20, batch=False, noise=0.0):
    """ADFQ posterior update version 2 (presented in appendix of the ADFQ paper)
    Parameters
    ----------
        n_means : a numpy array of mean values for the next state (s'). shape = (anum,) or (batch_size, anum)
        n_vars : a numpy array of variance values for the next state (s'). shape = (anum,) or (batch_size, anum)
        c_mean : a mean value for the current state-action pair. shape = () or (batch_size,)
        c_var : a variance value for the current state-action pair. shape = () or (batch_size,)
        reward : the current observed reward. shape = () or (batch_size,)
        discount : the discount factor. Scalar.
        terminal : (=done) True if the environment is episodic and the current episode is done. shape = () or (batch_size,)
        varTH : variance threshold.
        logEps : log of the small probability used in the approximation (Eq.16)
        scale_factor : scale factor.
        asymptotic : True to use the asymptotic update 
        asymptotic_trigger : a value to decide when to start the asymptotic update if "asymptotic==True"
        noise : noise added to c_var for the update in stochastic environment
        batch : True if you are using a batch update
        REW_VAR : when terminal == True, TD target = reward. p(r) = Gaussian with mean=0 and variance = REW_VAR.
    """   
    c_var = c_var + noise
    if batch:
        batch_size = len(n_means)
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_var = np.reshape(c_var, (batch_size,1))
        reward = np.reshape(reward, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))

    target_means = reward +discount*np.array(n_means, dtype = DTYPE)
    target_vars = discount*discount*np.array(n_vars, dtype = DTYPE)
    bar_vars = 1./(1./c_var + 1./target_vars)
    bar_means = bar_vars*(c_mean/c_var + target_means/target_vars)
    add_vars = c_var + target_vars

    sorted_idx = np.argsort(target_means, axis=int(batch))
    if batch:
        ids = range(0,batch_size)
        bar_targets = target_means[ids, sorted_idx[:,-1], np.newaxis]*np.ones(target_means.shape)
        bar_targets[ids, sorted_idx[:,-1]] = target_means[ids, sorted_idx[:,-2]]
    else:
        bar_targets = target_means[sorted_idx[-1]]*np.ones(target_means.shape)
        bar_targets[sorted_idx[-1]] = target_means[sorted_idx[-2]]
    thetas = np.heaviside(bar_targets-bar_means,0.0)  
    if asymptotic :
        t = asymptotic_trigger/scale_factor
        if (n_vars <= t).all() and (c_var <= t):
            b_rep = np.argmin((target_means-c_mean)**2 - 2*add_vars*logEps*thetas) # logEps*thetas = log(nu)
            weights = np.zeros(np.shape(target_means))
            weights[b_rep] = 1.0
            return bar_means[b_rep], np.maximum(varTH, bar_vars[b_rep]), (bar_means, bar_vars, weights)

    log_weights = -0.5*( np.log(np.pi*2) + np.log(add_vars) + (c_mean-target_means)**2/add_vars/scale_factor) + logEps*thetas
    log_weights = log_weights - np.max(log_weights, axis=int(batch), keepdims=batch)
    log_weights = log_weights - logsumexp(log_weights, axis=int(batch), keepdims=batch)
    weights = np.exp(log_weights, dtype=DTYPE)

    mean_new = np.sum(np.multiply(weights, bar_means), axis=int(batch), keepdims=batch)
    var_new = (np.sum(np.multiply(weights, bar_means**2), axis=int(batch), keepdims=batch) - mean_new**2)/scale_factor \
                + np.sum(np.multiply(weights, bar_vars), axis=int(batch), keepdims=batch)
    # For Terminals
    var_new  = (1.-terminal)*var_new + terminal*1./(1./c_var + scale_factor/REW_VAR)
    mean_new = (1.-terminal)*mean_new + terminal*var_new*(c_mean/c_var + scale_factor*reward/REW_VAR)
    if np.isnan(mean_new).any() or np.isnan(var_new).any():
        pdb.set_trace()
    if batch:
        return np.squeeze(mean_new), np.squeeze(np.maximum(varTH, var_new)), (bar_means, bar_vars, weights)
    else:
        return mean_new, np.maximum(varTH, var_new), (bar_means, bar_vars, weights)

def posterior_hybrid(n_means, n_vars, c_mean, c_var, reward, discount, terminal, varTH=1e-10, 
        logEps=-1e+20,  scale_factor = 1.0, trigger = 1e-4, batch=False, noise=0.0):
    """ADFQ posterior update using both ADFQ and ADFQ-V2
    Parameters
    ----------
        trigger : threshold value to start ADFQ-V2 update
    """
    if (c_var*scale_factor <= trigger).all() and (c_var*scale_factor<=trigger).all():
        mean_new, var_new, stats = posterior_adf_v2(n_means, n_vars, c_mean, c_var, reward, discount, terminal, 
            varTH=varTH, logEps=logEps, scale_factor = scale_factor,  batch=batch, noise=noise)
    else:
        mean_new, var_new, stats = posterior_adf(n_means, n_vars, c_mean, c_var, reward, discount, terminal, 
            varTH=varTH,  scale_factor = scale_factor,  batch=batch, noise=noise)

    return mean_new, var_new, stats

def plot_IQR(T, data, labels, x_label, y_label, x_vals = None, save=False, shadow = True, legend=(True, (1,1)), pic_name = None):
    import matplotlib.pyplot as plt
    """Plot with interquartile
        T : True finite-time horizon
        data : data to plot
        labels : labels to display in legend
        x_vals : x values. If not given (None), it is determined with T.
        save : True to save the plot rather than display
        shadow : plot shadow
        legend : Tuple with - legend[0] is True if you want to display a legend. legend[1] = a tuple for anchor location.
        pic_name : a name of an image file of the plot
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if not(colors):
        colors = ['g','k','c','b','m','r','y','burlywood','chartreuse','0.8','--', '-.', ':']
    plot_err = []
    f1, ax1 = plt.subplots()
    if len(data.shape)==2:
        N = data.shape[1]+1
        if x_vals == None :
            x_vals = range(0,T-1,T/N)
        xnew = np.linspace(x_vals[0],x_vals[-1],300) #300 represents number of points to make between T.min and T.max
        for (i,y) in enumerate(data):
            power_smooth = spline(x_vals,[0]+list(y),xnew)
            power_smooth = [max(0,el) for el in power_smooth]
            tmp, = ax1.plot(xnew, power_smooth, colors[i], label=labels[i], linewidth=2.0)
            plot_err.append(tmp)
    else:
        if x_vals == None:  
            x_vals = range(0,T) 
        for (i,y) in enumerate(data):
            m, ids25, ids75  = iqr(y)
            tmp, = ax1.plot(x_vals, m, MARKER[i+2], color=colors[i], markeredgecolor = colors[i], markerfacecolor='None', label=labels[i], linewidth=2.0)
            plot_err.append(tmp) 
            if shadow:       
                ax1.fill_between(x_vals, list(ids75), list(ids25), facecolor=colors[i], alpha=0.15)

    if legend[0]:
        ax1.legend(plot_err, labels ,loc='upper right', shadow=True,
                    prop={'family':'Times New Roman', 'size':16})#, bbox_to_anchor=legend[1])
    ax1.tick_params(axis='both',which='major',labelsize=15)
    ax1.set_xlabel(x_label,fontsize=16, fontname="Times New Roman")
    ax1.set_ylabel(y_label,fontsize=16, fontname="Times New Roman")
    ax1.grid()
    
    if save:
        f1.savefig(pic_name,  bbox_inches='tight', 
              pad_inches=0)
    else:
        plt.show()

def plot_smoothing(t, y, labels, x_label='x', y_label='y', title=None, save=False, pic_name=None, 
        legend=(True,(0.8,0.5)), window=4):
    import matplotlib.pyplot as plt
    """Plot with interquartile
        T : True finite-time horizon
        data : data to plot
        labels : labels to display in legend
        save : True to save the plot rather than display
        legend : Tuple with - legend[0] is True if you want to display a legend. legend[1] = a tuple for anchor location.
        pic_name : a name of an image file of the plot
        window : smoothing window size
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if not(colors):
        colors = ['g','k','c','b','m','r','y','burlywood','chartreuse','0.8','--', '-.', ':']
    x_vals = range(0, t, int(t/(y.shape[-1])))
    f, ax = plt.subplots()
    if len(y.shape)==3:
        y = np.mean(y, axis=1)
    plot_list = []
    for i in range(y.shape[0]):
        if MARKER[i] in [':','-','--']:
            lw = 1.5
        else:
            lw = 1.5
        tmp, = ax.plot(x_vals, smoothing(y[i], window),  MARKER[i], linewidth=lw, color = colors[i], markeredgecolor = colors[i], markerfacecolor='None')
        plot_list.append(tmp)
    pdb.set_trace()
    for i in range(y.shape[0]):
        tmp, = ax.plot(x_vals, y[i], colors[i], alpha = 0.2)
    if legend[0]:
        ax.legend(plot_list, labels ,loc='upper left', shadow=True,
                    prop={'family':'Times New Roman', 'size':12})#, bbox_to_anchor=legend[1])
    ax.tick_params(axis='both',which='major',labelsize=12)
    ax.set_xlabel(x_label,fontsize=15, fontname="Times New Roman")
    ax.set_ylabel(y_label,fontsize=15, fontname="Times New Roman")
    ax.set_xlim((0,x_vals[-1]))
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
    i25 = int(0.25*x.shape[0])
    i75 = int(0.75*x.shape[0])
    x=x.T
    ids25=[]
    ids75=[]
    m = []
    for y in x:
        tmp = np.sort(y)
        ids25.append(tmp[i25])
        ids75.append(tmp[i75])
        m.append(np.mean(tmp,dtype=np.float32))
    return m, ids25, ids75

def smoothing(x, window):
    """Smoothing
    Parameters
    ----------
        x : an array of data
        window : smoothing window size
    """
    interval = int(np.floor(window*0.5))
    smoothed = list(x[:interval])
    for i in range(len(x)-window):
        smoothed.append(np.mean(x[i:i+window+1]))
    smoothed.extend(x[-interval:])
    return smoothed

def value_plot(Q_tab, env, isMaze = True, arrow = True):
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
        idx2cell = env.idx2cell
        for i in range(8):
            f,ax = plt.subplots()
            y_mat = np.zeros(env.dim)
            for j in range(len(idx2cell)):
                pos = idx2cell[j]
                y_mat[pos[0], pos[1]] = V[8*j+i]
                if arrow:
                    a = best_action[8*j+i]
                    ax.arrow(pos[1], pos[0], direction[a][0], direction[a][1], 
                        head_width=0.05, head_length=0.1, fc='r', ec='r')
            y_mat[env.goal_pos] = max(V)+0.1
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
        tab[env.goal_pos] = max(V[:-1])+0.1
        plt.imshow(tab,cmap='gray') 
    plt.show()

