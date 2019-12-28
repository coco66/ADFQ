"""
Main functions for the ADFQ algorithms

posterior_numeric_exact
posterior_numeric
posterior_adfq
posterior_adfq_v2
"""
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

REW_VAR_0 = 1e-3
DTYPE = np.float64

def posterior_numeric_exact(n_means, n_vars, c_mean, c_var, reward, discount,
        terminal, varTH=1e-10, scale_factor=1.0, num_interval=2000, width=10.0,
        noise=0.0, batch=False, REW_VAR=REW_VAR_0):
    """
    The mean and variance of the true posterior when |A|=2.
    """
    assert(len(n_means) == 2)
    target_means = reward + discount*np.array(n_means, dtype = DTYPE)
    target_vars = discount*discount*np.array(n_vars, dtype = DTYPE)

    bar_vars = 1./(1/c_var + 1./(target_vars + noise))
    bar_means = bar_vars*(c_mean/c_var + target_means/(target_vars+noise))

    cdf1 = norm.cdf((bar_means[0]-target_means[1])/np.sqrt(bar_vars[0]+target_vars[1]))
    cdf2 = norm.cdf((bar_means[1]-target_means[0])/np.sqrt(bar_vars[1]+target_vars[0]))
    pdf1 = norm.pdf((bar_means[0]-target_means[1])/np.sqrt(bar_vars[0]+target_vars[1]))/np.sqrt(bar_vars[0]+target_vars[1])
    pdf2 = norm.pdf((bar_means[1]-target_means[0])/np.sqrt(bar_vars[1]+target_vars[0]))/np.sqrt(bar_vars[1]+target_vars[0])

    c1 = norm.pdf((target_means[0]-c_mean)/np.sqrt(c_var+target_vars[0]))/np.sqrt(c_var+target_vars[0])
    c2 = norm.pdf((target_means[1]-c_mean)/np.sqrt(c_var+target_vars[1]))/np.sqrt(c_var+target_vars[1])

    Z = c1*cdf1 + c2*cdf2
    mean = (c1*(bar_means[0]*cdf1+ bar_vars[0]*pdf1) + c2*(bar_means[1]*cdf2+bar_vars[1]*pdf2)) / Z
    sec_moment = c1*((bar_means[0]**2 + bar_vars[0])*cdf1 +2*bar_means[0]*bar_vars[0]*pdf1 \
                    - (bar_means[0]-target_means[1])/(bar_vars[0]+target_vars[1])*bar_vars[0]**2*pdf1) \
                + c2*((bar_means[1]**2 + bar_vars[1])*cdf2 +2*bar_means[1]*bar_vars[1]*pdf2 \
                    - (bar_means[1]-target_means[0])/(bar_vars[1]+target_vars[0])*bar_vars[1]**2*pdf2)
    var = sec_moment/Z - mean**2
    return mean, var

def posterior_numeric(n_means, n_vars, c_mean, c_var, reward, discount,
        terminal, varTH = 1e-10, scale_factor=1.0, num_interval=2000, width=10.0,
        noise=0.0, noise_c=0.0, batch=False, REW_VAR = REW_VAR_0):
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
    noise = noise/scale_factor
    noise_c = noise_c/scale_factor
    c_var += noise_c
    if batch:
        batch_size = n_means.shape[0]
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_var = np.reshape(c_var, (batch_size,1))
        reward = np.reshape(reward, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))
    else:
        if terminal:
            var_new = 1./(1./c_var + scale_factor/(REW_VAR+noise))
            sd_new = np.sqrt(var_new, dtype=DTYPE)
            mean_new = var_new*(c_mean/c_var + scale_factor*reward/(REW_VAR+noise))
            try:
                x = np.arange(mean_new-0.5*width*sd_new, mean_new+0.5*width*sd_new, width*sd_new/num_interval)
            except:
                import pdb; pdb.set_trace()
                print(mean_new, sd_new)
            return mean_new, var_new, None #(x, norm.pdf(x, mean_new, sd_new))

    target_means = reward + discount*np.array(n_means, dtype = DTYPE)
    target_vars = discount*discount*np.array(n_vars, dtype = DTYPE)

    bar_vars = 1./(1/c_var + 1./(target_vars + noise))
    bar_means = bar_vars*(c_mean/c_var + target_means/(target_vars+noise))
    if (target_vars < 0.0).any() or (bar_vars < 0.0).any():
        pdb.set_trace()
    sd_range = np.sqrt(np.concatenate((target_vars, bar_vars) ,axis=-1), dtype=DTYPE)
    mean_range = np.concatenate((target_means, bar_means),axis=-1)
    x_max = np.max(mean_range + 0.5*width*sd_range, axis=-1)
    x_min = np.min(mean_range - 0.5*width*sd_range, axis=-1)
    interval = (x_max-x_min)/num_interval
    # Need to better way for batch
    if batch :
        mean_new = []
        var_new = []
        for j in range(batch_size):
            m, v, _ = posterior_numeric_helper(target_means[j], target_vars[j], c_mean[j], c_var[j],
                bar_means[j], bar_vars[j], x_max[j], x_min[j], interval[j], noise=noise)
            mean_new.append(m)
            var_new.append(v)

        var_new  = (1.-terminal)*np.reshape(var_new, (batch_size,1)) \
                                + terminal*1./(1./c_var + scale_factor/(REW_VAR+noise))
        mean_new = (1.-terminal)*np.reshape(mean_new, (batch_size,1)) \
                                + terminal*var_new*(c_mean/c_var + scale_factor*reward/(REW_VAR+noise))
        return mean_new, np.maximum(varTH, var_new), None
    else:
        mean_new, var_new, (x, prob) = posterior_numeric_helper(target_means, target_vars, c_mean, c_var,
            bar_means, bar_vars,  x_max, x_min, interval, noise =noise)
        if np.isnan(var_new).any():
            print("Variance is NaN")
        return mean_new, np.maximum(varTH, var_new), (x, prob)

def posterior_numeric_helper(target_means, target_vars, c_mean, c_var, bar_means,
        bar_vars, x_max, x_min, interval, noise=0.0):
    """ADFQ - Numeric posterior update helper function
    """
    anum = target_means.shape[-1]
    add_vars = c_var+target_vars+noise
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
    if (Z== 0.0).any():
        print('All probabilities are 0.0')
        import pdb; pdb.set_trace()

    prob_tot = prob_tot/Z/interval
    x = x[min_id:]
    m = interval*np.inner(x, prob_tot)

    return m, interval*np.inner((x-m)**2, prob_tot), (x, prob_tot)

def posterior_adfq(n_means, n_vars, c_mean, c_var, reward, discount, terminal,
        varTH=1e-20, REW_VAR=REW_VAR_0, scale_factor=1.0, asymptotic=False,
        asymptotic_trigger=1e-20, noise=0.0, noise_c=0.0, batch=False):

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
    noise = noise/scale_factor
    noise_c = noise_c/scale_factor
    c_var += noise_c
    target_vars = discount*discount*np.array(n_vars, dtype=DTYPE)#+ noise
    t = asymptotic_trigger/scale_factor
    if batch:
        batch_size = n_means.shape[0]
        target_means = np.reshape(reward, (batch_size,1)) \
                        + discount*np.array(n_means, dtype=DTYPE)
        stats = posterior_adfq_batch_helper(target_means, target_vars, c_mean,
            c_var, discount, scale_factor=scale_factor, asymptotic=asymptotic,
            noise=noise)
        reward = np.array(reward, dtype=DTYPE)
        terminal = np.array(terminal, dtype=int)
        if asymptotic :
            is_asymptotic = np.prod((n_vars <= t), axis=-1) * np.prod((c_var <= t), axis=-1)
    else:
        target_means = reward + discount*np.array(n_means, dtype=DTYPE)
        sorted_idx = np.flip(np.argsort(target_means), axis=-1)
        stats = posterior_adfq_helper(target_means, target_vars, c_mean, c_var,
            discount, scale_factor=scale_factor, sorted_idx=sorted_idx,
            asymptotic=asymptotic, noise=noise)
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

    var_new  = (1.-terminal)*var_new + terminal*1./(1./c_var + scale_factor/(REW_VAR+noise))
    mean_new = (1.-terminal)*np.squeeze(mean_new) + terminal*var_new*(c_mean/c_var + scale_factor*reward/(REW_VAR+noise))

    if np.isnan(mean_new).any() or np.isinf(mean_new).any():
        print("NaN or Inf value at Mean")
        import pdb; pdb.set_trace()
    if np.isnan(var_new).any() or np.isinf(var_new).any():
        print("NaN or Inf value at Var")
        import pdb; pdb.set_trace()

    if batch:
        return np.squeeze(mean_new), np.maximum(varTH, np.squeeze(var_new)), (np.squeeze(stats[:,:,0]), np.squeeze(stats[:,:,1]), np.squeeze(weights))
    else:
        return mean_new[0], np.maximum(varTH, var_new[0]), (np.squeeze(stats[:,:,0]), np.squeeze(stats[:,:,1]), np.squeeze(weights))


def posterior_adfq_batch_helper(target_means, target_vars, c_mean, c_var,
        discount, scale_factor=1.0, asymptotic=False, noise=0.0):
    """ADFQ - Numeric posterior update helper function for batch update
    """
    batch_stats = []
    sorted_idx = np.flip(np.argsort(target_means), axis=-1)
    for k in range(target_means.shape[0]):
        noise_k = noise[k] if type(noise)== np.ndarray else noise
        stats = posterior_adfq_helper(target_means[k], target_vars[k],c_mean[k],
                    c_var[k], discount, sorted_idx[k], scale_factor, asymptotic,
                    noise=noise_k)
        batch_stats.append(stats)
    return np.array(batch_stats)

def posterior_adfq_helper(target_means, target_vars, c_mean, c_var, discount,
        sorted_idx, scale_factor=1.0, asymptotic=False, noise=0.0):
    """ADFQ - Numeric posterior update helper function
    """
    anum = target_means.shape[-1]

    dis2 = discount*discount
    bar_vars = 1./(1./c_var + 1./(target_vars+noise))
    bar_means = bar_vars*(c_mean/c_var + target_means/(target_vars+noise))
    add_vars = c_var + target_vars + noise
    stats = np.zeros((anum, 3))
    # Search a range for mu_star
    for (j,b) in enumerate(sorted_idx):
        b_primes = [c for c in sorted_idx if c!=b] # anum-1
        outcome = iter_search(anum, b_primes, target_means, target_vars,
            target_means[b], bar_means[b], bar_vars[b], add_vars[b], c_mean,
            discount, asymptotic=asymptotic, scale_factor=scale_factor)
        if outcome is not None:
            stats[b] = outcome
        else:
            print("Could not find a maching mu star")
    return np.array([stats[i] for i in range(anum)], dtype=DTYPE)

# @jit(nopython=True) # Enable For GPU COMPUTATION (from numba import jit)
def iter_search(anum, b_primes, target_means, target_vars, target_means_b,
        bar_means_b, bar_vars_b, add_vars_b, c_mean, discount, asymptotic=False,
        scale_factor=1.0):
    upper = 1e+20
    dis2 = discount*discount
    for i in range(anum):
        n_target_means= np.array([target_means[b_primes[k]] for k in range(i)])
        n_target_vars = np.array([target_vars[b_primes[k]] for k in range(i)])
        if i == (anum-1):
            lower = -1e+20
        else:
            lower = target_means[b_primes[i]]
        mu_star = (np.sum(n_target_means/n_target_vars) + bar_means_b/bar_vars_b)\
                    /(1./bar_vars_b + np.sum(1./n_target_vars))
        if (float(mu_star) >= float(lower)) and (float(mu_star) <= float(upper)):
            var_star = 1./(1./bar_vars_b + np.sum(1./n_target_vars))
            if asymptotic :
                logk = (target_means_b - c_mean)**2 / add_vars_b \
                        + (mu_star - bar_means_b)**2 / bar_vars_b \
                        + np.sum((n_target_means - mu_star)**2 / n_target_vars)
            else:
                logk = 0.5 * (np.log(var_star) - np.log(bar_vars_b) \
                        - np.log(2*np.pi) - np.log(add_vars_b)) \
                    - 0.5/scale_factor * ((target_means_b - c_mean)**2 \
                        / add_vars_b + (mu_star - bar_means_b)**2 / bar_vars_b \
                        + np.sum((n_target_means - mu_star)**2 / n_target_vars ))
            return mu_star, var_star, logk
        else:
            upper = lower

def posterior_adfq_v2(n_means, n_vars, c_mean, c_var, reward, discount,
        terminal, varTH=1e-20, REW_VAR=REW_VAR_0, logEps=-1e+20,
        scale_factor=1.0, asymptotic=False, asymptotic_trigger=1e-20,
        batch=False, noise=0.0):
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
    if batch:
        batch_size = len(n_means)
        c_mean = np.reshape(c_mean, (batch_size,1))
        c_var = np.reshape(c_var, (batch_size,1))
        reward = np.reshape(reward, (batch_size,1))
        terminal = np.reshape(terminal, (batch_size,1))

    target_means = reward +discount*np.array(n_means, dtype = DTYPE)
    target_vars = discount*discount*(np.array(n_vars, dtype = DTYPE))
    bar_vars = 1./(1./c_var + 1./(target_vars + noise))
    bar_means = bar_vars*(c_mean/c_var + target_means/(target_vars + noise))
    add_vars = c_var + target_vars + noise

    sorted_idx = np.argsort(target_means, axis=int(batch))
    if batch:
        ids = range(0,batch_size)
        bar_targets = target_means[ids, sorted_idx[:,-1], np.newaxis]*np.ones(target_means.shape)
        bar_targets[ids, sorted_idx[:,-1]] = target_means[ids, sorted_idx[:,-2]]
    else:
        bar_targets = target_means[sorted_idx[-1]]*np.ones(target_means.shape)
        bar_targets[sorted_idx[-1]] = target_means[sorted_idx[-2]]
    thetas = np.heaviside(bar_targets-bar_means,0.0)
    t = asymptotic_trigger/scale_factor
    if asymptotic :
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
    var_new  = (1.-terminal)*var_new + terminal*1./(1./c_var + scale_factor/(REW_VAR+noise))
    mean_new = (1.-terminal)*mean_new + terminal*var_new*(c_mean/c_var + scale_factor*reward/(REW_VAR+noise))
    if np.isnan(mean_new).any() or np.isnan(var_new).any():
        pdb.set_trace()
    if batch:
        return np.squeeze(mean_new), np.squeeze(np.maximum(varTH, var_new)), (bar_means, bar_vars, weights)
    else:
        return mean_new, np.maximum(varTH, var_new), (bar_means, bar_vars, weights)

def posterior_hypbrid(n_means, n_vars, c_mean, c_var, reward, discount,
        terminal, varTH=1e-20, logEps=-1e+20,  scale_factor=1.0, trigger=1e-4,
        batch=False, noise=0.0):
    """ADFQ posterior update using both ADFQ and ADFQ-V2
    Parameters
    ----------
        trigger : threshold value to start ADFQ-V2 update
    """
    if (c_var*scale_factor <= trigger).all() and (c_var*scale_factor<=trigger).all():
        mean_new, var_new, stats = posterior_adfq_v2(n_means, n_vars, c_mean, c_var, reward, discount, terminal, varTH=varTH,
        logEps=logEps,  scale_factor = scale_factor,  batch=batch, noise=noise)
    else:
        mean_new, var_new, stats = posterior_adfq(n_means, n_vars, c_mean, c_var, reward, discount, terminal, varTH=varTH,
        scale_factor = scale_factor,  batch=batch, noise=noise)

    return mean_new, var_new, stats
