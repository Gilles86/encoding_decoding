import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

# Settings for all code experiments. setting paradigm
contextual_prior = "uniform"
scaling = 1.7 # THIS IS if scaled_prior mapping is used.

encoding_prior_used = "wei" #"steep", #wei

experimentRange = "00to180" #"00to180" # "45to225"
max_val = 42
min_val = 2


# Fixed stuff
factor_val = max_val - min_val

if experimentRange == "00to45" or experimentRange == "45to90" or experimentRange == "00to90":
    end = int(experimentRange[-2:])
    start = int(experimentRange[0:2])
if experimentRange == "90to135" or experimentRange == "90to180":
    end = int(experimentRange[-3:])
    start = int(experimentRange[0:2])
if experimentRange == "135to180":
    end = int(experimentRange[-3:])
    start = int(experimentRange[0:3])
if experimentRange == "00to180":
    end = 180
    start = 0
if experimentRange == "45to225":
    end = 225
    start = 45

factor_ori = (end-start)/90.*np.pi
displace = start/90.*np.pi

stim_ori_grid = np.linspace(start, end, 501) * np.pi / 90
rep_ori_grid = np.linspace(start, end, 501) * np.pi / 90

rep_val_grid = np.linspace(min_val, max_val, 501)

addedFactor = max_val
if encoding_prior_used == "steep":
    steepnessFactor = addedFactor*0.95 #addedFactor/factor_val*factor_val
if encoding_prior_used == "wei": # generates prior of 2-abs(sin(x))
    steepnessFactor = addedFactor*0.5

# Getting a value function given a grid of orientations (stim_ori_grid for example). Always between 0 and 2pi.
# As assumed in analytical solutions, G = G' mapping
def value_function_ori(x, type, line_frac = 0):
    x = np.array(x)
    line = abs(stim_ori_grid/np.pi/2.*factor_val - max_val)*line_frac
    
    if type == "cdf_prior":
        value_function = np.zeros_like(x)
        value_function[x > 2*np.pi] = (((steepnessFactor)*np.cos(x)[x > 2*np.pi]+(addedFactor)*(x-np.pi)[x > 2*np.pi])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val #integrate.cumtrapz(prior_ori(x), stim_ori_grid, initial=0.0)
        value_function[x <= np.pi] = (((steepnessFactor)*np.cos(x)[x <= np.pi]+(addedFactor)*(x)[x <= np.pi])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val-displace/(2*np.pi)*factor_val 
        value_function[(x > np.pi) & (x <= 2*np.pi)] = (((steepnessFactor)*-np.cos(x)[(x > np.pi) & (x <= 2*np.pi)]-2*steepnessFactor+(addedFactor)*(x)[(x > np.pi) & (x <= 2*np.pi)])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val-displace/(2*np.pi)*factor_val

    if type == "scaled_cdf":
        value_function = np.zeros_like(x)
        value_function[x > 2*np.pi] = (((steepnessFactor*scaling)*np.cos(x)[x > 2*np.pi]+(addedFactor)*(x-np.pi)[x > 2*np.pi])/(2*np.pi*addedFactor-4*steepnessFactor*scaling) - steepnessFactor*scaling/(2*np.pi*addedFactor-4*steepnessFactor*scaling))*factor_val+min_val #integrate.cumtrapz(prior_ori(x), stim_ori_grid, initial=0.0)
        value_function[x <= np.pi] = (((steepnessFactor*scaling)*np.cos(x)[x <= np.pi]+(addedFactor)*(x)[x <= np.pi])/(2*np.pi*addedFactor-4*steepnessFactor*scaling) - steepnessFactor*scaling/(2*np.pi*addedFactor-4*steepnessFactor*scaling))*factor_val+min_val-displace/(2*np.pi)*factor_val 
        value_function[(x > np.pi) & (x <= 2*np.pi)] = (((steepnessFactor*scaling)*-np.cos(x)[(x > np.pi) & (x <= 2*np.pi)]-2*scaling*steepnessFactor+(addedFactor)*(x)[(x > np.pi) & (x <= 2*np.pi)])/(2*np.pi*addedFactor-4*steepnessFactor*scaling) - steepnessFactor*scaling/(2*np.pi*addedFactor-4*steepnessFactor*scaling))*factor_val+min_val-displace/(2*np.pi)*factor_val

    if type == "inverse_cdf":
        cdf_function = np.zeros_like(x)
        cdf_function[x > 2*np.pi] = (((steepnessFactor)*np.cos(x)[x > 2*np.pi]+(addedFactor)*(x-np.pi)[x > 2*np.pi])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val #integrate.cumtrapz(prior_ori(x), stim_ori_grid, initial=0.0)
        cdf_function[x <= np.pi] = (((steepnessFactor)*np.cos(x)[x <= np.pi]+(addedFactor)*(x)[x <= np.pi])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val-displace/(2*np.pi)*factor_val 
        cdf_function[(x > np.pi) & (x <= 2*np.pi)] = (((steepnessFactor)*-np.cos(x)[(x > np.pi) & (x <= 2*np.pi)]-2*steepnessFactor+(addedFactor)*(x)[(x > np.pi) & (x <= 2*np.pi)])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val-displace/(2*np.pi)*factor_val
        distance = (cdf_function - (factor_val*x/2/np.pi + min_val))/np.sqrt(2)
        value_function = (factor_val*x/2/np.pi + min_val) - distance
    
    if type == "curved_cdf":
        cdf_function = np.zeros_like(x)
        value_function = np.sin(x*2) + factor_val*x/2/np.pi + min_val

    if type == "prior":
        cdf_function = np.zeros_like(x)
        value_function = -np.sin(x*2) + factor_val*x/2/np.pi + min_val

    if type == "linearIncrease":
        value_function = min_val + abs((max_val-min_val)*(x-displace)/2/np.pi)

    if type == "linearDecrease":
        value_function = max_val -abs((max_val-min_val)*(x-displace)/2/np.pi)

    if type == "increasingSin":
        value_function = min_val + factor_val*np.sin(x/4.)      


    order = np.argsort(value_function)[::-1]
    value_function = value_function*(1-line_frac)+line[order]

    return value_function

# To identify how often is a distribution greater than a point (median analysis)
def dist_greater(grid, p, v): # v should lie on the grid
    p = np.squeeze(p)
    p = p[np.argsort(grid)]
    grid = np.sort(grid)

    # grid: 1d
    # p1/p2: n_orienations x n(grid)
    cdf = integrate.cumtrapz(p, grid, initial=0.0, axis=-1)

    cdf_v = cdf[grid <= v][-1]
    p_dist_greater = 1 - cdf_v
    return p_dist_greater

# Calculate how often distribution 1 is larger than distribution 2
# When both stimuli are gabors
def diff_dist(grid, p1, p2):
    p = []

    p1 = p1[:, np.argsort(grid)]
    p2 = p2[:, np.argsort(grid)]
    grid = np.sort(grid)

    # grid: 1d
    # p1/p2: n_orienations x n(grid)
    cdf2 = integrate.cumtrapz(p2, grid, initial=0.0, axis=1)

    # for every grid point, distribution 1 is bigger than distribution 2
    # with a probability of being that value times the probability that dist
    # 2 is lower than that value
    prob = p1*cdf2

    p.append(prob)

    # Cummulative probability
    return integrate.trapz(p, grid)

def get_rnp(safe_payoff, risky_payoffs, p_chose_risky, risk_prob):
    y = p_chose_risky.ravel()
    x = risky_payoffs.ravel()

    def get_probit(x, intercept, slope):
        return ss.norm(0.0, 1.0).cdf(intercept + slope*x)

    def cost(xs, ps, intercept, slope):
        return np.sum((get_probit(xs, intercept, slope) - ps)**2)

    def cost_(pars, *args):
        intercept, slope = pars
        return cost(x, y, intercept, slope)

    result = minimize(cost_, (-safe_payoff/risk_prob, 1.0), method='L-BFGS-B')

    intercept_est, slope_est = result.x

    indifference_point = -intercept_est/slope_est

    rnp = safe_payoff / indifference_point
    return rnp, slope_est


def inverse_monotonic(y_0, type, line_frac = 0):
    x = stim_ori_grid
    y = value_function_ori(x, type = type, line_frac = line_frac)
    indices = np.argsort(y)
    y_sorted = y[indices]
    
    def find_closest_index(y_0):
        i = np.searchsorted(y_sorted, y_0, side='left')
        if i == 0:
            return 0
        elif i == len(y_sorted):
            return len(y_sorted) - 1
        else:
            if y_0 - y_sorted[i-1] < y_sorted[i] - y_0:
                return i - 1
            else:
                return i

    i = find_closest_index(y_0)
    x_inverse = float(x[indices[i]]) # returns in radians like everything else
    return x_inverse

# This function takes in the original grid and the function that transforms the grid (random variable)
# and then throws out the transformed gridwith the function and the probability density
def ori_to_val_dist(grid, p, type, line_frac = 0.0, bins=500, monotonic=True, interpolation_kind='linear'):
    # The last dimension of p gives the probability distribution over the original grid
    # the first dimension could refer to another variable which dictates the probability distribution
    # So for example, for different thetas we could have distributions centered differently on stim_ori_grid 
    x_stim = np.array(grid)
    p_stim = p

    assert (x_stim.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"
    if p_stim.ndim == 1:
        p_stim = p_stim[np.newaxis,:]

    # If the transformation is monotonic and differentiable, then we can simply use the formula P(x)dx = P(y)dy for probabilities
    # This can be seen as ps = pstim*(d_stim/d_val)
    # also the grid just gets tranformed accorindg to the transformation function
    if monotonic:
        x_value = value_function_ori(x_stim, type, line_frac = line_frac)
        bin_centers = x_value
        grad_val = abs(np.gradient(x_value, x_stim)) #grad_value_ori(x_stim, type, line_frac)
        ps = p_stim
        # The last dimension of p which gives probability is stretched for new grid
        ps[...,:] = ps[...,:]/grad_val

    # If the function is not monotonic and differentiable, we use histograms
    else:
        # For every bin in x_stim, calculate the probability mass within that bin
        dx = x_stim[..., 1:] - x_stim[..., :-1]
        p_mass = ((p_stim[..., 1:] + p_stim[..., :-1]) / 2) * dx

        # Get the center of every bin
        x_value = value_function_ori(x_stim[:-1] + dx / 2., type, line_frac)
        ps = []
        for ix in range(len(p_stim)):
            h, edges = np.histogram(x_value, bins=bins, weights=p_mass[ix], density=True)
            ps.append(h)

        ps = np.array(ps)
        bin_centers = (edges[1:] + edges[:-1]) / 2

        f = interpolate.interp1d(bin_centers, ps, axis=1,
                                 kind=interpolation_kind, fill_value='extrapolate')

        ps = f(bin_centers)
        
    ps /= abs(trapezoid(ps, bin_centers, axis=1)[:, np.newaxis])
    # we sort the probabilities on the grid such that the grid is sorted in an increasing order as well.
    # This is useful for other functions we have
    ps = ps[..., np.argsort(bin_centers)]
    bin_centers = np.sort(bin_centers)

    return bin_centers, ps

# Prior here dictates the encoding of orientations and the cdf which governs the encoding transformation.
# Based on accuracy maximized codes that we propogate during training. 
def prior_ori(x):
    if experimentRange == "00to180" or experimentRange == "45to225":
        return (addedFactor - np.abs(steepnessFactor*np.sin(x))) / (-4*steepnessFactor + 2*addedFactor*np.pi)# return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 4.0
    if experimentRange == "00to90" or experimentRange == "90to180":
        return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 2.0
    if experimentRange == "00to45" or experimentRange == "45to90" or experimentRange == "90to135" or experimentRange == "135to180":
        return (2 - np.abs(np.sin(x))) / (np.pi - 1)
    
def cdf_ori(x, grid): # goes from 0 to 2pi
    cdf_ori = integrate.cumtrapz(prior_ori(x), grid, initial=0.0)*factor_ori
    return cdf_ori
    
## This is now the contextual prior which governs the prior used in bAYESIAN DECODING OF ORIENTATIONS AND ALSO 
# the priors formed for values and encoding of values (cdf_val)
def context_prior_ori(x):
    if contextual_prior == "uniform":
        return np.repeat(1/(2.*np.pi), len(x))
    if contextual_prior == "gaussian":
        return ss.vonmises(loc=np.pi, kappa=0.5).pdf(x)
    if contextual_prior == "increasing":
        return (np.pi+x)/(2*(np.pi**2))
    if contextual_prior == "decreasing":
        return (3*np.pi-x)/(2*(np.pi**2))

# Getting a prior over values given a orientation grid and a type, we can use the contextual prior over ori here only because the training was donje with
# 0 noise conditions
def prior_val(type, line_frac = 0):
    p_ori = context_prior_ori(stim_ori_grid)
    # Probability on each point of ori grid can be converted to probability on val grid points.
    # the value grid is just a functional transform of ori grid. stim_val_grid is simply the transform of
    #the original stim_otri_grid
    stim_val_grid, ps = ori_to_val_dist(stim_ori_grid, p_ori, type, line_frac)
    ps = np.squeeze(ps) # Brings it back to 1 dime
    return stim_val_grid, ps


# Takes in the orientation grid and gives out the cdf over values
def cdf_val(type, line_frac = 0):
    stim_val_grid, ps = prior_val(type, line_frac = line_frac)
    cdf_value = np.squeeze(integrate.cumtrapz(ps, stim_val_grid, initial=0.0))*factor_val
    return stim_val_grid, cdf_value

def stimulus_ori_noise(x, kappa_s, grid):
    # return np.exp(kappa_s*(np.cos(x - grid)-1))
    return ss.vonmises(loc=x, kappa=kappa_s).pdf(grid)

def sensory_ori_noise(m, kappa_r, grid):
    sigma_rep = np.sqrt(factor_val/kappa_r)
    if experimentRange == "00to180" or experimentRange == "45to225":
        return ss.vonmises(loc=m, kappa=kappa_r).pdf(grid)
    else:
        truncBoth = ss.truncnorm.pdf(grid,(start*np.pi/90 - m) / sigma_rep, (end*np.pi/90-m) / sigma_rep, m, sigma_rep)
        return truncBoth
    
# Getting the noisy input stimulus distribution in value space mapping
# Input parameters define noise in orientation space buy function gives out the noisy distribution in value space
def stimulus_val_noise(x, kappa_s, grid, type, line_frac = 0.0):
    if np.isscalar(x):
        x = np.array([x])
    else:
        x = np.array(x)
    x = x[:, np.newaxis]

    p_noise_ori = ss.vonmises(loc=x, kappa=kappa_s).pdf(grid[np.newaxis, :])
    # The second dimension of p which hass probability over original geid gets changed to probability
    # over new grid. the new grid is also returned. The first dimension remains unchanged for ps.
    stim_val_grid, ps = ori_to_val_dist(grid, p_noise_ori, type, line_frac = line_frac)

    return stim_val_grid, ps

def sensory_val_noise(m, sigma_rep, grid):
    truncBoth = ss.truncnorm.pdf(grid,(min_val - m) / sigma_rep, (max_val -m) / sigma_rep, m, sigma_rep)
    return truncBoth