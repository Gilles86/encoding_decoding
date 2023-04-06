import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

# Settings for all code experiments. setting paradigm
prior_used = "wei" #"steep", #wei
experimentRange = "00to180" #"00to180" # "45to225"
max_val = 42
min_val = 2

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

displace = start/90.*np.pi

stim_ori_grid = np.linspace(start, end, 501) * np.pi / 90
rep_ori_grid = np.linspace(start, end, 501) * np.pi / 90

rep_val_grid = np.linspace(min_val, max_val, 501)

addedFactor = max_val
if prior_used == "steep":
    steepnessFactor = addedFactor*0.95 #addedFactor/factor_val*factor_val
if prior_used == "wei": # generates prior of 2-abs(sin(x))
    steepnessFactor = addedFactor*0.5

# Getting a value function given a grid of orientations (stim_ori_grid for example). Always between 0 and 2pi.
def value_function_ori(x, type, line_frac = 0):
    x = np.array(x)
    line = abs(stim_ori_grid/np.pi/2.*factor_val - max_val)*line_frac

    if experimentRange == "00to180":

        if type == "cdf_prior":

            value_function = np.zeros_like(x)
            value_function[x <= np.pi] = (((steepnessFactor)*np.cos(x)[x <= np.pi]+(addedFactor)*x[x <= np.pi])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val #integrate.cumtrapz(prior_ori(x), stim_ori_grid, initial=0.0)
            value_function[x > np.pi] = (((steepnessFactor)*-np.cos(x)[x > np.pi]-2*steepnessFactor+(addedFactor)*x[x > np.pi])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val

        if type == "increase_cdf_prior":  # makes probabilities over values linearly increasing
            value_function = np.zeros_like(x)
            value_function[x <= np.pi] = (((steepnessFactor)*np.cos(x)[x <= np.pi]+(addedFactor)*x[x <= np.pi])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val #integrate.cumtrapz(prior_ori(x), stim_ori_grid, initial=0.0)
            value_function[x > np.pi] = (((steepnessFactor)*-np.cos(x)[x > np.pi]-2*steepnessFactor+(addedFactor)*x[x > np.pi])/(2*np.pi*addedFactor-4*steepnessFactor) - steepnessFactor/(2*np.pi*addedFactor-4*steepnessFactor))*factor_val+min_val
            value_function = np.sqrt(value_function)*factor_val+min_val

        if type == "prior":
            value_function = np.zeros_like(x)
            value_function[x <= np.pi/2] = (max_val - ((max_val-min_val)/4)*(np.sin((x)[x <= np.pi/2])))
            value_function[(x > np.pi/2) & (x <= 3*np.pi/2)] = (max_val - (max_val-min_val)/2) - ((max_val-min_val)/4)*( - np.sin((x)[(x > np.pi/2) & (x <= 3*np.pi/2)]))
            value_function[x > 3*np.pi/2] = min_val - ((max_val-min_val)/4)*(np.sin((x)[x > 3*np.pi/2]))
            
        if type == "linearIncrease":
            value_function = min_val + abs((max_val-min_val)*x/2/np.pi)

        if type == "linearDecrease":
            value_function = max_val -abs((max_val-min_val)*x/2/np.pi)

        if type == "increasingSin":
            value_function = min_val + factor_val*np.sin(x/4.)      

        if type == "curvedPrior":
            value_function = np.zeros_like(x)
            value_function[x <= np.pi] = (max_val)-((max_val-min_val)/4)*(1-np.cos(x[x <= np.pi]))
            value_function[(x > np.pi) & (x <= np.pi*2)] = (max_val+min_val)/2 -((max_val-min_val)/4)*(np.cos(x[(x > np.pi) & (x <= np.pi*2)])+1)

        if type == "inversePrior":
            value_function = np.zeros_like(x)
            value_function[x <= np.pi/2] = min_val + ((max_val-min_val)/4)*(np.sin(x[x <= np.pi/2]))
            value_function[(x > np.pi/2) & (x <= 3*np.pi/2)] = min_val + ((max_val-min_val)/4)*(2 - np.sin(x[(x > np.pi/2) & (x <= 3*np.pi/2)]))
            value_function[x > 3*np.pi/2] = ((max_val-min_val)/4)*(np.sin(x[x > 3*np.pi/2])) + max_val

        if type == "inverseCurvedPrior":
            value_function = np.zeros_like(x)
            value_function[x <= np.pi] = min_val + ((max_val-min_val)/4)*(1-np.cos(x[x <= np.pi]))
            value_function[(x > np.pi) & (x <= np.pi*2)] = (max_val+min_val)/2 +((max_val-min_val)/4)*(np.cos(x[(x > np.pi) & (x <= np.pi*2)])+1)


    if experimentRange == "45to225":

        if type == "cdf_prior":
            value_function = integrate.cumtrapz(prior_ori(x-displace), stim_ori_grid, initial=0.0)*factor_val+min_val

    order = np.argsort(value_function)[::-1]
    value_function = value_function*(1-line_frac)+line[order]

    return value_function



# Calculate how often distribution 1 is larger than distribution 2
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
    return rnp

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
    x_stim = np.array(grid)
    p_stim = p

    assert (x_stim.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"
    if p_stim.ndim == 1:
        p_stim = p_stim[np.newaxis,:]

    # If the transformation is monotonic, then we can simply use the formula P(x)dx = P(y)dy for probabilities
    # This can be seen as ps = pstim*(d_stim/d_val)
    # also the grid just gets tranformed accorindg to the transformation function
    if monotonic:
        x_value = value_function_ori(x_stim, type, line_frac = line_frac)
        bin_centers = x_value
        grad_val = np.gradient(x_value, x_stim) #grad_value_ori(x_stim, type, line_frac)
        grad_ori = 1/grad_val
        ps = p_stim*abs(grad_ori)

    # If the function is not monotonic, we use histograms
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
        
    ps /= trapezoid(ps, bin_centers, axis=1)[:, np.newaxis]
    # we sort the probabilities on the grid such that the grid is sorted in an increasing order as well.
    # This is useful for other functions we have
    ps = ps[..., np.argsort(bin_centers)]
    bin_centers = np.sort(bin_centers)

    return bin_centers, ps

# Prior
def prior_ori(x):
    if experimentRange == "00to180" or experimentRange == "45to225":
        return (addedFactor - np.abs(steepnessFactor*np.sin(x))) / (-4*steepnessFactor + 2*addedFactor*np.pi)# return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 4.0
    if experimentRange == "00to90" or experimentRange == "90to180":
        return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 2.0
    if experimentRange == "00to45" or experimentRange == "45to90" or experimentRange == "90to135" or experimentRange == "135to180":
        return (2 - np.abs(np.sin(x))) / (np.pi - 1)


# analytical solutions for gradient of the original value function with respect to the orientations
def grad_value_ori(x, type, line_frac = 0.0):
    x = np.array(x)
    if experimentRange == "00to180":

        if type == "cdf_prior":
            grad_val = prior_ori(x)*factor_val

        if type == "increase_cdf_prior":
            grad_val = prior_ori(x)*factor_val/2./np.sqrt(integrate.cumtrapz(prior_ori(x), stim_ori_grid, initial=0.00001))

        if type == "prior":
            grad_val = np.zeros_like(x)
            grad_val[x <= np.pi/2] = ((max_val-min_val)/4)*(np.cos((x)[x <= np.pi/2]))
            grad_val[(x > np.pi/2) & (x <= 3*np.pi/2)] = ((max_val-min_val)/4)*( - np.cos((x)[(x > np.pi/2) & (x <= 3*np.pi/2)]))
            grad_val[x > 3*np.pi/2] = ((max_val-min_val)/4)*(np.cos((x)[x > 3*np.pi/2]))

        if type == "linearPrior":
            grad_val = -abs((max_val-min_val)*1/2/np.pi)

        if type == "curvedPrior":
            grad_val = np.zeros_like(x)
            grad_val[x <= np.pi] = ((max_val-min_val)/4)*(1+np.sin(x[x <= np.pi]))
            grad_val[(x > np.pi) & (x <= np.pi*2)] = -((max_val-min_val)/4)*(-np.sin(x[(x > np.pi) & (x <= np.pi*2)]))

        if type == "inversePrior":
            grad_val = np.zeros_like(x)
            grad_val[x <= np.pi/2] = ((max_val-min_val)/4)*(np.cos(x[x <= np.pi/2]))
            grad_val[(x > np.pi/2) & (x <= 3*np.pi/2)] = ((max_val-min_val)/4)*(2 - np.cos(x[(x > np.pi/2) & (x <= 3*np.pi/2)]))
            grad_val[x > 3*np.pi/2] = ((max_val-min_val)/4)*(np.cos(x[x > 3*np.pi/2]))

        if type == "inverseLinearPrior":
            grad_val = abs((max_val-min_val)*1/2/np.pi)

        if type == "inverseCurvedPrior":
            grad_val = np.zeros_like(x)
            grad_val[x <= np.pi] = ((max_val-min_val)/4)*(1+np.sin(x[x <= np.pi]))
            grad_val[(x > np.pi) & (x <= np.pi*2)] = ((max_val-min_val)/4)*(-np.sin(x[(x > np.pi) & (x <= np.pi*2)]))

    if experimentRange == "45to225":

        if type == "cdf_prior":
            grad_val = prior_ori(x-displace)*factor_val

        if type == "prior":
            grad_val = np.zeros_like(x)
            grad_val[x <= np.pi+displace] = ((max_val-min_val)/4)*(1+np.sin((x-displace)[x <= np.pi+displace]))
            grad_val[(x > np.pi+displace) & (x <= np.pi*2+displace)] = -((max_val-min_val)/4)*(-np.sin((x-displace)[(x > np.pi+displace) & (x <= np.pi*2+displace)])+1)

        if type == "linearPrior":
            grad_val = -abs((max_val-min_val)*(1)/2/np.pi)

        if type == "curvedPrior":
            grad_val = np.zeros_like(x)
            grad_val[x <= np.pi/2+displace] = ((max_val-min_val)/4)*(np.cos((x-displace)[x <= np.pi/2+displace]))
            grad_val[(x > np.pi/2+displace) & (x <= 3*np.pi/2+displace)] = - ((max_val-min_val)/4)*( - np.cos((x-displace)[(x > np.pi/2+displace) & (x <= 3*np.pi/2+displace)]))
            grad_val[x > 3*np.pi/2+displace] = - ((max_val-min_val)/4)*(np.cos((x-displace)[x > 3*np.pi/2+displace]))

        if type == "inversePrior":
            grad_val = np.zeros_like(x)
            grad_val[x <= np.pi+displace] = ((max_val-min_val)/4)*(1+np.sin((x-displace)[x <= np.pi+displace]))
            grad_val[(x > np.pi+displace) & (x <= np.pi*2+displace)] = ((max_val-min_val)/4)*(-np.sin((x-displace)[(x > np.pi+displace) & (x <= np.pi*2+displace)])+1)


        if type == "inverseLinearPrior":
            grad_val = abs((max_val-min_val)*(1)/2/np.pi)

        if type == "inverseCurvedPrior":
            grad_val = np.zeros_like(x)
            grad_val[x <= np.pi/2+displace] = ((max_val-min_val)/4)*(-np.cos((x-displace)[x <= np.pi/2+displace]))
            grad_val[(x > np.pi/2+displace) & (x <= 3*np.pi/2+displace)] = ((max_val-min_val)/4)*(2 - np.cos((x-displace)[(x > np.pi/2+displace) & (x <= 3*np.pi/2+displace)]))
            grad_val[x > 3*np.pi/2+displace] = ((max_val-min_val)/4)*(np.cos((x-displace)[x > 3*np.pi/2+displace]))

    grad_val = grad_val + line_frac/np.pi/2.*factor_val

    return grad_val
