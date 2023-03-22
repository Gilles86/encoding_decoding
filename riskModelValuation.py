import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
# from scipy.stats import gaussian_kde


# experiment = "oneSideTruncated" # "noBoundaryEffects" #"bothSideTruncated" #oneSideTruncated, bothSideTruncated, bothSideFolded
# this needs to be set based on how yopu do your experiment
experimentRange = "00to180" #"00to45", "45to90", "90to135", "135to180", "00to90", "90to180", "00to180", "noBoundaryEffects" # For now we predict the sensory noise type to be shaped (trucated, folded whatever)

if experimentRange == "00to45" or experimentRange == "45to90" or experimentRange == "00to90":
    end = int(experimentRange[-2:])
    start = int(experimentRange[0:2])
if experimentRange == "90to135" or experimentRange == "90to180":
    end = int(experimentRange[-3:])
    start = int(experimentRange[0:2])
if experimentRange == "135to180":
    end = int(experimentRange[-3:])
    start = int(experimentRange[0:3])
if experimentRange == "00to180" or experimentRange == "noBoundaryEffects":
    end = 180
    start = 0
factor = (end-start)/90.*np.pi

stim_grid = np.linspace(start, end, 501) * np.pi / 90
rep_grid = np.linspace(0, 1, 301)  # value based representation in this code

max_val = 42
min_val = 2

# Getting a prior over orientation given a grid (x)
def prior_ori(x):
    if experimentRange == "00to180" or experimentRange == "noBoundaryEffects":
        return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 4.0
    if experimentRange == "00to90" or experimentRange == "90to180":
        return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 2.0
    if experimentRange == "00to45" or experimentRange == "45to90" or experimentRange == "90to135" or experimentRange == "135to180":
        return (2 - np.abs(np.sin(x))) / (np.pi - 1)


# Getting a value function given a grid of orientations (stim_grid for example). Always between 0 and 2pi.
def value_function_ori(x, type):
    x = np.array(x)
    if type == "prior":
        value_function = np.zeros_like(x)
        value_function[x <= np.pi/2] = max_val - ((max_val-min_val)/4)*(np.sin(x[x <= np.pi/2]))
        value_function[(x > np.pi/2) & (x <= 3*np.pi/2)] = (max_val - (max_val-min_val)/2) - ((max_val-min_val)/4)*( - np.sin(x[(x > np.pi/2) & (x <= 3*np.pi/2)]))
        value_function[x > 3*np.pi/2] = min_val - ((max_val-min_val)/4)*(np.sin(x[x > 3*np.pi/2]))

    if type == "linearPrior":
        value_function = max_val -abs((max_val-min_val)*x/2/np.pi)

    if type == "curvedPrior":
        value_function = np.zeros_like(x)
        value_function[x <= np.pi] = (max_val)-((max_val-min_val)/4)*(1-np.cos(x[x <= np.pi]))
        value_function[(x > np.pi) & (x <= np.pi*2)] = (max_val+min_val)/2 -((max_val-min_val)/4)*(np.cos(x[(x > np.pi) & (x <= np.pi*2)])+1)

    if type == "inversePrior":
        value_function = np.zeros_like(x)
        value_function[x <= np.pi/2] = min_val + ((max_val-min_val)/4)*(np.sin(x[x <= np.pi/2]))
        value_function[(x > np.pi/2) & (x <= 3*np.pi/2)] = min_val + ((max_val-min_val)/4)*(2 - np.sin(x[(x > np.pi/2) & (x <= 3*np.pi/2)]))
        value_function[x > 3*np.pi/2] = ((max_val-min_val)/4)*(np.sin(x[x > 3*np.pi/2])) + max_val

    if type == "inverseLinearPrior":
        value_function = min_val + abs((max_val-min_val)*x/2/np.pi)

    if type == "inverseCurvedPrior":
        value_function = np.zeros_like(x)
        value_function[x <= np.pi] = min_val + ((max_val-min_val)/4)*(1-np.cos(x[x <= np.pi]))
        value_function[(x > np.pi) & (x <= np.pi*2)] = (max_val+min_val)/2 +((max_val-min_val)/4)*(np.cos(x[(x > np.pi) & (x <= np.pi*2)])+1)

    return value_function


# A general numerical method is used here to find new distributions of functions of distributions we know.
# Takes in the grid used in original distribution, and the distribution of the original distribution and then
# finds the distribution of the functional value distribution.
def ori_to_val_dist(grid, p, type, interpolation_kind='linear', bins=100, slow=True):
    x_stim = np.array(grid)
    p_stim = p

    assert (x_stim.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"
    if p_stim.ndim == 1:
        p_stim = p_stim[np.newaxis,:]

    # For every bin in x_stim, calculate the probability mass within that bin
    dx = x_stim[..., 1:] - x_stim[..., :-1]
    p_mass = ((p_stim[..., 1:] + p_stim[..., :-1]) / 2) * dx

    # Get the center of every bin
    x_value = value_function_ori(x_stim[:-1] + dx / 2., type)

    if slow:
        ps = []
        for ix in range(len(p_stim)):
            h, edges = np.histogram(x_value, bins=bins, weights=p_mass[ix], density=True)
            ps.append(h)

        ps = np.array(ps)
        bin_centers = (edges[1:] + edges[:-1]) / 2

        f = interpolate.interp1d(bin_centers, ps, axis=1,
                                 kind=interpolation_kind, fill_value='extrapolate')

        ps = f(bin_centers)
        ps /= np.trapz(ps, bin_centers, axis=1)[:, np.newaxis]

    return bin_centers, ps


# Getting a prior over values given a orientation grid and a type
def prior_val(grid, type, interpolation_kind='linear', bins=100, slow=True):

    p_ori = prior_ori(grid)

    bin_centers, ps = ori_to_val_dist(grid, p_ori, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)
    ps = np.squeeze(ps) # Brings it back to 1 dime

    return bin_centers, ps


# The noisy input stimulus being presented
def stimulus_ori_noise(x, kappa_s, grid):
    p_noise_ori = ss.vonmises(loc=x, kappa=kappa_s).pdf(grid)
    return p_noise_ori

# Getting the noisy input stimulus distribution in value space mapping
# Input parameters define noise in orientation space buy function gives out the noisy distribution in value space
def stimulus_val_noise(x, kappa_s, grid, type, interpolation_kind='linear', bins=100, slow=True):
    # grid = grid[np.newaxis, :]
    # grid = np.array(grid)
    if np.isscalar(x):
        x = np.array([x])
    else:
        x = np.array(x)
    x = x[:, np.newaxis]

    p_noise_ori = ss.vonmises(loc=x, kappa=kappa_s).pdf(grid[np.newaxis, :])

    bin_centers, ps = ori_to_val_dist(grid, p_noise_ori, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)

    return ps 


# This noise will be added to the representation space of the variable being encoded.
# This variable is value in this case
# It should not be a vonmises anymore. In fact what it should be depends on the specific type of experimental setup we
# design in my opinion. If participants are always just showed one quadrant, makes more sense that it is a truncated
# normal in my opinion. hOWEVER, IF IT IS 2 QUADRANTS, SHOULD be a recursive value that repeats back in opposite direction.
def sensory_noise(m, sigma_rep, grid, type):
    truncBoth = ss.truncnorm.pdf(grid,(0.0 - m) / sigma_rep, (1.0 -m) / sigma_rep, m, sigma_rep)
    # truncUp = ss.truncnorm.pdf(grid, (-np.Inf - m) / sigma_rep, (1. - m) / sigma_rep, m, sigma_rep)
    # truncLow = ss.truncnorm.pdf(grid, (0.0 - m) / sigma_rep, (np.Inf - m) / sigma_rep, m, sigma_rep)
    # foldUp = ss.foldnorm.pdf(1-grid, (1-m)/sigma_rep, 0.0, sigma_rep) # 0.0 here is essentially the point of the folding
    # foldLow = ss.foldnorm.pdf(grid, m/sigma_rep, 0.0, sigma_rep)
    # if experimentRange == "00to90" or "90to180":
    #     if type == "linearPrior" or type == "prior" or type =="curvedPrior":
    #         return (1-m)*foldLow + m*truncUp # The lowed values get folded while upper ones are truncated
    #     if type == "inverseLinearPrior" or type == "inversePrior" or type =="inverseCurvedPrior":
    #         return m*foldUp + (1-m)*truncLow # The lowed values get truncated while upper ones are folded
    # if experimentRange == "00to180" or experimentRange == "noBoundaryEffects":
        # We assume that they represent the information here totally and the bounds do not mean truncation but rather,
        # they jusst shift the boundary values to include 5 standard deviations from the noise, whatever the noise is
        # return ss.norm.pdf(grid, m+5*sigma_rep*(0.5-m), sigma_rep)
    return truncBoth

# Takes in the orientation grid and gives out the cdf over values
def cdf_val(grid, type, bins =100):
    bin_centers, ps = prior_val(grid, type, bins = bins)
    cdf_val = np.squeeze(integrate.cumtrapz(ps, bin_centers, initial=0.0))
    return cdf_val


# Take input orientation and gives the decoded distribution
def value_efficient_encoding(theta0, kappa_s, sigma_rep, type, interpolation_kind='linear', bins = 100, slow=True):
    # Note that the sigma_stim is in ori space

    theta0 = np.atleast_1d(theta0)
    # Prior over values for the given type of value function mapping
    val_centers, val_prior = prior_val(stim_grid, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)

    # val0 (has same dim as theta0) x theta_gen x m_gen
    # Add stimulus noise to get distribution of values given theta0 - val0 x value_gen
    p_val_given_theta0 = stimulus_val_noise(theta0, kappa_s, stim_grid, type, bins=bins)
    # Add sensory noise to see what ms for value you get given a value 0 - value_gen_rep x m_gen(rep)
    # For each point in rep space given by the cdf of the value function grid, we add sensory noise constant
    p_m_given_val = sensory_noise(cdf_val(stim_grid, type, bins = bins)[np.newaxis, :, np.newaxis], sigma_rep, rep_grid[np.newaxis, np.newaxis, :], type)

    # Combine sensory and stimulus noise
    p_m_given_val0 = p_m_given_val * p_val_given_theta0[..., np.newaxis]

    # Integrate out different generated values (due to stim noise), so we just have ms given theta0
    p_m_given_val0 = trapezoid(p_m_given_val0, val_centers, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (value likelihood)
    p_m_given_val = stimulus_val_noise(stim_grid, kappa_s, stim_grid, type, bins=bins)[..., np.newaxis] * \
        sensory_noise(cdf_val(stim_grid, type, bins = bins)[np.newaxis, :, np.newaxis], sigma_rep, rep_grid[np.newaxis, np.newaxis, :], type)

    # Integrate out the realized values
    p_m_given_val = trapezoid(p_m_given_val, stim_grid, axis=0)

    # Representations of values
    return p_m_given_val0, p_m_given_val

# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, kappa_s, sigma_rep, type, interpolation_kind='linear', bins=100, slow=True):

    # There is a one to one correspondence between theta0 and corresponding val0
    # val0 is implicitly presented val by presenting theta0.

    p_m_given_val0, p_m_given_val = value_efficient_encoding(theta0, kappa_s, sigma_rep, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)
    # just changed this next line from the comebted one
    p_val_given_m = p_m_given_val*np.array(prior_val(stim_grid, type, bins = bins)[1])[:, np.newaxis]
    # p_val_given_m = p_m_given_val * prior_ori(stim_grid)[:, np.newaxis]

    # Normalize with p(m)
    p_val_given_m = p_val_given_m / (trapezoid(p_val_given_m, prior_val(stim_grid, type, bins=bins)[0] , axis=0)[np.newaxis,:])

    # theta0 x theta_tilde x m
    # Probability of estimating \hat{theta} given theta0
    p_value_est_given_val0 = p_m_given_val0[:, np.newaxis, :] * p_val_given_m[np.newaxis, ...]

    # Get rid of m
    p_value_est_given_val0 = trapezoid(p_value_est_given_val0, rep_grid, axis=2)

    # normalize (99% sure that not necessary)
    # p_thetaest_given_theta0 /= trapezoid(p_thetaest_given_theta0, stim_grid, axis=1)[:, np.newaxis]
    val = prior_val(stim_grid, type, bins=bins)[0]
    return val, p_value_est_given_val0

def risky_value_dist(theta1, kappa_s, sigma_rep, risk_prob, type, interpolation_kind='linear', bins=100, slow=True):

    bin_centers, ps = value_bayesian_decoding(theta1, kappa_s, sigma_rep, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)

    risky_value = bin_centers*risk_prob
    p_risky = ps/risk_prob

    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(bin_centers)

    return bin_centers, p_risky
