import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
# from scipy.stats import gaussian_kde

import tools as tools

# experiment = "oneSideTruncated" # "noBoundaryEffects" #"bothSideTruncated" #oneSideTruncated, bothSideTruncated, bothSideFolded
# this needs to be set based on how you do your experiment
experimentRange = tools.experimentRange #"00to45", "45to90", "90to135", "135to180", "00to90", "90to180", "00to180", "noBoundaryEffects" # For now we predict the sensory noise type to be shaped (trucated, folded whatever)
start = tools.start
end = tools.end

stim_ori_grid = tools.stim_ori_grid
rep_val_grid = tools.rep_val_grid

max_val = tools.max_val
min_val = tools.min_val

factor_val = max_val - min_val

# Getting a prior over values given a orientation grid and a type
def prior_val(type, line_frac = 0):
    p_ori = tools.prior_ori(stim_ori_grid)
    # Probability on each point of ori grid can be converted to probability on val grid points.
    # the value grid is just a functional transform of ori grid. stim_val_grid is simply the transform of
    #the original stim_otri_grid
    stim_val_grid, ps = tools.ori_to_val_dist(stim_ori_grid, p_ori, type, line_frac)
    ps = np.squeeze(ps) # Brings it back to 1 dime
    return stim_val_grid, ps

# Takes in the orientation grid and gives out the cdf over values
def cdf_val(type, line_frac = 0):
    stim_val_grid, ps = prior_val(type, line_frac = line_frac)
    cdf_value = np.squeeze(integrate.cumtrapz(ps, stim_val_grid, initial=0.0))*factor_val
    return stim_val_grid, cdf_value

# The noisy input stimulus being presented
def stimulus_ori_noise(x, kappa_s, grid):
    p_noise_ori = ss.vonmises(loc=x, kappa=kappa_s).pdf(grid)
    return p_noise_ori

# Getting the noisy input stimulus distribution in value space mapping
# Input parameters define noise in orientation space buy function gives out the noisy distribution in value space
def stimulus_val_noise(x, kappa_s, grid, type, line_frac = 0.0):
    # grid = grid[np.newaxis, :]
    # grid = np.array(grid)
    if np.isscalar(x):
        x = np.array([x])
    else:
        x = np.array(x)
    x = x[:, np.newaxis]

    p_noise_ori = ss.vonmises(loc=x, kappa=kappa_s).pdf(grid[np.newaxis, :])

    # The second dimension of p which hass probability over original geid gets changed to probability
    # over new grid. the new grid is also returned. The first dimension remains unchanged for ps.
    stim_val_grid, ps = tools.ori_to_val_dist(grid, p_noise_ori, type, line_frac = line_frac)

    return stim_val_grid, ps 


# This noise will be added to the representation space of the variable being encoded.
# This variable is value in this case. Thus in rep_vel_grid domain
# It should not be a vonmises anymore. In fact what it should be depends on the specific type of experimental setup we
# design in my opinion. If participants are always just showed one quadrant, makes more sense that it is a truncated
# normal in my opinion. hOWEVER, IF IT IS 2 QUADRANTS, SHOULD be a recursive value that repeats back in opposite direction.
def sensory_noise(m, sigma_rep, grid, type):
    truncBoth = ss.truncnorm.pdf(grid,(min_val - m) / sigma_rep, (max_val -m) / sigma_rep, m, sigma_rep)
    return truncBoth

# Take input orientation and gives the encoded distribution over value representation
def value_efficient_encoding(theta0, kappa_s, sigma_rep, type, line_frac = 0):
    # Note that the kappa_s is in ori space
    theta0 = np.atleast_1d(theta0)
    # The new grid is basically just the original grid transformed with the value function
    # The cdf over the new grid is basically the probability over the new grid integrated over the new grid
    stim_val_grid, cdf_value = cdf_val(type)

    # val0 (has same dim as theta0) x theta_gen x m_gen
    # Stimulus noise in theta domain simply gets transformed to the stimulus noise distribution in the new domain, without 
    # any distortion effects or so.
    stim_val_grid, p_val_given_theta0 = stimulus_val_noise(theta0, kappa_s, stim_ori_grid, type)

    # Add sensory noise to see what ms for value you get given a value 0 - value_gen_rep x m_gen(rep)
    # Each presented value gives a distribution in sensory space that is centered around the distorted mean according to the cdf
    # of the encoded variable (cdf_val) and on an equallyt sized representational grid.
    p_m_given_val = sensory_noise(cdf_value[np.newaxis, :, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :], type)

    # Combine sensory and stimulus noise
    p_m_given_theta0 = p_m_given_val * p_val_given_theta0[..., np.newaxis]

    # Integrate out different generated values (due to stim noise), so we just have ms given theta0. The integration is on
    # value grid generated from mapping of orientation to value. The spacing of the grid depends on whether the mapping was
    # linear or non linear
    p_m_given_theta0 = trapezoid(p_m_given_theta0, stim_val_grid, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (value likelihood)
    p_m_given_theta = (stimulus_val_noise(stim_ori_grid, kappa_s, stim_ori_grid, type)[1])[..., np.newaxis] *\
        sensory_noise(cdf_value[np.newaxis, :, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :], type)

    # Integrate out the original values resulting from noisy stimulus thetas. We now get a grid of m's for all possible equally spaced
    # points pon the theta grid
    p_m_given_theta = trapezoid(p_m_given_theta, stim_val_grid, axis=1)

    # Since the value function is monotonic, each theta gets transformed to the value grid. The prob
    #over values for a theta is the same as the prob over values for the given transformed val for the theta
    # Rearranging thetas such that p_m_given_val is on an increasing value grid always
    p_m_given_val = p_m_given_theta[np.argsort(tools.value_function_ori(tools.stim_ori_grid, type))]

    # Each point of theta in theta grid can be transformed to value space and the corresponding probabilities into tyhat space by using 
    # the general function from tools
    # stim_val_grid, p_m_given_val = tools.ori_to_val_dist(stim_ori_grid, p_m_given_theta, type, line_frac = line_frac)

    # Representations of values
    return p_m_given_theta0, p_m_given_val

# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, kappa_s, sigma_rep, type, line_frac = 0):

    # encoded m's for given presented value and value grid (distorted val grid based on function)
    p_m_given_theta0, p_m_given_val = value_efficient_encoding(theta0, kappa_s, sigma_rep, type, line_frac=line_frac)

    # we need to use the transformed stim vbalues and prior over values now
    safe_value, val_prior = prior_val(type, line_frac = line_frac)

    # Applying bayes rule to get the p(val/m). Combining evidence in representation with the prior over variable of interest
    p_val_given_m = p_m_given_val*np.array(val_prior)[:, np.newaxis]

    # Normalize with p(m) = p(m|val)*p(val) that we just defined as p(val|m) in above line
    p_val_given_m = p_val_given_m / trapezoid(p_val_given_m, safe_value, axis=0)[np.newaxis,:]

    # Probability of estimating \hat{value} given val0
    p_value_est_given_theta0 = p_m_given_theta0[:, np.newaxis, :] * p_val_given_m[np.newaxis, ...]

    # Get rid of m
    p_value_est_given_theta0 = trapezoid(p_value_est_given_theta0, rep_val_grid, axis=2)

    p_value_est_given_theta0 /= trapezoid(p_value_est_given_theta0, safe_value, axis=1)[:, np.newaxis]

    return safe_value, p_value_est_given_theta0

def safe_value_dist(theta0, kappa_s, sigma_rep, type, line_frac = 0):
    safe_value, safe_prob = value_bayesian_decoding(theta0, kappa_s, sigma_rep, type, line_frac = line_frac)
    return safe_value, safe_prob

def risky_value_dist(theta1, kappa_s, sigma_rep, risk_prob, type, line_frac = 0):

    safe_value, safe_prob = value_bayesian_decoding(theta1, kappa_s, sigma_rep, type, line_frac = line_frac)

    risky_value = safe_value*risk_prob
    p_risky = safe_prob/risk_prob

    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(safe_value)

    return safe_value, p_risky
