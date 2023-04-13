import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

import tools as tools
import riskSingleObserverPercept as perception

experimentRange = tools.experimentRange
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

def sensory_val_noise(m, sigma_rep, grid, type):
    truncBoth = ss.truncnorm.pdf(grid,(min_val - m) / sigma_rep, (max_val -m) / sigma_rep, m, sigma_rep)
    return truncBoth

# Gives the value distribution output coming out of the perceptual system
def input_to_val_system(theta0, kappa_s, kappa_r, type):
    stim_val_grid, val_input = perception.safe_value_dist(theta0, kappa_s, kappa_r, type, line_frac = 0.0)
    return stim_val_grid, val_input

# Take the val_input as the incoming signal from the first observer and add an additional 
# representational noise in the value representation to then get the encoded value distribution
# for the first system
def value_efficient_encoding(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = 0):
    theta0 = np.atleast_1d(theta0)
    stim_val_grid, val_input = input_to_val_system(theta0, kappa_s, kappa_r, type)

    # The new grid is basically just the original grid transformed with the value function
    # The cdf over the new grid is basically the probability over the new grid integrated over the new grid
    stim_val_grid, cdf_value = cdf_val(type)

    # Add sensory noise to see what ms for value you get given a value 0 - value_gen_rep x m_gen(rep)
    # Each presented value gives a distribution in sensory space that is centered around the distorted mean according to the cdf
    # of the encoded variable (cdf_val) and on an equallyt sized representational grid.
    p_mVal_given_val = sensory_val_noise(cdf_value[np.newaxis, :, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :], type)

    # Combine sensory and stimulus noise
    p_mVal_given_theta0 = p_mVal_given_val * val_input[..., np.newaxis]

    # Integrate out different generated values (due to stim noise), so we just have ms given theta0. The integration is on
    # value grid generated from mapping of orientation to value. The spacing of the grid depends on whether the mapping was
    # linear or non linear
    p_mVal_given_theta0 = trapezoid(p_mVal_given_theta0, stim_val_grid, axis=1)

    # now useful for decode later
    p_mVal_given_val = input_to_val_system(stim_ori_grid, kappa_s, kappa_r, type)[1] [..., np.newaxis] *\
        sensory_val_noise(cdf_value[np.newaxis, :, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :], type)

    # Integrate out the original values resulting from noisy stimulus thetas. We now get a grid of m's for all possible equally spaced
    # points pon the theta grid
    p_mVal_given_val = trapezoid(p_mVal_given_val, stim_val_grid, axis=1)

    # Since the value function is monotonic, each theta gets transformed to the value grid. The prob
    #over values for a theta is the same as the prob over values for the given transformed val for the theta
    # Rearranging thetas such that p_m_given_val is on an increasing value grid always
    p_mVal_given_val = p_mVal_given_val[np.argsort(tools.value_function_ori(tools.stim_ori_grid, type))]

    return p_mVal_given_theta0, p_mVal_given_val


# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = 0):

    # encoded m's for given presented value and value grid (distorted val grid based on function)
    p_mVal_given_theta0, p_mVal_given_val = value_efficient_encoding(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac=line_frac)

    # we need to use the transformed stim vbalues and prior over values now
    safe_value, val_prior = prior_val(type, line_frac = line_frac)

    # Applying bayes rule to get the p(val/m). Combining evidence in representation with the prior over variable of interest
    p_val_given_mVal = p_mVal_given_val*np.array(val_prior)[:, np.newaxis]

    # Normalize with p(m) = p(m|val)*p(val) that we just defined as p(val|m) in above line
    p_val_given_mVal = p_val_given_mVal / trapezoid(p_val_given_mVal, safe_value, axis=0)[np.newaxis,:]

    # Probability of estimating \hat{value} given val0
    p_value_est_given_theta0 = p_mVal_given_theta0[:, np.newaxis, :] * p_val_given_mVal[np.newaxis, ...]

    # Get rid of m
    p_value_est_given_theta0 = trapezoid(p_value_est_given_theta0, rep_val_grid, axis=2)

    p_value_est_given_theta0 /= trapezoid(p_value_est_given_theta0, safe_value, axis=1)[:, np.newaxis]

    return safe_value, p_value_est_given_theta0

def safe_value_dist(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = 0):
    safe_value, safe_prob = value_bayesian_decoding(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = line_frac)
    return safe_value, safe_prob

def risky_value_dist(theta1, kappa_s, kappa_r, sigma_rep, risk_prob, type, line_frac = 0):

    safe_value, safe_prob = value_bayesian_decoding(theta1, kappa_s, kappa_r, sigma_rep, type, line_frac = line_frac)

    risky_value = safe_value*risk_prob
    p_risky = safe_prob/risk_prob

    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(safe_value)

    return safe_value, p_risky

