import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
# from scipy.stats import gaussian_kde

import tools as tools

stim_ori_grid = tools.stim_ori_grid
rep_val_grid = tools.rep_val_grid

# Take input orientation and gives the encoded distribution over value representation
def value_efficient_encoding(theta0, kappa_s, sigma_rep, type):
    # Note that the kappa_s is in ori space
    theta0 = np.atleast_1d(theta0)

    # val0 (has same dim as theta0) x theta_gen x m_gen
    # Stimulus noise in theta domain simply gets transformed to the stimulus noise distribution in the new domain.
    # Distortions are introduced through the value mapping non linearty.
    stim_val_grid, p_stimVal_given_theta0 = tools.stimulus_val_noise(theta0, kappa_s, stim_ori_grid, type)

    # Efficient coding of value. Pdf of value dictated by the value function and the contextual priors
    stim_val_grid, cdf_value = tools.cdf_val(type)

    # Add sensory noise to see what ms for value you get given a value 0 - value_gen_rep x m_gen(rep)
    # Each presented value gives a distribution in sensory space that is centered around the distorted mean according to the cdf
    # of the encoded variable (cdf_val) and on an equallyt sized representational grid.
    p_m_given_val = tools.sensory_val_noise(cdf_value[np.newaxis, :, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :])

    # Combine sensory and stimulus noise
    p_m_given_theta0 = p_m_given_val * p_stimVal_given_theta0[..., np.newaxis]
    p_m_given_theta0 = trapezoid(p_m_given_theta0, stim_val_grid, axis=1)

    return p_m_given_theta0

# Each point in rep_val_grid corresponds to val_estimate
def subject_val_estimate(kappa_s, sigma_rep, type, p_val = 2):
    # Efficient coding of value. Pdf of value dictated by the value function. 
    stim_val_grid, cdf_value = tools.cdf_val(type)
    # Make a big array that for many thetas (its actually values, varible not named 100 % right) gives the probability of observing ms (value likelihood)
    p_m_given_theta = (tools.stimulus_val_noise(stim_ori_grid, kappa_s, stim_ori_grid, type)[1])[..., np.newaxis] *\
        tools.sensory_val_noise(cdf_value[np.newaxis, :, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :])
    p_m_given_theta = trapezoid(p_m_given_theta, stim_val_grid, axis=1)

    # Since the value function is monotonic, we simply just put it in an increasing grid here
    # Read the above comment, the above already was p_m_given_valjust named wrong
    p_m_given_val = p_m_given_theta[np.argsort(tools.value_function_ori(tools.stim_ori_grid, type))]

   # The prior_val here is the prior over values constructed from combining the
    # value function and the contextual decoding prior developed in orienattion 
    # perception of the experiment.
    safe_value, val_prior = tools.prior_val(type)

    # Applying bayes rule to get the p(val/m). Combining evidence in representation with the prior over variable of interest.
    # p_m_given_val is vals from non unoform stim_val_grid and also the priors are over that non unofrkm grid.
    # So the below multiplication can be performed
    p_val_given_m = p_m_given_val*np.array(val_prior)[:, np.newaxis]
    # Normalize with p(m) = p(m|val)*p(val) that we just defined as p(val|m) in above line
    p_val_given_m = p_val_given_m / trapezoid(p_val_given_m, safe_value, axis=0)[np.newaxis,:]

    x0 = trapezoid(safe_value[:, np.newaxis]*p_val_given_m, safe_value, axis=0)
    if p_val == 2:
        val_estimates = x0

    else:
        val_estimates = []
        for ix in range(len(x0)):
            cost_function = lambda valest: np.sum(p_val_given_m[:, ix] * np.abs(safe_value - valest)**(p_val))
            jacobian = lambda valest: -np.sum(p_val_given_m[:, ix] * (p_val * np.abs(safe_value - valest)**(p_val-1)))

            x = minimize(cost_function, x0[ix], method='BFGS', jac=jacobian).x[0]
            val_estimates.append(x)
        
        val_estimates = np.array(val_estimates)

    return val_estimates

# Take input orientation and gives the decoded distribution
def experimenter_val_obs(theta0, kappa_s, sigma_rep, type, p_val = 2):

    # encoded m's for given presented value and value grid (distorted val grid based on function)
    p_m_given_theta0 = value_efficient_encoding(theta0, kappa_s, sigma_rep, type)

    # The prior_val here is the prior over values constructed from combining the
    # value function and the contextual decoding prior developed in orienattion 
    # perception of the experiment.
    safe_value, val_prior = tools.prior_val(type)

    val_estimates = subject_val_estimate(kappa_s, sigma_rep, type, p_val = p_val)

    safe_value, p_value_est_given_theta0_grid = tools.prob_transform(rep_val_grid, val_estimates, p_m_given_theta0)
    
    return safe_value, p_value_est_given_theta0_grid

def safe_value_dist(theta0, kappa_s, sigma_rep, type, p_val = 2):
    safe_value, safe_prob = experimenter_val_obs(theta0, kappa_s, sigma_rep, type, p_val = p_val)
    return safe_value, safe_prob

def risky_value_dist(theta1, kappa_s, sigma_rep, risk_prob, type, p_val = 2):

    safe_value, safe_prob = experimenter_val_obs(theta1, kappa_s, sigma_rep, type, p_val = p_val)

    risky_value = safe_value*risk_prob
    p_risky = safe_prob/risk_prob

    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(safe_value)

    return safe_value, p_risky





### Earlier code without los sfunction implementation 

# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, kappa_s, sigma_rep, type):

    # encoded m's for given presented value and value grid (distorted val grid based on function)
    p_m_given_theta0, p_m_given_val = value_efficient_encoding(theta0, kappa_s, sigma_rep, type)

    # The prior_val here is the prior over values constructed from combining the
    # value function and the contextual decoding prior developed in orienattion 
    # perception of the experiment.
    safe_value, val_prior = tools.prior_val(type)

    # Applying bayes rule to get the p(val/m). Combining evidence in representation with the prior over variable of interest.
    # p_m_given_val is vals from non unoform stim_val_grid and also the priors are over that non unofrkm grid.
    # So the below multiplication can be performed
    p_val_given_m = p_m_given_val*np.array(val_prior)[:, np.newaxis]

    # Normalize with p(m) = p(m|val)*p(val) that we just defined as p(val|m) in above line
    p_val_given_m = p_val_given_m / trapezoid(p_val_given_m, safe_value, axis=0)[np.newaxis,:]

    # Probability of estimating \hat{value} given val0
    p_value_est_given_theta0 = p_m_given_theta0[:, np.newaxis, :] * p_val_given_m[np.newaxis, ...]

    # Get rid of m
    p_value_est_given_theta0 = trapezoid(p_value_est_given_theta0, rep_val_grid, axis=2)

    p_value_est_given_theta0 /= trapezoid(p_value_est_given_theta0, safe_value, axis=1)[:, np.newaxis]

    return safe_value, p_value_est_given_theta0