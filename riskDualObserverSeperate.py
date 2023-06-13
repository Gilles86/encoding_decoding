import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

import tools as tools
import riskSingleObserverPercept as perception
# import riskSingleObserverValuation as valuation

stim_ori_grid = tools.stim_ori_grid
rep_val_grid = tools.rep_val_grid

# Gives the value distribution output coming out of the perceptual system
# This is on the grid stim_value that is non uniform when the transformation
# value function is non linear.
def input_to_val_system(theta0, kappa_s, kappa_r, type, p_per = 8):
    stim_val_grid, val_input = perception.safe_value_dist(theta0, kappa_s, kappa_r, type, p_per = p_per)
    return stim_val_grid, val_input
# note that this input to value system acts as the stimulus noies in a sense

# Take the val_input as the incoming stimulus value noise from the first observer and add an additional 
# representational noise in the value representation to then get the encoded value distribution
# for the first system
def value_efficient_encoding(theta0, kappa_s, kappa_r, sigma_rep, type, p_per = 8):
    theta0 = np.atleast_1d(theta0)
    stim_val_grid, val_input = input_to_val_system(theta0, kappa_s, kappa_r, type, p_per = p_per)

    # The non uniform grid of representation distorted acc to value function
    # and the correspondiong cdf of value probabilties on taht grid
    rep_v_grid, cdf_value = tools.cdf_val(type)

    # Add sensory noise to get m's for given value 0 - value_gen_rep x m_gen(rep)
    # Each presented value gives a distribution in sensory space that is centered around the distorted mean according to the cdf
    # of the encoded variable (cdf_val) and on an equallyt sized representational grid.
    p_mVal_given_val = tools.sensory_val_noise(cdf_value[np.newaxis, :, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :])

    # Combine sensory and stimulus noise
    # For a theta0, you have a distorted stim_val_grid and val_input as probability on that
    # distorted grid. Each point on that distorted grid gives sensory value noise.
    p_mVal_given_theta0 =  val_input[..., np.newaxis] * p_mVal_given_val
    # Integrate out different generated values (due to stim noise), so we just have ms given theta0. The integration is on
    # value grid generated from mapping of orientation to value. The spacing of the grid depends on whether the mapping was
    # linear or non linear
    p_mVal_given_theta0 = trapezoid(p_mVal_given_theta0, stim_val_grid, axis=1)

    # The returned probabilties are on equally spaced rep_val_grid from tools
    return p_mVal_given_theta0

# Each point in rep_val_grid corresponds to val_estimate
def subject_val_estimate(kappa_s, kappa_r, sigma_rep, type, p_val = 2):
    # Efficient coding of value. Pdf of value dictated by the value function
    # and the contextual priors
    stim_val_grid, cdf_value = tools.cdf_val(type)
    # Make a big array that for many thetas gives the probability of observing ms (value likelihood)
    p_mVal_given_theta = (input_to_val_system(tools.stim_ori_grid, kappa_s, kappa_r, type, p_per = 8)[1])[..., np.newaxis] *\
        tools.sensory_val_noise(cdf_value[np.newaxis, :, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :])
    p_mVal_given_theta = trapezoid(p_mVal_given_theta, stim_val_grid, axis=1)

    # Since the value function is monotonic, again simply putting it in increasing order of values.
    # Just increase or decreasse based rearrangement. Nothing else. 
    p_mVal_given_val = p_mVal_given_theta[np.argsort(tools.value_function_ori(tools.stim_ori_grid, type))]

   # The prior_val here is the prior over values constructed from combining the
    # value function and the contextual decoding prior developed during training.
    safe_value, val_prior = tools.prior_val(type)

    # Applying bayes rule to get the p(val/m).
    p_val_given_mVal = p_mVal_given_val*np.array(val_prior)[:, np.newaxis]
    # Normalize with p(m) = p(m|val)*p(val) that we just defined as p(val|m) in above line
    p_val_given_mVal = p_val_given_mVal / trapezoid(p_val_given_mVal, safe_value, axis=0)[np.newaxis,:]

    x0 = trapezoid(safe_value[:, np.newaxis]*p_val_given_mVal, safe_value, axis=0)
    if p_val == 2:
        val_estimates = x0

    else:
        val_estimates = []
        for ix in range(len(x0)):
            cost_function = lambda valest: np.sum(p_val_given_mVal[:, ix] * np.abs(safe_value - valest)**(p_val))
            jacobian = lambda valest: -np.sum(p_val_given_mVal[:, ix] * (p_val * np.abs(safe_value - valest)**(p_val-1)))

            x = minimize(cost_function, x0[ix], method='BFGS', jac=jacobian).x[0]
            val_estimates.append(x)
        
        val_estimates = np.array(val_estimates)

    return val_estimates

# Take input orientation and gives the decoded distribution
def experimenter_val_obs(theta0, kappa_s, kappa_r, sigma_rep, type, p_val = 2, p_per = 8):

    # encoded m's for given presented value and value grid (distorted val grid based on function)
    p_mVal_given_theta0 = value_efficient_encoding(theta0, kappa_s, kappa_r, sigma_rep, type, p_per = p_per)
    val_estimates = subject_val_estimate(kappa_s, kappa_r, sigma_rep, type, p_val=p_val)
    # we need to use the transformed stim values and prior over values now
    safe_value, val_prior = tools.prior_val(type)

    safe_value, p_value_est_given_theta0_grid = tools.prob_transform(rep_val_grid, val_estimates, p_mVal_given_theta0)

    p_value_est_given_theta0_grid /= trapezoid(p_value_est_given_theta0_grid, safe_value, axis=1)[:, np.newaxis]

    return safe_value, p_value_est_given_theta0_grid


def safe_value_dist(theta0, kappa_s, kappa_r, sigma_rep, type, p_val = 2, p_per = 8):
    safe_value, safe_prob = experimenter_val_obs(theta0, kappa_s, kappa_r, sigma_rep, type, p_val = p_val, p_per = p_per)
    return safe_value, safe_prob

def risky_value_dist(theta1, kappa_s, kappa_r, sigma_rep, risk_prob, type, p_val = 2, p_per = 8):

    safe_value, safe_prob = experimenter_val_obs(theta1, kappa_s, kappa_r, sigma_rep, type, p_val = p_val, p_per = p_per)

    risky_value = safe_value*risk_prob
    p_risky = safe_prob/risk_prob

    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(safe_value)

    return safe_value, p_risky








# Earlier implementation 
# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, kappa_s, kappa_r, sigma_rep, type):

    # encoded m's for given presented value and value grid (distorted val grid based on function)
    p_mVal_given_theta0, p_mVal_given_val = value_efficient_encoding(theta0, kappa_s, kappa_r, sigma_rep, type)

    # we need to use the transformed stim values and prior over values now
    safe_value, val_prior = tools.prior_val(type)

    # Applying bayes rule to get the p(val/m). Combining evidence in representation with the prior over variable of interest
    # Note that the p_mVal_given_val also has values that are distorted acc to the function
    # both that and prior_val are on the distoted stim_val_grid or safe_value
    p_val_given_mVal = p_mVal_given_val*np.array(val_prior)[:, np.newaxis]

    # Normalize with p(m) = p(m|val)*p(val) that we just defined as p(val|m) in above line
    p_val_given_mVal = p_val_given_mVal / trapezoid(p_val_given_mVal, safe_value, axis=0)[np.newaxis,:]

    # Probability of estimating \hat{value} given val0, mVals are on uniformly spaced rep_val_grid
    p_value_est_given_theta0 = p_mVal_given_theta0[:, np.newaxis, :] * p_val_given_mVal[np.newaxis, ...]

    # Get rid of m
    p_value_est_given_theta0 = trapezoid(p_value_est_given_theta0, rep_val_grid, axis=2)

    p_value_est_given_theta0 /= trapezoid(p_value_est_given_theta0, safe_value, axis=1)[:, np.newaxis]

    return safe_value, p_value_est_given_theta0