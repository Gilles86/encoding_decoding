import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

import tools as tools

stim_ori_grid = tools.stim_ori_grid
rep_ori_grid = tools.rep_ori_grid

rep_val_grid = tools.rep_val_grid

# Take input orientation and gives the encoded distribution
def MI_orientation_encoding(theta0, kappa_s, kappa_r, normalize = False):

    theta0 = np.atleast_1d(theta0)
    # theta0 x theta_gen x m_gen
    # Add stimulus noise to get distribution of thetas given theta0 - theta0 x theta_gen
    # theta0 is in stim space while theta
    p_theta_given_theta0 = tools.stimulus_ori_noise(theta0[:, np.newaxis], kappa_s=kappa_s, grid=stim_ori_grid[np.newaxis, :])

    # Add sensory noise to see what ms you get given a theta 0 - theta0 x theta_gen_rep x m_gen
    p_mOri_given_theta = tools.sensory_ori_noise(tools.cdf_ori(stim_ori_grid, stim_ori_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                     grid=rep_ori_grid[np.newaxis, np.newaxis, :])

    # Combine sensory and stimulus noise
    p_mOri_given_theta0 = p_mOri_given_theta * p_theta_given_theta0[..., np.newaxis]
    p_mOri_given_theta0 = trapezoid(p_mOri_given_theta0, stim_ori_grid, axis=1)
    # p_mOri_given_theta0 = np.sum(p_mOri_given_theta0, axis=1)


    # Make a big array that for many thetas gives the probability of observing ms (subject likelihood)
    p_mOri_given_theta = tools.stimulus_ori_noise(stim_ori_grid[:, np.newaxis], kappa_s=kappa_s, grid=stim_ori_grid[np.newaxis, :])[
                          ..., np.newaxis] * \
                      tools.sensory_ori_noise(tools.cdf_ori(stim_ori_grid, stim_ori_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                    grid=rep_ori_grid[np.newaxis, np.newaxis, :])
    p_mOri_given_theta = trapezoid(p_mOri_given_theta, stim_ori_grid, axis=1)
    # p_m_given_theta = np.sum(p_m_given_theta, axis=1)

    if normalize:
        # p_m_given_theta /= trapezoid(p_m_given_theta, stim_grid, axis=1)[:, np.newaxis, :]
        p_mOri_given_theta /= trapezoid(p_mOri_given_theta, rep_ori_grid, axis=1)[:, np.newaxis]

    # The mOri probabilities are on equally sized grids.
    return p_mOri_given_theta0, p_mOri_given_theta

def val_encoded(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = 0):
    # tHESE probabilities are on uniformly spaced rep_val_grid
    p_mOri_given_theta0, p_mOri_given_theta = MI_orientation_encoding(theta0, kappa_s, kappa_r, normalize = False)

    #p_mOri_given_theta0 is 1*rep_ori_grid. Each of the rep_ori_grid creates a noise value representation
    mVal = tools.value_function_ori(rep_ori_grid, type, line_frac = line_frac)
    # 1*uniform rep_ori_grid * uniform rep_val_grid
    p_mVal_given_theta0 = p_mOri_given_theta0[..., np.newaxis] * tools.sensory_val_noise(mVal[np.newaxis,:, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :])
    p_mVal_given_theta0 = trapezoid(p_mVal_given_theta0, rep_ori_grid, axis = 1)

    # Now for all possible thetas
    p_mVal_given_theta = p_mOri_given_theta[..., np.newaxis] * tools.sensory_val_noise(mVal[np.newaxis,:, np.newaxis], sigma_rep, rep_val_grid[np.newaxis, np.newaxis, :])
    # Size now is converted to stim_ori_grid * rep_val_grid, both are uniformly spaced
    p_mVal_given_theta = trapezoid(p_mVal_given_theta, rep_ori_grid, axis = 1)
    # Coverting probability densities to the unequely spaced stim_val_grid space.
    # This next line of code might seem wrong but it is probably right.
    stim_val_grid, p_mVal_given_val = tools.ori_to_val_dist(stim_ori_grid, p_mVal_given_theta, type)
    # mVal is on equally spaced grid but val from p_mVal_given_val is on distorted stim_val_grid
    return p_mVal_given_theta0, p_mVal_given_val

# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = 0.0):

    # Val_encoded gives dists on equally sized grid for the rep_val dimension but the stim_val dimension 
    # is unequally spaced according to stim_val_grid
    p_mVal_given_theta0, p_mVal_given_Val = val_encoded(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = line_frac)
    
    stim_val_grid, prior_val = tools.prior_val(type)
    # stim_val_grid*rep_val_grid
    p_val_given_mVal = p_mVal_given_Val*np.array(prior_val)[:, np.newaxis]

    # Normalize first dimension with stim_val_grid elements coming from prior
    p_val_given_mVal = p_val_given_mVal / abs(trapezoid(p_val_given_mVal, stim_val_grid, axis=0)[np.newaxis,:])

    # theta0 x theta_tilde x m
    p_value_est_given_val0 = p_mVal_given_theta0[:, np.newaxis, :] * p_val_given_mVal[np.newaxis, ...]
    # Get rid of rep val m's which are equally spaced
    p_value_est_given_val0 = abs(trapezoid(p_value_est_given_val0, rep_val_grid, axis=2))

    # normalize (99% sure that not necessary)
    p_value_est_given_val0 /= abs(trapezoid(p_value_est_given_val0, stim_val_grid, axis=1)[:, np.newaxis])

    return stim_val_grid, p_value_est_given_val0

def safe_value_dist(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = 0.0):

    safe_value, safe_prob = value_bayesian_decoding(theta0, kappa_s, kappa_r, sigma_rep, type, line_frac = line_frac)

    return safe_value, safe_prob

def risky_value_dist(theta1, kappa_s, kappa_r, sigma_rep, risk_prob, type, line_frac = 0.0):

    stim_val_grid, p_value_est_given_val0 = safe_value_dist(theta1, kappa_s, kappa_r, sigma_rep, type, line_frac = line_frac)

    risky_value = stim_val_grid*risk_prob
    p_risky = p_value_est_given_val0/risk_prob

    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(stim_val_grid)

    return stim_val_grid, p_risky
