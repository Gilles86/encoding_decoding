import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

import tools as tools
import riskSingleObserverValuation as valuation

stim_ori_grid = tools.stim_ori_grid
rep_ori_grid = tools.rep_ori_grid

rep_val_grid = tools.rep_val_grid

# Take input orientation and gives the encoded distribution
def MI_orientation_encoding(theta0, kappa_s, kappa_r):

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

    return p_mOri_given_theta0

# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, kappa_s, kappa_r, sigma_rep, type):

    # Getting encoded orientation
    p_mOri_given_theta0 = MI_orientation_encoding(theta0, kappa_s, kappa_r)

    # encoded orientation directly maps to values in the subject's brain
    p_mVal_given_theta0 = tools.ori_to_val_dist(rep_ori_grid, p_mOri_given_theta0, type)
    val_estimates = valuation.subject_val_estimate(kappa_s, sigma_rep, type, p_val = p_val)


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
