import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

import tools as tools

# experiment = "oneSideTruncated" # "noBoundaryEffects" #"bothSideTruncated" #oneSideTruncated, bothSideTruncated, bothSideFolded
# this needs to be set based on how yopu do your experiment
experimentRange = tools.experimentRange #"00to45", "45to90", "90to135", "135to180", "00to90", "90to180", "00to180", "noBoundaryEffects" # For now we predict the sensory noise type to be shaped (trucated, folded whatever)

start = tools.start
end = tools.end

factor_ori = (end-start)/90.*np.pi

stim_ori_grid = tools.stim_ori_grid
rep_ori_grid = tools.rep_ori_grid

def cdf_ori(x, grid): # goes from 0 to 2pi
    cdf_ori = integrate.cumtrapz(tools.prior_ori(x), grid, initial=0.0)*factor_ori
    return cdf_ori

# The noisy input stimulus being presented
def stimulus_ori_noise(x, kappa_s, grid):
    p_noise_ori = ss.vonmises(loc=x, kappa=kappa_s).pdf(grid)
    return p_noise_ori

def sensory_ori_noise(m, kappa_r, grid):
    sigma_rep = np.sqrt(1/(2*np.pi*kappa_r))
    if experimentRange == "00to180" or experimentRange == "45to225":
        return ss.vonmises(loc=m, kappa=kappa_r).pdf(grid)
    else:
        truncBoth = ss.truncnorm.pdf(grid,(start*np.pi/90 - m) / sigma_rep, (end*np.pi/90-m) / sigma_rep, m, sigma_rep)
        return truncBoth

# Take input orientation and gives the encoded distribution
def MI_orientation_encoding(theta0, kappa_s, kappa_r, normalize = False):

    theta0 = np.atleast_1d(theta0)
    # theta0 x theta_gen x m_gen
    # Add stimulus noise to get distribution of thetas given theta0 - theta0 x theta_gen
    # theta0 is in stim space while theta
    p_theta_given_theta0 = stimulus_ori_noise(theta0[:, np.newaxis], kappa_s=kappa_s, grid=stim_ori_grid[np.newaxis, :])
    if normalize:
        p_theta_given_theta0 /= trapezoid(p_theta_given_theta0, stim_ori_grid, axis=1)[:, np.newaxis]
    # Add sensory noise to see what ms you get given a theta 0 - theta0 x theta_gen_rep x m_gen
    p_mOri_given_theta = sensory_ori_noise(cdf_ori(stim_ori_grid, stim_ori_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                     grid=rep_ori_grid[np.newaxis, np.newaxis, :])
    if normalize:
        p_mOri_given_theta /= trapezoid(p_mOri_given_theta, rep_ori_grid, axis=2)[:, :, np.newaxis]

    # Combine sensory and stimulus noise
    p_mOri_given_theta0 = p_mOri_given_theta * p_theta_given_theta0[..., np.newaxis]

    # Integrate out different thetas, so we just have ms given theta0
    p_mOri_given_theta0 = trapezoid(p_mOri_given_theta0, stim_ori_grid, axis=1)
    # p_mOri_given_theta0 = np.sum(p_mOri_given_theta0, axis=1)

    # dx = stim_ori_grid[..., 1:] - stim_ori_grid[..., :-1]
    # x_stim = stim_ori_grid[:-1] + dx
    # Make a big array that for many thetas gives the probability of observing ms (subject likelihood)
    p_mOri_given_theta = stimulus_ori_noise(stim_ori_grid[:, np.newaxis], kappa_s=kappa_s, grid=stim_ori_grid[np.newaxis, :])[
                          ..., np.newaxis] * \
                      sensory_ori_noise(cdf_ori(stim_ori_grid, stim_ori_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                    grid=rep_ori_grid[np.newaxis, np.newaxis, :])

    # Integrate out the realized thetas
    p_mOri_given_theta = trapezoid(p_mOri_given_theta, stim_ori_grid, axis=1)
    # p_m_given_theta = np.sum(p_m_given_theta, axis=1)

    if normalize:
        # p_m_given_theta /= trapezoid(p_m_given_theta, stim_grid, axis=1)[:, np.newaxis, :]
        p_mOri_given_theta /= trapezoid(p_mOri_given_theta, rep_ori_grid, axis=1)[:, np.newaxis]

    return p_mOri_given_theta0, p_mOri_given_theta


# Getting a prior over values given a orientation grid and a type
def prior_val(type):
    p_ori = tools.prior_ori(stim_ori_grid)
    stim_val_grid, ps = tools.ori_to_val_dist(stim_ori_grid, p_ori, type)
    ps = np.squeeze(ps) # Brings it back to 1 dime
    return stim_val_grid, ps


def val_encoded(theta0, kappa_s, kappa_r, type):

    p_mOri_given_theta0, p_mOri_given_theta = MI_orientation_encoding(theta0, kappa_s, kappa_r, normalize = False)

    rep_val_grid, p_mVal_given_Val0 = tools.ori_to_val_dist(rep_ori_grid, p_mOri_given_theta0, type)
 
    rep_val_grid, p_mVal_given_Val = tools.ori_to_val_dist(rep_ori_grid, p_mOri_given_theta, type)

    return rep_val_grid, p_mVal_given_Val0, p_mVal_given_Val

# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, kappa_s, kappa_r, type):

    rep_val_grid, p_mVal_given_Val0, p_mVal_given_Val = val_encoded(theta0, kappa_s, kappa_r, type)
    
    # stim_val_grid*rep_val_grid
    p_val_given_mVal = p_mVal_given_Val*np.array(prior_val(type)[1])[:, np.newaxis]

    stim_val_grid = prior_val(type)[0]
    # Normalizefirst dimension with stim_val_grid elements coming from prior
    p_val_given_mVal = p_val_given_mVal / abs(trapezoid(p_val_given_mVal, stim_val_grid, axis=0)[np.newaxis,:])

    # theta0 x theta_tilde x m
    p_value_est_given_val0 = p_mVal_given_Val0[:, np.newaxis, :] * p_val_given_mVal[np.newaxis, ...]
    # Get rid of m
    p_value_est_given_val0 = abs(trapezoid(p_value_est_given_val0, rep_val_grid, axis=2))

    # normalize (99% sure that not necessary)
    p_value_est_given_val0 /= abs(trapezoid(p_value_est_given_val0, stim_val_grid, axis=1)[:, np.newaxis])
    return rep_val_grid, stim_val_grid, p_value_est_given_val0

def risky_value_dist(theta1, kappa_s, sigma_rep, risk_prob, type, interpolation_kind='linear', bins=bins, monotonic=True):

    rep_val_grid, stim_val_grid, p_value_est_given_val0 = value_bayesian_decoding(theta1, kappa_s, sigma_rep, type, interpolation_kind=interpolation_kind, bins=bins, monotonic=monotonic)

    risky_value = stim_val_grid*risk_prob
    p_risky = p_value_est_given_val0/risk_prob

    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(stim_val_grid)

    return stim_val_grid, p_risky
