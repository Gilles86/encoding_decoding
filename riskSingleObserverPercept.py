import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

import tools as tools

# this needs to be set based on how yopu do your experiment
experimentRange = tools.experimentRange #"00to45", "45to90", "90to135", "135to180", "00to90", "90to180", "00to180", "noBoundaryEffects" # For now we predict the sensory noise type to be shaped (trucated, folded whatever)
start = tools.start
end = tools.end

factor_ori = (end-start)/90.*np.pi

stim_ori_grid = tools.stim_ori_grid
rep_ori_grid = tools.rep_ori_grid

max_val = tools.max_val
min_val = tools.min_val

def cdf_ori(x, grid): # goes from 0 to 2pi
    cdf_ori = integrate.cumtrapz(tools.prior_ori(x), grid, initial=0.0)*factor_ori
    return cdf_ori

def stimulus_ori_noise(x, kappa_s, grid):
    # return np.exp(kappa_s*(np.cos(x - grid)-1))
    return ss.vonmises(loc=x, kappa=kappa_s).pdf(grid)

def sensory_ori_noise(m, kappa_r, grid):
    sigma_rep = np.sqrt(1/(2*np.pi*kappa_r))
    if experimentRange == "00to180" or experimentRange == "45to225":
        return ss.vonmises(loc=m, kappa=kappa_r).pdf(grid)
    else:
        truncBoth = ss.truncnorm.pdf(grid,(start*np.pi/90 - m) / sigma_rep, (end*np.pi/90-m) / sigma_rep, m, sigma_rep)
        return truncBoth


# Take input orientation and gives the decoded distribution
def MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize = False):

    theta0 = np.atleast_1d(theta0)
    # theta0 x theta_gen x m_gen
    # Add stimulus noise to get distribution of thetas given theta0 - theta0 x theta_gen
    # theta0 is in stim space while theta
    p_theta_given_theta0 = stimulus_ori_noise(theta0[:, np.newaxis], kappa_s=kappa_s, grid=stim_ori_grid[np.newaxis, :])
    if normalize:
        p_theta_given_theta0 /= trapezoid(p_theta_given_theta0, stim_ori_grid, axis=1)[:, np.newaxis]
    # Add sensory noise to see what ms you get given a theta 0 - theta0 x theta_gen_rep x m_gen
    p_m_given_theta = sensory_ori_noise(cdf_ori(stim_ori_grid, stim_ori_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                     grid=rep_ori_grid[np.newaxis, np.newaxis, :])
    if normalize:
        p_m_given_theta /= trapezoid(p_m_given_theta, rep_ori_grid, axis=2)[:, :, np.newaxis]

    # Combine sensory and stimulus noise
    p_m_given_theta0 = p_m_given_theta * p_theta_given_theta0[..., np.newaxis]

    # Integrate out different thetas, so we just have ms given theta0
    p_m_given_theta0 = trapezoid(p_m_given_theta0, stim_ori_grid, axis=1)
    # p_m_given_theta0 = np.sum(p_m_given_theta0, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (subject likelihood)
    p_m_given_theta = stimulus_ori_noise(stim_ori_grid[:, np.newaxis], kappa_s=kappa_s, grid=stim_ori_grid[np.newaxis, :])[
                          ..., np.newaxis] * \
                      sensory_ori_noise(cdf_ori(stim_ori_grid, stim_ori_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                    grid=rep_ori_grid[np.newaxis, np.newaxis, :])

    # Integrate out the realized thetas
    p_m_given_theta = trapezoid(p_m_given_theta, stim_ori_grid, axis=1)
    # p_m_given_theta = np.sum(p_m_given_theta, axis=1)

    if normalize:
        # p_m_given_theta /= trapezoid(p_m_given_theta, stim_grid, axis=1)[:, np.newaxis, :]
        p_m_given_theta /= trapezoid(p_m_given_theta, rep_ori_grid, axis=1)[:, np.newaxis]

    return p_m_given_theta0, p_m_given_theta

# Take input orientation and gives the decoded distribution
def bayesian_decoding(theta0, kappa_s, kappa_r, normalize = False):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    # Multiply with prior on thetas
    p_theta_given_m = p_m_given_theta * tools.prior_ori(stim_ori_grid)[:, np.newaxis]

    # Normalize with p(m) to get posterior
    # if normalize:
    p_theta_given_m = p_theta_given_m / trapezoid(p_theta_given_m, stim_ori_grid, axis=0)[np.newaxis, :]
    # p_theta_given_m = p_theta_given_m / np.sum(p_theta_given_m, axis=0)[np.newaxis, :]

    # theta0 x theta_tilde x m
    # Probability of estimating \hat{theta} given theta0
    p_thetaest_given_theta0 = p_m_given_theta0[:, np.newaxis, :] * p_theta_given_m[np.newaxis, ...]

    # Get rid of m
    # p_thetaest_given_theta0 = np.sum(p_thetaest_given_theta0, axis=2)
    p_thetaest_given_theta0 = trapezoid(p_thetaest_given_theta0, rep_ori_grid, axis=2)

    # normalize (99% sure that not necessary)
    if normalize:
        p_thetaest_given_theta0 /= trapezoid(p_thetaest_given_theta0, stim_ori_grid, axis=1)[:, np.newaxis]

    return p_thetaest_given_theta0


# Takes in orientation and gives mean decoded orientation
def expected_thetahat_theta0(theta0, kappa_s, kappa_r, normalize = False):
    p_thetaest_given_theta0 = bayesian_decoding(theta0, kappa_s, kappa_r, normalize)
    return np.angle(trapezoid(np.exp(1j*stim_ori_grid[np.newaxis, :])*p_thetaest_given_theta0, stim_ori_grid, axis=1)) % (2*np.pi)
    # return np.angle(np.sum(np.exp(1j*stim_grid[np.newaxis, :])*p_thetaest_given_theta0, axis=1)) % (2*np.pi)


# NOW WEI AND STOCKER code
def wei_theta_m_subject(theta0, kappa_s, kappa_r, normalize = True):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    p_theta_given_m = p_m_given_theta * tools.prior_ori(stim_ori_grid)[:, np.newaxis]
    # Normalize with p(m) to get posterior
    p_theta_given_m = p_theta_given_m / trapezoid(p_theta_given_m, stim_ori_grid, axis=0)[np.newaxis, :]
    # sO FAR  EXACTLY THE SAME CODE AS OURS

    bayes_mean = np.zeros(len(rep_ori_grid))

    for j in range(len(rep_ori_grid)):
        posterior = p_theta_given_m[:, j]

        bayes_mean[j] = np.arctan2(trapezoid(np.sin(stim_ori_grid) * posterior, stim_ori_grid), trapezoid(np.cos(stim_ori_grid) * posterior, stim_ori_grid))
        # bayes_mean[j] = np.arctan2(np.sum(np.sin(stim_grid) * posterior), np.sum(np.cos(stim_grid) * posterior))
        bayes_mean[j] = bayes_mean[j] - np.floor(bayes_mean[j] / (2 * np.pi)) * 2 * np.pi

    return bayes_mean

def wei_bias(theta0, kappa_s, kappa_r, normalize = True):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    bayes_mean = wei_theta_m_subject(theta0, kappa_s, kappa_r, normalize)

    bias_mean = np.zeros(len(stim_ori_grid))

    for i in range(len(stim_ori_grid)):
        tem = p_m_given_theta[i, :] / np.sum(p_m_given_theta[i, :])
        weight = tem #* tools.prior_ori(rep_grid)
        bias_mean[i] = np.arctan2(trapezoid(np.sin(bayes_mean) * weight, rep_ori_grid), trapezoid(np.cos(bayes_mean) * weight, rep_ori_grid))
        # bias_mean[i] = np.arctan2(np.sum(np.sin(bayes_mean) * weight), np.sum(np.cos(bayes_mean) * weight))
        bias_mean[i] = bias_mean[i] - np.floor(bias_mean[i] / (2 * np.pi)) * 2 * np.pi
        bias_mean[i] = bias_mean[i] - stim_ori_grid[i]

    return bias_mean


def safe_value_dist(theta0, kappa_s, kappa_r, type, line_frac = 0.0):

    p_stim = bayesian_decoding(theta0, kappa_s, kappa_r)

    safe_value, safe_prob = tools.ori_to_val_dist(stim_ori_grid, p_stim, type, line_frac = line_frac)

    # nORMALIZATION not necessary 
    # ps /= abs(trapezoid(ps, bin_centers, axis = -1)[:, np.newaxis])

    return safe_value, safe_prob

def risky_value_dist(theta1, kappa_s, kappa_r, risk_prob, type):

    x_value, p_value = safe_value_dist(theta1, kappa_s, kappa_r, type)

    risky_value = x_value*risk_prob
    p_risky = p_value/risk_prob
    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(x_value)

    return x_value, p_risky
