import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

import tools as tools

stim_ori_grid = tools.stim_ori_grid
rep_ori_grid = tools.rep_ori_grid

# Take input orientation and gives the encoded distribution
def MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize = False):
    theta0 = np.atleast_1d(theta0)
    
    # theta0 x theta_gen x m_gen
    # Add stimulus noise to get distribution of thetas given theta0 - theta0 x theta_gen
    p_theta_given_theta0 = tools.stimulus_ori_noise(theta0[:, np.newaxis], kappa_s=kappa_s, grid=stim_ori_grid[np.newaxis, :])

    # Add sensory noise to see what ms you get given a theta 0 - theta_gen_rep x m_gen
    p_m_given_theta = tools.sensory_ori_noise(tools.cdf_ori(stim_ori_grid, stim_ori_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                     grid=rep_ori_grid[np.newaxis, np.newaxis, :])

    # Combine sensory and stimulus noise - theta0 x theta_gen_rep x m_gen
    p_m_given_theta0 = p_m_given_theta * p_theta_given_theta0[..., np.newaxis]
    p_m_given_theta0 = trapezoid(p_m_given_theta0, stim_ori_grid, axis=1)
    # p_m_given_theta0 = np.sum(p_m_given_theta0, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (subject likelihood)
    p_m_given_theta = tools.stimulus_ori_noise(stim_ori_grid[:, np.newaxis], kappa_s=kappa_s, grid=stim_ori_grid[np.newaxis, :])[
                          ..., np.newaxis] * \
                      tools.sensory_ori_noise(tools.cdf_ori(stim_ori_grid, stim_ori_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                    grid=rep_ori_grid[np.newaxis, np.newaxis, :])
    p_m_given_theta = trapezoid(p_m_given_theta, stim_ori_grid, axis=1)
    # p_m_given_theta = np.sum(p_m_given_theta, axis=1)

    if normalize:
        # p_m_given_theta /= trapezoid(p_m_given_theta, stim_grid, axis=1)[:, np.newaxis, :]
        p_m_given_theta /= trapezoid(p_m_given_theta, rep_ori_grid, axis=1)[:, np.newaxis]

    return p_m_given_theta0, p_m_given_theta

# Take input orientation and gives the decoded distribution in original stim_ori_grid equal spaced grid
def bayesian_decoding(theta0, kappa_s, kappa_r, normalize = False):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    # Multiply with contextual prior on thetas to get posterior
    p_theta_given_m = p_m_given_theta * tools.context_prior_ori(stim_ori_grid)[:, np.newaxis]
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

def bayesian_decoding_p(theta0, kappa_s, kappa_r, p_exp=2, normalize = False):

    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    
    # theta x m
    p_theta_given_m = p_m_given_theta * tools.context_prior_ori(stim_ori_grid)[:, np.newaxis]
    p_theta_given_m = p_theta_given_m / trapezoid(p_theta_given_m, stim_ori_grid, axis=0)[np.newaxis, :]

    stim_ori_grid_complex = np.exp(1j*stim_ori_grid)
    x0 = np.angle(trapezoid(stim_ori_grid_complex[:, np.newaxis]*p_theta_given_m, stim_ori_grid, axis=0))  % (2*np.pi)
    if p_exp == 2:
        theta_estimates = x0
    else:
        # cost_function = lambda thetaest: np.sum(p_theta_given_m * (1 - np.cos(stim_ori_grid[:, np.newaxis] - thetaest[np.newaxis, :]))**(p_exp/2))
        # jacobian = lambda thetaest:  -np.diag(np.sum(p_theta_given_m * (.5 * p_exp * np.sin(stim_ori_grid[:, np.newaxis] - thetaest[np.newaxis, :]) * (1 - np.cos(stim_ori_grid[:, np.newaxis] - thetaest[np.newaxis, :]))**(p_exp/2 - 1)), 0))

        theta_estimates = []
        for ix in range(len(x0)):
            cost_function = lambda thetaest: np.sum(p_theta_given_m[:, ix] * (1 - np.cos(stim_ori_grid - thetaest))**(p_exp/2))
            jacobian = lambda thetaest: -np.sum(p_theta_given_m[:, ix] * (.5 * p_exp * np.sin(stim_ori_grid - thetaest) * (1 - np.cos(stim_ori_grid - thetaest))**(p_exp/2 - 1)))

            x = minimize(cost_function, x0[ix], method='BFGS', jac=jacobian).x[0]
            theta_estimates.append(x)
        
        theta_estimates = np.array(theta_estimates)

    problem_ix = np.where(np.diff(theta_estimates) < 0.0)
    # problem_ix = np.where(theta_estimates[0:10] > np.pi)
    theta_estimates[problem_ix] = theta_estimates[problem_ix] - 2*np.pi

    grad_val = np.abs(np.gradient(theta_estimates, rep_ori_grid))

    p_thetaest_given_theta0 = p_m_given_theta0 / grad_val[np.newaxis, :]

    p_thetaest_given_theta0 = np.array([np.interp(stim_ori_grid, theta_estimates, p_thetaest_given_theta0[ix], period=2*np.pi) for ix in range(len(p_thetaest_given_theta0))])

    return p_thetaest_given_theta0
    

# Takes in orientation and gives mean decoded orientation
def expected_thetahat_theta0(theta0, kappa_s, kappa_r, normalize = False):
    p_thetaest_given_theta0 = bayesian_decoding(theta0, kappa_s, kappa_r, normalize)
    return np.angle(trapezoid(np.exp(1j*stim_ori_grid[np.newaxis, :])*p_thetaest_given_theta0, stim_ori_grid, axis=1)) % (2*np.pi)

def expected_thetahat_theta0_p(theta0, kappa_s, kappa_r, p_exp=2., normalize = False):
    p_thetaest_given_theta0 = bayesian_decoding_p(theta0, kappa_s, kappa_r, p_exp=p_exp, normalize=normalize)

    return np.angle(trapezoid(np.exp(1j*stim_ori_grid[np.newaxis, :])*p_thetaest_given_theta0, stim_ori_grid, axis=1)) % (2*np.pi)
    # return np.angle(np.sum(np.exp(1j*stim_grid[np.newaxis, :])*p_thetaest_given_theta0, axis=1)) % (2*np.pi)

# NOW WEI AND STOCKER code
def wei_theta_m_subject(theta0, kappa_s, kappa_r, normalize = True):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    p_theta_given_m = p_m_given_theta * tools.context_prior_ori(stim_ori_grid)[:, np.newaxis]
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
        weight = tem #* tools.context_prior_ori(rep_grid)
        bias_mean[i] = np.arctan2(trapezoid(np.sin(bayes_mean) * weight, rep_ori_grid), trapezoid(np.cos(bayes_mean) * weight, rep_ori_grid))
        # bias_mean[i] = np.arctan2(np.sum(np.sin(bayes_mean) * weight), np.sum(np.cos(bayes_mean) * weight))
        bias_mean[i] = bias_mean[i] - np.floor(bias_mean[i] / (2 * np.pi)) * 2 * np.pi
        bias_mean[i] = bias_mean[i] - stim_ori_grid[i]

    return bias_mean

# Gives new value grid and probability on this new value grid
def safe_value_dist(theta0, kappa_s, kappa_r, type, p_exp = 2.0, line_frac = 0.0):

    p_stim = bayesian_decoding_p(theta0, kappa_s, kappa_r, p_exp = p_exp)

    safe_value, safe_prob = tools.ori_to_val_dist(stim_ori_grid, p_stim, type, line_frac = line_frac)

    safe_prob /= abs(trapezoid(safe_prob, safe_value, axis = -1)[:, np.newaxis])

    return safe_value, safe_prob

def risky_value_dist(theta1, kappa_s, kappa_r, risk_prob, type, p_exp = 2.0, line_frac = 0.0):

    x_value, p_value = safe_value_dist(theta1, kappa_s, kappa_r, type, p_exp = p_exp, line_frac = line_frac)

    risky_value = x_value*risk_prob
    p_risky = p_value/risk_prob
    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(x_value)

    return x_value, p_risky
