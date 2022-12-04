import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
# from scipy.stats import gaussian_kde


value_rep_bins = 100
stim_grid = np.linspace(0, np.pi*2., 500, True)
# As rep grid is for value representation, same no of points as bins for value
rep_grid = np.linspace(0, 1., 300, True)

# Getting a prior over orientation given a grid (x)
def prior_ori(x):
    return (2 - np.abs(np.sin(2 * x))) / (np.pi - 1) / 4.0

# Getting a value function given a grid of orientations (stim_grid for example)
def value_function_ori(x, type):
    if type == "prior":
        value_function = (12-11*np.abs(np.sin(2*x)))

    if type == "linearPrior":
        value_function = 1+abs(11-abs(11-abs(11-abs(11-abs(11-abs(11-abs(11-x*44/np.pi)))))))

    if type == "curvedPrior":
        value_function = 1+abs(11*np.cos(2*x))

    if type == "inversePrior":
        value_function = 1 + abs(11 * np.sin(2 * x))

    if type == "inverseLinearPrior":
        value_function = 12 - abs(11 - abs(11 - abs(11 - abs(11 - abs(11 - abs(11 - abs(11 - x * 44 / np.pi)))))))

    if type == "inverseCurvedPrior":
        value_function = 12 - abs(11 * np.cos(2 * x))

    return value_function


def ori_to_val(x, p, type):
    x = np.array(x)
    p_dist = p

    assert (x.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"

    # For every bin in x_stim, calculate the probability mass within that bin
    dx = x[..., 1:] - x[..., :-1]
    p_mass = ((p_dist[..., 1:] + p_dist[..., :-1]) / 2) * dx

    # Get the center of every bin
    x_value = value_function_ori(x[:-1] + dx / 2., type)

    ps = []
    h, edges = np.histogram(x_value, bins=value_rep_bins, weights=p_mass, density=True)
    ps.append(h)

    ps = np.squeeze(np.array(ps))
    bin_centers = (edges[1:] + edges[:-1]) / 2

    return bin_centers, ps

# Getting a prior over values given a orientation grid and a type
def prior_val(x, type):

    x = np.array(x)
    p_val = prior_ori(x)

    bin_centers, ps = ori_to_val(x, p_val, type)

    return bin_centers, ps

def cdf_val(x, type):
    bin_centers, ps = prior_val(x, type)
    cdf_val = np.squeeze(integrate.cumtrapz(ps, bin_centers, initial=0.0))
    return cdf_val


def stimulus_val_noise(x, sd, grid, type):
    grid = grid[np.newaxis, :]
    x = x[:, np.newaxis]
    grid = np.array(grid)
    p_noise_ori = ss.vonmises(loc=x, kappa=1/sd**2).pdf(grid)

    # For every bin in x_stim, calculate the probability mass within that bin
    d_grid = grid[..., 1:] - grid[..., :-1]
    p_mass = ((p_noise_ori[..., 1:] + p_noise_ori[..., :-1]) / 2) * d_grid

    # Get the center of every bin
    x_value = value_function_ori(grid[..., :-1] + d_grid / 2., type)

    ps = []
    for ix in range(len(x)):
        h, edges = np.histogram(np.squeeze(x_value), bins=value_rep_bins, weights=p_mass[ix, :], density=True)
        ps.append(h)
    ps = np.squeeze(np.array(ps))
    bin_centers = (edges[1:] + edges[:-1]) / 2

    return ps


def sensory_noise(m, sd, grid):
    return ss.vonmises(loc=m*np.pi*2., kappa=1./(sd*np.pi*2.)**2).pdf(grid*np.pi*2)*np.pi*2.


# Take input orientation and gives the decoded distribution
def value_efficient_encoding(theta0, sigma_stim, sigma_rep, type):

    theta0 = np.atleast_1d(theta0)
    val_centers, val_prior = prior_val(stim_grid, type)

    # theta0 x theta_gen x m_gen
    # Add stimulus noise to get distribution of values given theta0 - theta0 x value_gen
    p_val_given_theta0 = stimulus_val_noise(theta0, sigma_stim, stim_grid, type)
    # Add sensory noise to see what ms for value you get given a theta 0 - theta0 x value_gen_rep x m_gen(rep)
    p_m_given_theta0 = sensory_noise(cdf_val(stim_grid, type)[np.newaxis, :, np.newaxis], sd=sigma_rep,
                                     grid=rep_grid[np.newaxis, np.newaxis, :])

    # Combine sensory and stimulus noise
    p_m_given_theta0 = p_m_given_theta0 * p_val_given_theta0[..., np.newaxis]

    # Integrate out different values, so we just have ms given theta0
    p_m_given_theta0 = trapezoid(p_m_given_theta0, val_centers, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (value likelihood)
    p_m_given_theta = stimulus_val_noise(stim_grid, sigma_stim, stim_grid, type)[..., np.newaxis] * \
        sensory_noise(cdf_val(stim_grid, type)[np.newaxis, :, np.newaxis], sigma_rep, rep_grid[np.newaxis, np.newaxis, :])

    # Integrate out the realized values
    p_m_given_theta = trapezoid(p_m_given_theta, val_centers, axis=1)

    # Representations of values
    return p_m_given_theta0, p_m_given_theta


# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, sigma_stim, sigma_rep, type, bins = value_rep_bins):

    val_centers, val_prior = prior_val(stim_grid, type)
    p_m_given_theta0, p_m_given_theta = value_efficient_encoding(theta0, sigma_stim, sigma_rep, type)
    p_theta_given_m = p_m_given_theta * prior_val[:, np.newaxis]

    # Normalize with p(m)
    p_theta_given_m = p_theta_given_m / trapezoid(p_theta_given_m, stim_grid, axis=0)[np.newaxis, :]

    # theta0 x theta_tilde x m
    # Probability of estimating \hat{theta} given theta0
    p_thetaest_given_theta0 = p_m_given_theta0[:, np.newaxis, :] * p_theta_given_m[np.newaxis, ...]

    # Get rid of m
    p_thetaest_given_theta0 = trapezoid(p_thetaest_given_theta0, rep_grid, axis=2)

    # normalize (99% sure that not necessary)
    # p_thetaest_given_theta0 /= trapezoid(p_thetaest_given_theta0, stim_grid, axis=1)[:, np.newaxis]

    return p_thetaest_given_theta0



def safe_value_dist(theta0, sigma_stim, sigma_rep, type, bins=100, slow=True):

    x_stim = np.array(stim_grid)
    p_stim = bayesian_decoding(theta0, sigma_stim, sigma_rep)
    # p_stim = get_thetahat_dist(theta0, sigma_stim, sigma_rep)

    assert (x_stim.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"

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

    return bin_centers, ps

def risky_value_dist(theta1, sigma_stim, sigma_rep, risk_prob, type, bins=100, slow=True):

    x_stim = np.array(stim_grid)
    p_stim = bayesian_decoding(theta1, sigma_stim, sigma_rep)
    #p_stim = get_thetahat_dist(theta1, sigma_stim, sigma_rep)

    assert (x_stim.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"

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

        risky_value = bin_centers*risk_prob
        p_risky = ps/risk_prob

        p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
        p_risky = p_risky_(bin_centers)

    return bin_centers, p_risky

# Calculate how often distribution 1 is larger than distribution 2
def diff_dist(grid, p1, p2):
    p = []
    # grid: 1d
    # p1/p2: n_orienations x n(grid)
    cdf2 = integrate.cumtrapz(p2, grid, initial=0.0, axis=1)


    # for every grid point, distribution 1 is bigger than distribution 2
    # with a probability of being that value times the probability that dist
    # 2 is lower than that value
    prob = p1*cdf2
    p.append(prob)

    # Cummulative probability
    return integrate.trapz(p, grid)