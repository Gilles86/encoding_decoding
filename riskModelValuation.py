import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
# from scipy.stats import gaussian_kde

experiment = "noBoundaryEffects" #"bothSideTruncated" #oneSideTruncated, bothSideTruncated, bothSideFolded

value_rep_bins = 300
ori_bins = 300
stim_space = np.pi * 2.
stim_grid = np.linspace(0, stim_space, ori_bins, True)
# As rep grid is for value representation, same no of points as bins for value
rep_grid = np.linspace(0, 1., 300, True)


# Getting a prior over orientation given a grid (x)
def prior_ori(x):
    if stim_space == np.pi * 2.:
        return (2 - np.abs(np.sin(2 * x))) / (np.pi - 1) / 4.0

    if stim_space == np.pi:
        return (2 - np.abs(np.sin(2 * x))) / (np.pi - 1) / 2.0

    if stim_space == np.pi / 2.:
        return (2 - np.abs(np.sin(2 * x))) / (np.pi - 1) / 1.0

    if stim_space == np.pi / 4.:
        return (2 - np.abs(np.sin(2 * x))) / (np.pi - 1) * 2.0


# Evaluates the number of global minimas in the prior_ori function.
prior_ori_repititions = sum(np.r_[True, prior_ori(stim_grid)[1:] < prior_ori(stim_grid)[:-1]] & np.r_[
    prior_ori(stim_grid)[:-1] < prior_ori(stim_grid)[1:], True])


# Getting a value function given a grid of orientations (stim_grid for example). Always between 0 and 2pi.
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

# A general numerical method is used here to find new distributions of functions of distributions we know.
# Takes in the grid used in original distribution, and the distribution of the original distribution and then
# finds the distribution of the functional value distribution.

def ori_to_val_dist(grid, p, type):
    if p.ndim == 1:
        p_dist = p[np.newaxis, :]
        # grid = grid[np.newaxis, :]
    else:
        p_dist = p
        # grid = grid[np.newaxis, :]
    p_dist = np.array(p_dist)
    grid = np.array(grid)
    # Till here just trying to make it general for different input dimensions.

    # # For every bin in x_stim, calculate the probability mass within that bin
    # # Using ellipsis (...) here since, we will have the grid and the p_mass in the last dimension anyway.
    # dgrid = grid[1:] - grid[:-1]
    # p_mass = ((p_dist[..., 1:] + p_dist[..., :-1]) / 2) * dgrid
    # # Making it the same dimension as we started
    # dgrid = np.append(dgrid, dgrid[1])
    # p_mass = np.append(p_mass, p_mass[..., -1])

    # Get the center of every bin in value functional space.
    grid_value = value_function_ori(grid[:] / 2., type)

    # Sorting values so that one can take means of nearby ones to get rid of issues arising from
    # non-bijective value function
    arranged_valGrid = np.sort(grid_value)
    arrangedIndex = np.argsort(grid_value)
    p_mass = p_dist[..., arrangedIndex]
    # This solves the problem of sstimulusNoise having signal in one quadrant but not in others, (the mean value signal should be taken)
    means_grid = np.mean(arranged_valGrid.reshape(-1, int(ori_range/(np.pi/4.))), axis=1)
    means_prob = np.empty((len(p_dist), len(means_grid))) # Initializing
    # The above two paragraphs have already found the mass in each grid of the original grid and the new grid.
    # We cannot just use that as there are multiple masses in the same grid_value due to non bijectivity of the
    # value function
    ps = []
    for ix in range(len(p_dist)):
        means_prob[ix, :] = np.mean(p_mass.reshape(-1, int(ori_range / (np.pi / 4.))), axis=1)
        val_prior = interpolate.interp1d(np.squeeze(means_grid), means_prob[ix, :])
        val_prior = val_prior(np.squeeze(means_grid))
        val_prior = val_prior / trapezoid(val_prior, means_grid)
        ps.append(val_prior)
    # ps = np.squeeze(np.array(ps))

    return means_grid, ps

# Getting a prior over values given a orientation grid and a type

# Getting a prior over values given a orientation grid and a type
def prior_val(grid, type):

    p_ori = prior_ori(grid)

    bin_centers, ps = ori_to_val_dist(grid, p_ori, type)

    return bin_centers, ps


# The noisy input stimulus being presented
def stimulus_ori_noise(x, sd, grid):
    p_noise_ori = ss.vonmises(loc=x, kappa=1/sd**2).pdf(grid)
    return p_noise_ori

# Getting the noisy input stimulus distribution in value space mapping
# Input parameters define noise in orientation space buy function gives out the noisy distribution in value space
def stimulus_val_noise(x, sd, grid, type):
    # grid = grid[np.newaxis, :]
    # grid = np.array(grid)
    if np.isscalar(x):
        x = np.array([x])
    else:
        x = np.array(x)
    x = x[:, np.newaxis]

    p_noise_ori = ss.vonmises(loc=x, kappa=1/sd**2).pdf(grid)

    bin_centers, ps = ori_to_val_dist(grid, p_noise_ori, type)

    return ps


# This noise will be added to the representation space of the variable being encoded.
# This variable is value in this case
# It should not be a vonmises anymore. In fact what it should be depends on the specific type of experimental setup we
# design in my opinion. If participants are always just showed one quadrant, makes more sense that it is a truncated
# normal in my opinion. hOWEVER, IF IT IS 2 QUADRANTS, SHOULD be a recursive value that repeats back in opposite direction.
def sensory_noise(m, sd, grid):
    if experiment == "bothSideTruncated":
        # This one is when experiment is shown only within a 45degree angle
        return ss.truncnorm.pdf(grid,(0.0 - m) / sd, (1.0 -m) / sd, m, sd)
    if experiment == "noBoundaryEffects":
        # We assume that they represent the information here totally and the bounds do not mean truncation but rather,
        # they jusst shift the boundary values to include 5 standard deviations from the noise, whatever the noise is
        return ss.norm.pdf(grid, m+5*sd*(0.5-m), sd)
    # else if experiment == "oneSideTruncated":
    #     return ss.truncnorm.pdf(grid, (0.0 - m) / sd, (np.inf - m) / sd, m, sd)
        # This one is when experiment is shown only within a 90degree angle


    # return ss.vonmises(loc=m*np.pi*2., kappa=1./(sd*np.pi*2.)**2).pdf(grid*np.pi*2)*np.pi*2.

# Takes in the orientation grid and gives out the cdf over values
def cdf_val(grid, type):
    bin_centers, ps = prior_val(grid, type)
    cdf_val = np.squeeze(integrate.cumtrapz(ps, bin_centers, initial=0.0))
    return cdf_val


# Take input orientation and gives the decoded distribution
def value_efficient_encoding(theta0, sigma_stim, sigma_rep, type):
    # Note that the sigma_stim is in ori space

    theta0 = np.atleast_1d(theta0)
    # Prior over values for the given type of value function mapping
    val_centers, val_prior = prior_val(stim_grid, type)

    # val0 (has same dim as theta0) x theta_gen x m_gen
    # Add stimulus noise to get distribution of values given theta0 - val0 x value_gen
    p_val_given_theta0 = stimulus_val_noise(theta0, sigma_stim, stim_grid, type)
    # Add sensory noise to see what ms for value you get given a value 0 - value_gen_rep x m_gen(rep)
    # For each point in rep space given by the cdf of the value function grid, we add sensory noise constant
    p_m_given_val0 = sensory_noise(cdf_val(theta0, type)[np.newaxis, :, np.newaxis], sigma_rep, rep_grid[np.newaxis, np.newaxis, :])

    # Combine sensory and stimulus noise
    p_m_given_val0 = p_m_given_val0 * p_val_given_theta0[..., np.newaxis]

    # Integrate out different generated values (due to sstim noise), so we just have ms given theta0
    p_m_given_val0 = trapezoid(p_m_given_val0, val_centers, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (value likelihood)
    p_m_given_val = stimulus_val_noise(stim_grid, sigma_stim, stim_grid, type)[..., np.newaxis] * \
        sensory_noise(cdf_val(stim_grid, type)[np.newaxis, :, np.newaxis], sigma_rep, rep_grid[np.newaxis, np.newaxis, :])

    # Integrate out the realized values
    p_m_given_val = trapezoid(p_m_given_val, stim_grid, axis=0)

    # Representations of values
    return p_m_given_val0, p_m_given_val

# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, sigma_stim, sigma_rep, type):

    # There is a one to one correspondence between theta0 and corresponding val0
    # val0 is implicitly presented val by presenting theta0.

    p_m_given_val0, p_m_given_val = value_efficient_encoding(theta0, sigma_stim, sigma_rep, type)
    # just changed this next line from the comebted one
    # p_val_given_m = p_m_given_val*np.array(prior_val(stim_grid, type)[1])[:, np.newaxis]
    # p_val_given_m = p_m_given_val * prior_ori(stim_grid)[:, np.newaxis]

    # Normalize with p(m)
    p_val_given_m = p_val_given_m/ trapezoid(p_val_given_m, stim_grid, axis=0)[np.newaxis, :]

    # theta0 x theta_tilde x m
    # Probability of estimating \hat{theta} given theta0
    p_value_est_given_val0 = p_m_given_val0[:, np.newaxis, :] * p_val_given_m[np.newaxis, ...]

    # Get rid of m
    p_value_est_given_val0 = trapezoid(p_value_est_given_val0, rep_grid, axis=2)

    # normalize (99% sure that not necessary)
    # p_thetaest_given_theta0 /= trapezoid(p_thetaest_given_theta0, stim_grid, axis=1)[:, np.newaxis]

    # Right now I am doing a quick fix to get the answer. This code needs to be changed
    val, p_value_est_given_val0 = ori_to_val_dist(stim_grid, p_value_est_given_val0, type)
    return val, p_value_est_given_val0

def risky_value_dist(theta1, sigma_stim, sigma_rep, risk_prob, type):

    bin_centers, ps = value_bayesian_decoding(theta1, sigma_stim, sigma_rep, type)

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
    # cdf at each point on grid for the second probability distribution array
    # Put safe_prob as p2
    cdf2 = integrate.cumtrapz(p2, grid, initial=0.0, axis=0)


    # for every grid point, distribution 1 is bigger than distribution 2
    # with a probability of being that value times the probability that dist
    # 2 is lower than that value
    prob = p1*cdf2
    p.append(prob)

    # Cummulative probability
    return integrate.trapz(p, grid)