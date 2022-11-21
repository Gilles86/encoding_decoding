import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid


stim_grid = np.linspace(0, np.pi*2., 500, True)
rep_grid = np.linspace(0, 1., 500, True)

def prior(x):
    return (2 - np.abs(np.sin(2 * x))) / (np.pi - 1) / 4.0

def cdf(x):
    cdf = integrate.cumtrapz(prior(x), stim_grid, initial=0.0)
    return cdf

def stimulus_noise(x, sd, grid):
    return ss.vonmises(loc=x, kappa=1/sd**2).pdf(grid)

def sensory_noise(m, sd, grid):
    return ss.vonmises(loc=m*np.pi*2., kappa=1./(sd*np.pi*2.)**2).pdf(grid*np.pi*2)*np.pi*2.

# Take input orientation and gives the decoded distribution
def get_thetahat_dist(theta0, sigma_stim, sigma_rep):

    theta0 = np.atleast_1d(theta0)
    # theta0 x theta_gen x m_gen
    # Add stimulus noise to get distribution of thetas given theta0 - theta0 x theta_gen
    p_theta_given_theta0 = stimulus_noise(theta0[:, np.newaxis], sd=sigma_stim, grid=stim_grid[np.newaxis, :])
    # Add sensory noise to see what ms you get given a theta 0 - theta0 x theta_gen_rep x m_gen
    p_m_given_theta0 = sensory_noise(cdf(stim_grid)[np.newaxis, :, np.newaxis], sd=sigma_rep,
                                     grid=rep_grid[np.newaxis, np.newaxis, :])

    # Combine sensory and stimulus noise
    p_m_given_theta0 = p_m_given_theta0 * p_theta_given_theta0[..., np.newaxis]

    # Integrate out different thetas, so we just have ms given theta0
    p_m_given_theta0 = trapezoid(p_m_given_theta0, stim_grid, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (subject likelihood)
    p_m_given_theta = stimulus_noise(stim_grid[:, np.newaxis], sd=sigma_stim, grid=stim_grid[np.newaxis, :])[
                          ..., np.newaxis] * \
                      sensory_noise(cdf(stim_grid)[np.newaxis, :, np.newaxis], sd=sigma_rep,
                                    grid=rep_grid[np.newaxis, np.newaxis, :])

    # Integrate out the realized thetas
    p_m_given_theta = trapezoid(p_m_given_theta, stim_grid, axis=1)

    # Multiply with prior on thetas
    p_theta_given_m = p_m_given_theta * prior(stim_grid)[:, np.newaxis]

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

# Takes in orientation and gives mean decoded orientation
def expected_thetahat_theta0(theta0, sigma_stim, sigma_rep):

    p_thetaest_given_theta0 = get_thetahat_dist(theta0, sigma_stim, sigma_rep)

    return np.angle(trapezoid(np.exp(1j*stim_grid[np.newaxis, :])*p_thetaest_given_theta0, stim_grid, axis=1)) % (2*np.pi)


def value_function1(x):
    return (20-18*np.abs(np.sin(2*x)))

def value_function2(x):
    return 2+abs(18-abs(18-abs(18-abs(18-abs(18-abs(18-abs(18-x*72/np.pi)))))))

def value_function3(x):
    return 2+abs(18*np.cos(2*x))


def value_function4(x):
    return 2 + abs(18 * np.sin(2 * x))

def value_function5(x):
    return 20 - abs(18 - abs(18 - abs(18 - abs(18 - abs(18 - abs(18 - abs(18 - x * 72 / np.pi)))))))

def value_function6(x):
    return 20 - abs(18 * np.cos(2 * x))


def safe_value_dist(theta0, sigma_stim, sigma_rep, value_function, bins=20, slow=True):

    x_stim = np.array(stim_grid)
    p_stim = get_thetahat_dist(theta0, sigma_stim, sigma_rep)

    assert (x_stim.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"

    # For every bin in x_stim, calculate the probability mass within that bin
    dx = x_stim[..., 1:] - x_stim[..., :-1]
    p_mass = ((p_stim[..., 1:] + p_stim[..., :-1]) / 2) * dx

    # Get the center of every bin
    x_value = value_function(x_stim[:-1] + dx / 2.)

    if slow:
        ps = []
        for ix in range(len(p_stim)):
            h, edges = np.histogram(x_value, bins=bins, weights=p_mass[ix], density=True)
            ps.append(h)

        ps = np.array(ps)
        bin_centers = (edges[1:] + edges[:-1]) / 2

    return bin_centers, ps

def risky_value_dist(theta1, sigma_stim, sigma_rep, value_function, risk_prob, bins=20, slow=True):

    x_stim = np.array(stim_grid)
    p_stim = get_thetahat_dist(theta1, sigma_stim, sigma_rep)

    assert (x_stim.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"

    # For every bin in x_stim, calculate the probability mass within that bin
    dx = x_stim[..., 1:] - x_stim[..., :-1]
    p_mass = ((p_stim[..., 1:] + p_stim[..., :-1]) / 2) * dx

    # Get the center of every bin
    x_value = value_function(x_stim[:-1] + dx / 2.)

    if slow:
        ps = []
        for ix in range(len(p_stim)):
            h, edges = np.histogram(x_value, bins=bins, weights=p_mass[ix], density=True)
            ps.append(h)

        ps = np.array(ps)
        bin_centers = (edges[1:] + edges[:-1]) / 2

        risky_value = bin_centers*risk_prob
        p_risky = ps/risk_prob

        p_risky_ = interpolate.interp1d(risky_value, p_risky, fill_value='extrapolate', bounds_error=False)
        p_risky = p_risky_(bin_centers)

    return bin_centers, p_risky





