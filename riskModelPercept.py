import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid

# We need 1500 grid points in stim_grid to see behavior properly for small noise regimes that we are interested in
stim_grid = np.linspace(0, np.pi*2., 500, True)
rep_grid = np.linspace(0, 1., 300, True)

max_val = 81
min_val = 1

def prior(x):
    return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 4.0

def cdf(x):
    cdf = integrate.cumtrapz(prior(x), stim_grid, initial=0.0)
    return cdf

def stimulus_noise(x, sd, grid):
    return ss.vonmises(loc=x, kappa=1/sd**2).pdf(grid)

def sensory_noise(m, sd, grid):
    return ss.vonmises(loc=m*np.pi*2., kappa=1./(sd*np.pi*2.)**2).pdf(grid*np.pi*2)*np.pi*2.

# Take input orientation and gives the decoded distribution
def MI_efficient_encoding(theta0, sigma_stim, sigma_rep, normalize=True):

    theta0 = np.atleast_1d(theta0)
    # theta0 x theta_gen x m_gen
    # Add stimulus noise to get distribution of thetas given theta0 - theta0 x theta_gen
    p_theta_given_theta0 = stimulus_noise(theta0[:, np.newaxis], sd=sigma_stim, grid=stim_grid[np.newaxis, :])

    if normalize:
        p_theta_given_theta0 /= trapezoid(p_theta_given_theta0, axis=1)[:, np.newaxis]

    # Add sensory noise to see what ms you get given a theta 0 - theta0 x theta_gen_rep x m_gen
    p_m_given_theta0 = sensory_noise(cdf(stim_grid)[np.newaxis, :, np.newaxis], sd=sigma_rep,
                                     grid=rep_grid[np.newaxis, np.newaxis, :])
    if normalize:
        p_m_given_theta0 /= trapezoid(p_m_given_theta0, axis=2)[:, :, np.newaxis]

    # Combine sensory and stimulus noise
    p_m_given_theta0 = p_m_given_theta0 * p_theta_given_theta0[..., np.newaxis]

    # Integrate out different thetas, so we just have ms given theta0
    p_m_given_theta0 = trapezoid(p_m_given_theta0, stim_grid, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (subject likelihood)
    p_m_given_theta = stimulus_noise(stim_grid[:, np.newaxis], sd=sigma_stim, grid=stim_grid[np.newaxis, :])[
                          ..., np.newaxis] * \
                      sensory_noise(cdf(stim_grid)[np.newaxis, :, np.newaxis], sd=sigma_rep,
                                    grid=rep_grid[np.newaxis, np.newaxis, :])

    if normalize:
        p_m_given_theta /= trapezoid(p_m_given_theta, axis=1)[:, np.newaxis, :]
        p_m_given_theta /= trapezoid(p_m_given_theta, axis=2)[:, :, np.newaxis]

    # Integrate out the realized thetas
    p_m_given_theta = trapezoid(p_m_given_theta, stim_grid, axis=1)

    return p_m_given_theta0, p_m_given_theta

# Take input orientation and gives the decoded distribution
def bayesian_decoding(theta0, sigma_stim, sigma_rep, normalize=True):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, sigma_stim, sigma_rep, normalize=normalize)
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
    p_thetaest_given_theta0 = bayesian_decoding(theta0, sigma_stim, sigma_rep)
    #p_thetaest_given_theta0 = get_thetahat_dist(theta0, sigma_stim, sigma_rep)

    return np.angle(trapezoid(np.exp(1j*stim_grid[np.newaxis, :])*p_thetaest_given_theta0, stim_grid, axis=1)) % (2*np.pi)

def value_function_ori(x, type):
    if type == "prior":
        value_function = (max_val-(max_val-min_val)*np.abs(np.sin(2*x)))

    if type == "linearPrior":
        value_function = min_val+abs((max_val-min_val)-abs((max_val-min_val)-abs((max_val-min_val)-abs((max_val-min_val)-abs((max_val-min_val)-abs((max_val-min_val)-abs((max_val-min_val)-x*(max_val - min_val)*4/np.pi)))))))

    if type == "curvedPrior":
        value_function = min_val+abs((max_val-min_val)*np.cos(2*x))

    if type == "inversePrior":
        value_function = min_val + abs((max_val-min_val) * np.sin(2 * x))

    if type == "inverseLinearPrior":
        value_function = max_val - abs((max_val-min_val) - abs((max_val-min_val) - abs((max_val-min_val) - abs((max_val-min_val) - abs((max_val-min_val) - abs((max_val-min_val) - abs((max_val-min_val) - x * (max_val - min_val)*4 / np.pi)))))))

    if type == "inverseCurvedPrior":
        value_function = max_val - abs((max_val-min_val) * np.cos(2 * x))

    return value_function


def safe_value_dist(theta0, sigma_stim, sigma_rep, type, interpolation_kind='linear', bins=100, slow=True):


    # bins = np.linspace(1, max_val, n_bins)
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

        # return bin_centers, ps

        f = interpolate.interp1d(bin_centers, ps, axis=1,
                                 kind=interpolation_kind, fill_value='extrapolate')

        ps = f(edges)

        ps /= np.trapz(ps, edges, axis=1)[:, np.newaxis]

    return edges, ps

def risky_value_dist(theta1, sigma_stim, sigma_rep, risk_prob, type, interpolation_kind='linear', bins=100, slow=True):

    x_value, p_value = safe_value_dist(theta1, sigma_stim, sigma_rep, type, interpolation_kind, bins, slow)

    risky_value = x_value*risk_prob
    p_risky = p_value/risk_prob
    p_risky_ = interpolate.interp1d(risky_value, p_risky, bounds_error=False, fill_value=0)
    p_risky = p_risky_(x_value)

    return risky_value, p_risky

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