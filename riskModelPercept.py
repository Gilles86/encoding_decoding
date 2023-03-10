import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

# Note that if the end points are included, then use methods like trapezoid or simpson and not np.sum,
# but if end points are not included, then use np.sum (any binning method)
stim_grid = np.linspace(0, 180, 501) * np.pi / 90
rep_grid = np.linspace(0, 180, 301) * np.pi / 90  # np.linspace(0, 1., 300, True)
# stim_grid = stim_grid[:-1]
# rep_grid = rep_grid[:-1]

max_val = 12
min_val = 1

def prior(x):
    return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 4.0

def cdf(x, grid):
    cdf = integrate.cumtrapz(prior(x), grid, initial=0.0)*np.pi*2.
    return cdf

def stimulus_noise(x, kappa_s, grid):
    # return np.exp(kappa_s*(np.cos(x - grid)-1))
    return ss.vonmises(loc=x, kappa=kappa_s).pdf(grid)

def sensory_noise(m, kappa_r, grid):
    # return np.exp(kappa_r * (np.cos(m - grid) - 1))
    return ss.vonmises(loc=m, kappa=kappa_r).pdf(grid)

# Take input orientation and gives the decoded distribution
def MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize = False):

    theta0 = np.atleast_1d(theta0)
    # theta0 x theta_gen x m_gen
    # Add stimulus noise to get distribution of thetas given theta0 - theta0 x theta_gen
    # theta0 is in stim space while theta
    p_theta_given_theta0 = stimulus_noise(theta0[:, np.newaxis], kappa_s=kappa_s, grid=stim_grid[np.newaxis, :])
    if normalize:
        p_theta_given_theta0 /= trapezoid(p_theta_given_theta0, stim_grid, axis=1)[:, np.newaxis]
    # Add sensory noise to see what ms you get given a theta 0 - theta0 x theta_gen_rep x m_gen
    p_m_given_theta = sensory_noise(cdf(stim_grid, stim_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                     grid=rep_grid[np.newaxis, np.newaxis, :])
    if normalize:
        p_m_given_theta /= trapezoid(p_m_given_theta, rep_grid, axis=2)[:, :, np.newaxis]

    # Combine sensory and stimulus noise
    p_m_given_theta0 = p_m_given_theta * p_theta_given_theta0[..., np.newaxis]

    # Integrate out different thetas, so we just have ms given theta0
    p_m_given_theta0 = trapezoid(p_m_given_theta0, stim_grid, axis=1)
    # p_m_given_theta0 = np.sum(p_m_given_theta0, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (subject likelihood)
    p_m_given_theta = stimulus_noise(stim_grid[:, np.newaxis], kappa_s=kappa_s, grid=stim_grid[np.newaxis, :])[
                          ..., np.newaxis] * \
                      sensory_noise(cdf(stim_grid, stim_grid)[np.newaxis, :, np.newaxis], kappa_r=kappa_r,
                                    grid=rep_grid[np.newaxis, np.newaxis, :])

    # Integrate out the realized thetas
    p_m_given_theta = trapezoid(p_m_given_theta, stim_grid, axis=1)
    # p_m_given_theta = np.sum(p_m_given_theta, axis=1)

    if normalize:
        # p_m_given_theta /= trapezoid(p_m_given_theta, stim_grid, axis=1)[:, np.newaxis, :]
        p_m_given_theta /= trapezoid(p_m_given_theta, rep_grid, axis=1)[:, np.newaxis]

    return p_m_given_theta0, p_m_given_theta

# Take input orientation and gives the decoded distribution
def bayesian_decoding(theta0, kappa_s, kappa_r, normalize = False):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    # Multiply with prior on thetas
    p_theta_given_m = p_m_given_theta * prior(stim_grid)[:, np.newaxis]

    # Normalize with p(m) to get posterior
    # if normalize:
    p_theta_given_m = p_theta_given_m / trapezoid(p_theta_given_m, stim_grid, axis=0)[np.newaxis, :]
    # p_theta_given_m = p_theta_given_m / np.sum(p_theta_given_m, axis=0)[np.newaxis, :]

    # theta0 x theta_tilde x m
    # Probability of estimating \hat{theta} given theta0
    p_thetaest_given_theta0 = p_m_given_theta0[:, np.newaxis, :] * p_theta_given_m[np.newaxis, ...]

    # Get rid of m

    # p_thetaest_given_theta0[2] = circularize(p_thetaest_given_theta0[2], rep_grid)
    # p_thetaest_given_theta0 = simpson(p_thetaest_given_theta0, rep_grid, axis=2)
    # p_thetaest_given_theta0 = np.sum(p_thetaest_given_theta0, axis=2)
    p_thetaest_given_theta0 = trapezoid(p_thetaest_given_theta0, rep_grid, axis=2)

    # normalize (99% sure that not necessary)
    if normalize:
        p_thetaest_given_theta0 /= trapezoid(p_thetaest_given_theta0, stim_grid, axis=1)[:, np.newaxis]

    return p_thetaest_given_theta0


# Takes in orientation and gives mean decoded orientation
def expected_thetahat_theta0(theta0, kappa_s, kappa_r, normalize = False):
    p_thetaest_given_theta0 = bayesian_decoding(theta0, kappa_s, kappa_r, normalize)
    return np.angle(trapezoid(np.exp(1j*stim_grid[np.newaxis, :])*p_thetaest_given_theta0, stim_grid, axis=1)) % (2*np.pi)
    # return np.angle(np.sum(np.exp(1j*stim_grid[np.newaxis, :])*p_thetaest_given_theta0, axis=1)) % (2*np.pi)


# NOW WEI AND STOCKER code
def wei_theta_m_subject(theta0, kappa_s, kappa_r, normalize = True):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    p_theta_given_m = p_m_given_theta * prior(stim_grid)[:, np.newaxis]
    # Normalize with p(m) to get posterior
    p_theta_given_m = p_theta_given_m / trapezoid(p_theta_given_m, stim_grid, axis=0)[np.newaxis, :]
    # sO FAR  EXACTLY THE SAME CODE AS OURS

    bayes_mean = np.zeros(len(rep_grid))

    for j in range(len(rep_grid)):
        posterior = p_theta_given_m[:, j]

        bayes_mean[j] = np.arctan2(trapezoid(np.sin(stim_grid) * posterior, stim_grid), trapezoid(np.cos(stim_grid) * posterior, stim_grid))
        # bayes_mean[j] = np.arctan2(np.sum(np.sin(stim_grid) * posterior), np.sum(np.cos(stim_grid) * posterior))
        bayes_mean[j] = bayes_mean[j] - np.floor(bayes_mean[j] / (2 * np.pi)) * 2 * np.pi

    return bayes_mean

def wei_bias(theta0, kappa_s, kappa_r, normalize = True):
    p_m_given_theta0, p_m_given_theta = MI_efficient_encoding(theta0, kappa_s, kappa_r, normalize=normalize)
    bayes_mean = wei_theta_m_subject(theta0, kappa_s, kappa_r, normalize)

    bias_mean = np.zeros(len(stim_grid))

    for i in range(len(stim_grid)):
        tem = p_m_given_theta[i, :] / np.sum(p_m_given_theta[i, :])
        weight = tem #* prior(rep_grid)
        bias_mean[i] = np.arctan2(trapezoid(np.sin(bayes_mean) * weight, rep_grid), trapezoid(np.cos(bayes_mean) * weight, rep_grid))
        # bias_mean[i] = np.arctan2(np.sum(np.sin(bayes_mean) * weight), np.sum(np.cos(bayes_mean) * weight))
        bias_mean[i] = bias_mean[i] - np.floor(bias_mean[i] / (2 * np.pi)) * 2 * np.pi
        bias_mean[i] = bias_mean[i] - stim_grid[i]

    return bias_mean


def value_function_ori(x, type):
    if type == "prior":
        value_function = (max_val-(max_val-min_val)*np.abs(np.sin(x)))

    if type == "linearPrior":
        value_function = max_val - abs((max_val-min_val) - abs((max_val-min_val) - abs((max_val-min_val) - abs((max_val-min_val) - x * (max_val - min_val)*2 / np.pi))))

    if type == "curvedPrior":
        value_function = min_val+abs((max_val-min_val)*np.cos(x))

    if type == "inversePrior":
        value_function = min_val + abs((max_val-min_val) * np.sin(x))

    if type == "inverseLinearPrior":
        value_function = min_val+abs((max_val-min_val)-abs((max_val-min_val)-abs((max_val-min_val)-abs((max_val-min_val)-x*(max_val - min_val)*2/np.pi))))

    if type == "inverseCurvedPrior":
        value_function = max_val - abs((max_val-min_val) * np.cos(x))

    return value_function


def safe_value_dist(theta0, kappa_s, kappa_r, type, interpolation_kind='linear', bins=100, slow=True):

    # bins = np.linspace(1, max_val, n_bins)
    x_stim = np.array(stim_grid)
    p_stim = bayesian_decoding(theta0, kappa_s, kappa_r)

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

def risky_value_dist(theta1, kappa_s, kappa_r, risk_prob, type, interpolation_kind='linear', bins=100, slow=True):

    x_value, p_value = safe_value_dist(theta1, kappa_s, kappa_r, type, interpolation_kind, bins, slow)

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


def get_rnp(safe_payoff, risky_payoffs, p_chose_risky, risk_prob):

    def get_probit(x, intercept, slope):
        return ss.norm(0.0, 1.0).cdf(intercept + slope*x)

    def cost(xs, ps, intercept, slope):
        return np.sum((get_probit(xs, intercept, slope) - ps)**2)
    
    y = p_chose_risky.ravel()
    x = risky_payoffs.ravel()

    def cost_(pars, *args):
        intercept, slope = pars
        return cost(x, y, intercept, slope)

    result = minimize(cost_, (-safe_payoff/risk_prob, 1.0), method='L-BFGS-B')

    intercept_est, slope_est = result.x

    indifference_point = -intercept_est/slope_est

    rnp = safe_payoff / indifference_point

    return rnp