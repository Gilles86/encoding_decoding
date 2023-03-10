import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
# from scipy.stats import gaussian_kde


# experiment = "oneSideTruncated" # "noBoundaryEffects" #"bothSideTruncated" #oneSideTruncated, bothSideTruncated, bothSideFolded
experimentRange = "0to90" #"0to45", "45to90", "90to135", "135to180", "0to90", "90to180", "0to180", "noBoundaryEffects" # For now we predict the sensory noise type to be shaped (trucated, folded whatever)
#depedning on the angles in which experiment is presented.

stim_grid = np.linspace(0, 180, 501) * np.pi / 90
rep_grid = np.linspace(0, 1, 301)  # value based representation in this code

max_val = 12
min_val = 1


# Getting a prior over orientation given a grid (x)
def prior_ori(x):
    return (2 - np.abs(np.sin(x))) / (np.pi - 1) / 4.0


# Evaluates the number of global minimas in the prior_ori function.
prior_ori_repititions = sum(np.r_[True, prior_ori(stim_grid)[1:] < prior_ori(stim_grid)[:-1]] & np.r_[
    prior_ori(stim_grid)[:-1] < prior_ori(stim_grid)[1:], True])


# Getting a value function given a grid of orientations (stim_grid for example). Always between 0 and 2pi.
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


# A general numerical method is used here to find new distributions of functions of distributions we know.
# Takes in the grid used in original distribution, and the distribution of the original distribution and then
# finds the distribution of the functional value distribution.
def ori_to_val_dist(grid, p, type, interpolation_kind='linear', bins=25, slow=True):
    x_stim = np.array(grid)
    p_stim = p

    assert (x_stim.ndim == 1), "x_stim should have only one dimension (same grid for all p_stims)"
    if p_stim.ndim == 1:
        p_stim = p_stim[np.newaxis,:]

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


# Getting a prior over values given a orientation grid and a type
def prior_val(grid, type, interpolation_kind='linear', bins=25, slow=True):

    p_ori = prior_ori(grid)

    bin_centers, ps = ori_to_val_dist(grid, p_ori, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)

    return bin_centers, ps


# The noisy input stimulus being presented
def stimulus_ori_noise(x, kappa_s, grid):
    p_noise_ori = ss.vonmises(loc=x, kappa=kappa_s).pdf(grid)
    return p_noise_ori

# Getting the noisy input stimulus distribution in value space mapping
# Input parameters define noise in orientation space buy function gives out the noisy distribution in value space
def stimulus_val_noise(x, kappa_s, grid, type, interpolation_kind='linear', bins=25, slow=True):
    # grid = grid[np.newaxis, :]
    # grid = np.array(grid)
    if np.isscalar(x):
        x = np.array([x])
    else:
        x = np.array(x)
    x = x[:, np.newaxis]

    p_noise_ori = ss.vonmises(loc=x, kappa=kappa_s).pdf(grid)

    bin_centers, ps = ori_to_val_dist(grid, p_noise_ori, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)

    return ps


# This noise will be added to the representation space of the variable being encoded.
# This variable is value in this case
# It should not be a vonmises anymore. In fact what it should be depends on the specific type of experimental setup we
# design in my opinion. If participants are always just showed one quadrant, makes more sense that it is a truncated
# normal in my opinion. hOWEVER, IF IT IS 2 QUADRANTS, SHOULD be a recursive value that repeats back in opposite direction.
def sensory_noise(m, sd, grid, type):
    truncBoth = ss.truncnorm.pdf(grid,(0.0 - m) / sd, (1.0 -m) / sd, m, sd)
    truncUp = ss.truncnorm.pdf(grid, (-np.Inf - m) / sd, (1. - m) / sd, m, sd)
    truncLow = ss.truncnorm.pdf(grid, (0.0 - m) / sd, (np.Inf - m) / sd, m, sd)
    foldUp = ss.foldnorm.pdf(1-grid, (1-m)/sd, 0.0, sd) # 0.0 here is essentially the point of the folding
    foldLow = ss.foldnorm.pdf(grid, m/sd, 0.0, sd)
    if experimentRange == "0to45" or "45to90" or "90to135" or "135to180":
        # This one is when experiment is shown only within a 45degree angle. The distribution after truncation
        # gets redistributed elsewhere 
        return truncBoth
    if experimentRange == "0to90" or "90to180":
        if type == "linearPrior" or type == "prior" or type =="curvedPrior":
            return (1-m)*foldLow + m*truncUp # The lowed values get folded while upper ones are truncated
        if type == "inverseLinearPrior" or type == "inversePrior" or type =="inverseCurvedPrior":
            return m*foldUp + (1-m)*truncLow # The lowed values get truncated while upper ones are folded
    if experimentRange == "0to180":
        # This one is when experiment is shown only within a 45degree angle. The distribution after truncation
        # gets redistributed elsewhere 
        return m*foldUp+(1-m)*foldLow
    if experimentRange == "noBoundaryEffects":
        # We assume that they represent the information here totally and the bounds do not mean truncation but rather,
        # they jusst shift the boundary values to include 5 standard deviations from the noise, whatever the noise is
        return ss.norm.pdf(grid, m+5*sd*(0.5-m), sd)

# Takes in the orientation grid and gives out the cdf over values
def cdf_val(grid, type):
    bin_centers, ps = prior_val(grid, type)
    cdf_val = np.squeeze(integrate.cumtrapz(ps, bin_centers, initial=0.0))
    return cdf_val


# Take input orientation and gives the decoded distribution
def value_efficient_encoding(theta0, sigma_stim, sigma_rep, type, interpolation_kind='linear', bins = 25, slow=True):
    # Note that the sigma_stim is in ori space

    theta0 = np.atleast_1d(theta0)
    # Prior over values for the given type of value function mapping
    val_centers, val_prior = prior_val(stim_grid, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)

    # val0 (has same dim as theta0) x theta_gen x m_gen
    # Add stimulus noise to get distribution of values given theta0 - val0 x value_gen
    p_val_given_theta0 = stimulus_val_noise(theta0, sigma_stim, stim_grid, type)
    # Add sensory noise to see what ms for value you get given a value 0 - value_gen_rep x m_gen(rep)
    # For each point in rep space given by the cdf of the value function grid, we add sensory noise constant
    p_m_given_val0 = sensory_noise(cdf_val(theta0, type)[np.newaxis, :, np.newaxis], sigma_rep, rep_grid[np.newaxis, np.newaxis, :], type)

    # Combine sensory and stimulus noise
    p_m_given_val0 = p_m_given_val0 * p_val_given_theta0[..., np.newaxis]

    # Integrate out different generated values (due to sstim noise), so we just have ms given theta0
    p_m_given_val0 = trapezoid(p_m_given_val0, val_centers, axis=1)

    # Make a big array that for many thetas gives the probability of observing ms (value likelihood)
    p_m_given_val = stimulus_val_noise(stim_grid, sigma_stim, stim_grid, type)[..., np.newaxis] * \
        sensory_noise(cdf_val(stim_grid, type)[np.newaxis, :, np.newaxis], sigma_rep, rep_grid[np.newaxis, np.newaxis, :], type)

    # Integrate out the realized values
    p_m_given_val = trapezoid(p_m_given_val, stim_grid, axis=0)

    # Representations of values
    return p_m_given_val0, p_m_given_val

# Take input orientation and gives the decoded distribution
def value_bayesian_decoding(theta0, sigma_stim, sigma_rep, type, interpolation_kind='linear', bins=25, slow=True):

    # There is a one to one correspondence between theta0 and corresponding val0
    # val0 is implicitly presented val by presenting theta0.

    p_m_given_val0, p_m_given_val = value_efficient_encoding(theta0, sigma_stim, sigma_rep, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)
    # just changed this next line from the comebted one
    # p_val_given_m = p_m_given_val*np.array(prior_val(stim_grid, type)[1])[:, np.newaxis]
    p_val_given_m = p_m_given_val * prior_ori(stim_grid)[:, np.newaxis]

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
    val, p_value_est_given_val0 = ori_to_val_dist(stim_grid, p_value_est_given_val0, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)
    return val, p_value_est_given_val0

def risky_value_dist(theta1, sigma_stim, sigma_rep, risk_prob, type, interpolation_kind='linear', bins=25, slow=True):

    bin_centers, ps = value_bayesian_decoding(theta1, sigma_stim, sigma_rep, type, interpolation_kind=interpolation_kind, bins=bins, slow=slow)

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