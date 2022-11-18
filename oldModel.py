import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy.integrate import simpson, trapezoid

class EfficientCode(object):

    def __init__(self, stim_grid=None, rep_grid=None):

        if stim_grid is None:
            stim_grid = np.linspace(0, 2*np.pi, 1000, endpoint=False)

        if rep_grid is None:
            rep_grid = np.linspace(0, 1, 1000, endpoint=False)

        self.stim_grid = stim_grid
        self.rep_grid = rep_grid

        self.rep_grid_ = self.cdf(stim_grid)
        self.stim_grid_ = self.invcdf(rep_grid)

    def prior(self, x):
        ...

    # def cdf(self, x):
    #     ...

    def invcdf(self, x):
        ...

    def stim2rep(self, x_stim, p_stim=None):

        x_rep =  self.cdf(x_stim)

        if p_stim is None:
            return x_rep

        p_rep = p_stim / self.prior(x_stim)

        return x_rep, p_rep

    def rep2stim(self, x_rep, p_rep=None):

        x_stim = self.invcdf(x_rep)

        if p_rep is None:
            return x_stim

        p_stim = p_rep * self.prior(x_stim)

        return x_stim, p_stim

    def subject_likelihood(self, m, theta_rep):

        # The output is dim(m) x dim(theta)
        m = np.array(m)
        if m.ndim != 2:
            m = np.atleast_2d(m).T

        # The output is dim(m) x dim(theta)
        theta_rep = np.array(theta_rep)
        if theta_rep.ndim != 2:
            theta_rep = np.atleast_2d(theta_rep).T


class OrientationWei(EfficientCode):

    def __init__(self, stim_grid=None, rep_grid=None, sigma_rep=.1):

        if stim_grid is None:
            stim_grid = np.linspace(0, 2*np.pi, 1000, endpoint=False)

        self.invcdf = interpolate.interp1d(self.cdf(stim_grid), stim_grid,
                                           fill_value='extrapolate',
                                           bounds_error=False)

        self.sigma_rep = sigma_rep

        super().__init__(stim_grid=stim_grid, rep_grid=rep_grid)

    def prior(self, x):
        return (2-np.abs(np.sin(2*x)))/ (np.pi-1)/4.0

    def cdf(self, x):
        x_ = x % np.pi
        cdf = np.clip(
            (((np.cos(x_) ** 2) * np.sign(np.sin(2 * x_)) + 2 * x_) / (2 * (np.pi - 1.)) - (1 / (2 * (np.pi - 1.)))), 0,
            1.) * .5
        cdf += (x // np.pi) * .5
        return cdf


    def rep_likelihood(self, m, theta_rep, sigma_rep, norm=False):
        # eq. 10 in Wei & Stocker (2015): K(m, \tilde{\theta})

        # We work in 3D: dim(batch) x dim(m) x dim(theta)

        ## This is in sensory space (symmetric)
        m = np.atleast_1d(m)
        theta_rep = np.atleast_1d(theta_rep)
        
        if m.ndim == 1:
            m = m[np.newaxis, :, np.newaxis]

        if theta_rep.ndim == 1:
            theta_rep = theta_rep[np.newaxis, np.newaxis, :]

        p = sensory_noise_dist(theta_rep, sigma_rep, m)

        return p

    def subject_estimate_theta(self, m, sigma_rep=None):
        # eq. 11 in Wei & Stocker, 2015: \tilde{\theta}_{L_2}(m)

        # This is the posterior estimate of a subject who has a specefic representation (m), always the same for the same m. But omne theta0 never leads to the same m

        if sigma_rep is None:
            sigma_rep = self.sigma_rep

        p = self.rep_likelihood(m, self.rep_grid, sigma_rep)

        Ftheta = np.exp(1j*(self.stim_grid_[np.newaxis, np.newaxis, :]))
        integral = trapezoid(Ftheta*p, self.rep_grid, axis=2)
        return np.angle(integral) % (2*np.pi)

    def model_likelihood(self, theta0, sigma_rep=None):
        # eq. 13 in Wei & Stocker 2015: L_{\theta_0}(\tilde{\theta})
        if sigma_rep is None:
            sigma_rep = self.sigma_rep

        # dim(theta0) x dim(m) x dim(theta_rep)
        theta0 = np.atleast_1d(theta0)
        theta0_rep = self.stim2rep(theta0)

        if theta0_rep.ndim == 1:
            theta0_rep = theta0_rep[:, np.newaxis, np.newaxis]

        p_m_theta0 = sensory_noise_dist(theta0_rep, sigma_rep, self.rep_grid[np.newaxis, :, np.newaxis])

        # p_m_theta0 = self.rep_likelihood(theta0_rep, self.rep_grid[np.newaxis, :, np.newaxis], sigma_rep, norm=False)
        ll = self.rep_likelihood(self.rep_grid[np.newaxis, :, np.newaxis], self.rep_grid[np.newaxis, np.newaxis, :], sigma_rep)
        p = p_m_theta0 * ll

        p = trapezoid(p, self.rep_grid, axis=1)[:, np.newaxis, :]

        return p

    def model_estimate_theta(self, theta0, sigma_rep=None):
        # eq. 12 in Wei & Stocker, 2015: <\hat{\theta}_L_2>_{\theta_0}

        ## This is the average posterior estimate given a theta0. Also fixed

        p = self.model_likelihood(theta0, sigma_rep)

        Ftheta = np.exp(1j*(self.stim_grid_[np.newaxis, np.newaxis, :]))
        integral = trapezoid(Ftheta*p, self.rep_grid, axis=2)
        return np.angle(integral) % (2*np.pi)

    def theta_hat_dist(self, theta0, sigma_rep=None):

        if sigma_rep is None:
            sigma_rep = self.sigma_rep

        theta0_ = self.stim2rep(theta0)
        theta0_ = np.atleast_1d(theta0_)
        
        # Calculate estimated thetas for the equally-spaced sensory grid rep_grid (n_theta0xn_grid)
        theta_est_ = self.subject_estimate_theta(self.rep_grid, sigma_rep=sigma_rep).squeeze()

        # We make a lookup table that maps from estimated thetas to corresponding ms
        # f^{-1}(\hat{\theta}) -> m
        theta_est_inv = interpolate.interp1d(theta_est_, self.rep_grid, fill_value='extrapolate', bounds_error=False)

        theta_est_dx_ = np.gradient(theta_est_, self.rep_grid)
        theta_est_dx = interpolate.interp1d(self.rep_grid, theta_est_dx_, fill_value='extrapolate', bounds_error=False)

        # Now, for the equally-spaced orientations in model.stim_grid, we calculate
        # the density of p(\hat{theta} | \theta_0)
        # \frac{p(f^{-1}(\hat{\theta})|\theta_0)}{f'(f^{-1}(\hat{\theta}))}$
        
        # This grid of ms corresponds to possible theta-hats in the grid model.stim_grid
        m_grid = theta_est_inv(self.stim_grid)

        # theta0xstim_grid
        pm = sensory_noise_dist(theta0_[:, np.newaxis], sigma_rep, m_grid[np.newaxis, :])

        # theta0xstim_grid
        ptheta = pm/theta_est_dx(theta_est_inv(self.stim_grid))[np.newaxis, :]

        return ptheta

def sensory_noise_dist(loc, sd, x):
    return ss.vonmises(loc=loc*np.pi*2., kappa=1./(sd*np.pi*2.)**2).pdf(x*np.pi*2)*np.pi*2.

def sensory_noise_dist_sample(loc, sd, n=100):
    return ss.vonmises(loc=loc*np.pi*2., kappa=1./(sd*np.pi*2.)**2).rvs(n) % (2*np.pi) / 2. / np.pi