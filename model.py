import numpy as np
import scipy.stats as ss
from scipy import interpolate

class EfficientCode(object):

    def __init__(self, stim_grid=None, rep_grid=None):

        if stim_grid is None:
            stim_grid = np.linspace(0, 1, 500)

        if rep_grid is None:
            rep_grid = np.linspace(0, 1, 500)

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
            stim_grid = np.linspace(0, np.pi, 500)

        self.invcdf = interpolate.interp1d(self.cdf(stim_grid), stim_grid,
                                           fill_value='extrapolate',
                                           bounds_error=False)

        self.sigma_rep = sigma_rep

        super().__init__(stim_grid=stim_grid, rep_grid=rep_grid)

    def prior(self, x):
        return (2-np.abs(np.sin(2*x)))/ ((np.pi-1)) / 2.

    def cdf(self, x):
        cdf = ((np.cos(x)**2) * np.sign(np.sin(2*x)) + 2*x) / (2*(np.pi-1.)) - (1 / (2*(np.pi-1.)))
        return np.clip(cdf, 1e-9, 1-1e-9)


    def rep_likelihood(self, m, theta_rep, sigma_rep, norm=True):
        # eq. 10 in Wei & Stocker (2015): K(m, \tilde{\theta})

        # We work in 3D: dim(batch) x dim(m) x dim(theta)
        m = np.atleast_1d(m)
        theta_rep = np.atleast_1d(theta_rep)
        
        if m.ndim == 1:
            m = m[np.newaxis, :, np.newaxis]

        if theta_rep.ndim == 1:
            theta_rep = theta_rep[np.newaxis, np.newaxis, :]

        p = vonmises180(m, sigma_rep, theta_rep)

        if norm:
            p = p / p.sum(2)[..., np.newaxis]

        return p

    def subject_estimate_theta(self, m, sigma_rep=None):
        # eq. 11 in Wei & Stocker, 2015: \tilde{\theta}_{L_2}(m)

        if sigma_rep is None:
            sigma_rep = self.sigma_rep

        p = self.rep_likelihood(m, self.rep_grid, sigma_rep)

        # Integrate over \tilde{\theta}
        theta_estimate = (np.angle((np.exp(1j*(self.stim_grid_*2))*p).sum(2)) % (2*np.pi) / 2) % np.pi
        return theta_estimate

    def model_estimate_theta(self, theta0, sigma_rep=None):
        # eq. 12 in Wei & Stocker, 2015: <\hat{\theta}_L_2>_{\theta_0}

        p = self.model_likelihood(theta0, self.rep_grid, sigma_rep)

        theta_estimate = (np.angle((np.exp(1j*(self.stim_grid_*2))*p).sum(2)) % (2*np.pi) / 2) % np.pi

        return theta_estimate

    def model_likelihood(self, theta0, theta_rep=None, sigma_rep=None):
        # eq. 13 in Wei & Stocker 2015: L_{\theta_0}(\tilde{\theta})
        if sigma_rep is None:
            sigma_rep = self.sigma_rep

        if theta_rep is None:
            theta_rep = self.rep_grid

        # dim(theta0) x dim(m) x dim(theta_rep)
        theta0 = np.atleast_1d(theta0)
        theta0_rep = self.stim2rep(theta0)

        if theta0_rep.ndim == 1:
            theta0_rep = theta0_rep[:, np.newaxis, np.newaxis]

        p_m_theta0 = self.rep_likelihood(self.rep_grid, theta0_rep, sigma_rep, norm=False)
        print(p_m_theta0.shape)
        p = p_m_theta0 * self.rep_likelihood(self.rep_grid, theta_rep, sigma_rep)

        return p.sum(1)[:, np.newaxis, :]

    def estimate_likelihood(self, theta0, sigma_rep, rep_grid=None):
        if rep_grid is None:
            rep_grid = self.rep_grid

        theta0_rep = self.stim2rep(theta0)

        m = self.rep_likelihood(rep_grid, theta0_rep, sigma_rep)

        theta_rep = self.rep_likelihood(m, theta)
        # We work in 2D: dim(m) x dim(theta)



def vonmises180(loc, sd, x):
    return ss.vonmises(loc=loc*np.pi*2., kappa=1./(sd*np.pi*2.)**2).pdf(x*np.pi*2)*np.pi*2.
