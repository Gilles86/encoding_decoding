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
        self.stim_grid_ = self.invcdf(stim_grid)

    def prior(self, x):
        ...

    # def cdf(self, x):
    #     ...

    def invcdf(self, x):
        ...

    def stim2rep(self, x_stim, p_stim):
        x_rep = self.cdf(x_stim)
        p_rep = p_stim / self.prior(x_stim)

        return x_rep, p_rep

    def rep2stim(self, x_rep, p_rep):
        x_stim = self.invcdf(x_rep)
        p_stim = p_rep * self.prior(x_rep)

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

    def __init__(self, stim_grid=None, rep_grid=None):

        if stim_grid is None:
            stim_grid = np.linspace(0, np.pi, 500)

        self.invcdf = interpolate.interp1d(self.cdf(stim_grid), stim_grid,
                                           fill_value='extrapolate',
                                           bounds_error=False)

        super().__init__(stim_grid=stim_grid, rep_grid=rep_grid)

    def prior(self, x):
        return (2-np.abs(np.sin(2*x)))/ ((np.pi-1)) / 2.

    def cdf(self, x):
        cdf = ((np.cos(x)**2) * np.sign(np.sin(2*x)) + 2*x) / (2*(np.pi-1.)) - (1 / (2*(np.pi-1.)))
        return np.clip(cdf, 1e-9, 1-1e-9)


    def rep_likelihood(self, m, theta_rep, sigma_rep):
        # We work in 2D: dim(m) x dim(theta)
        m = np.atleast_1d(m)[:, np.newaxis]
        theta_rep = np.atleast_1d(theta_rep)[np.newaxis, :]

        p = vonmises180(m, sigma_rep, theta_rep)
        p = p / p.sum(1)[:, np.newaxis]

        return p

    def estimate_theta_rep(self, m, sigma_rep):
        p = self.rep_likelihood(m, self.rep_grid, sigma_rep)
        print(p.sum(1))
        theta_estimate = (np.angle((np.exp(1j*(self.rep_grid*2*np.pi))*p).sum(1)) % (2*np.pi) / (2*np.pi))
        return theta_estimate

    def estimate_likelihood(self, theta0, sigma_rep, rep_grid=None):
        if rep_grid is None:
            rep_grid = self.rep_grid

        theta0_rep = self.stim2rep(theta0)

        m = self.rep_likelihood(rep_grid, theta0_rep, sigma_rep)

        theta_rep = self.rep_likelihood(m, thetra)
        # We work in 2D: dim(m) x dim(theta)


    def stim2rep(self, x_stim, p_stim=None):
        if p_stim is None:
            return self.cdf(x_stim)

    def rep2stim(self, x_rep, p_rep=None):
        if p_rep is None:
            return self.invcdf(x_rep)

def vonmises180(loc, sd, x):
    return ss.vonmises(loc=loc*np.pi*2., kappa=1./(sd*np.pi*2.)**2).pdf(x*np.pi*2)*np.pi*2.
