import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid

class perception(object):


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

    def cdf(self, x):
        x_ = x % np.pi
        return integrate.cumtrapz(self.prior(x_), self.stim_grid, initial=0.0)

    def invcdf(self, x):
        return np.gradient(self.cdf(self.stim_grid))
    
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

