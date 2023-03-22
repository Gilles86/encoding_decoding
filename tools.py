import numpy as np
import scipy.stats as ss
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, trapezoid, cumulative_trapezoid, fixed_quad
import scipy.stats as ss
from scipy.optimize import minimize

import riskModelValuation as model

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
    y = p_chose_risky.ravel()
    x = risky_payoffs.ravel()

    def get_probit(x, intercept, slope):
        return ss.norm(0.0, 1.0).cdf(intercept + slope*x)

    def cost(xs, ps, intercept, slope):
        return np.sum((get_probit(xs, intercept, slope) - ps)**2)

    def cost_(pars, *args):
        intercept, slope = pars
        return cost(x, y, intercept, slope)

    result = minimize(cost_, (-safe_payoff/risk_prob, 1.0), method='L-BFGS-B')

    intercept_est, slope_est = result.x

    indifference_point = -intercept_est/slope_est

    rnp = safe_payoff / indifference_point
    return rnp

def inverse_monotonic(y_0, type):
    x = model.stim_grid
    y = model.value_function_ori(model.stim_grid, type = type)
    indices = np.argsort(y)
    y_sorted = y[indices]
    
    def find_closest_index(y_0):
        i = np.searchsorted(y_sorted, y_0, side='left')
        if i == 0:
            return 0
        elif i == len(y_sorted):
            return len(y_sorted) - 1
        else:
            if y_0 - y_sorted[i-1] < y_sorted[i] - y_0:
                return i - 1
            else:
                return i

    i = find_closest_index(y_0)
    x_inverse = x[indices[i]]*90./np.pi
    return x_inverse