import numpy as np
import math

x = np.linspace(0, 180, 91) * np.pi / 90
x = x[:90]
prior = np.ones(len(x))
accu = np.zeros(len(x))
base = 2

def accum(x, base):
    if x < np.pi:
        acc = (-1 + np.cos(x) + base * x) / (2 * np.pi * base - 4)
    if x >= np.pi:
        acc = 1/2 + (base * (x - np.pi) - 1 + np.cos(x - np.pi)) / (2 * np.pi * base - 4)
    return acc * 2 * np.pi

def den(x, base):
    de = base - np.abs(np.sin(x))
    return de / (2 * np.pi * base)

for i in range(len(prior)):
    prior[i] = den(x[i], base)

prior = (1 / 0.001) * prior / np.sum(prior)

for i in range(len(accu)):
    accu[i] = accum(x[i], base)


m = accu


def p_m_given_theta(kappa, kappa_s):
    temp = np.zeros(len(x))
    von = np.zeros(len(x))
    con = np.zeros((len(x), len(m)))

    for i in range(len(x)):
        for j in range(len(m)):
            for l in range(len(x)):
                von[l] = np.exp(kappa * (np.cos(m[j] - accu[l]) - 1))
                temp[l] = np.exp(kappa_s * (np.cos(x[i] - x[l]) - 1))

            con[i, j] = np.sum(temp * von)

    con = con / np.sum(con)

    return con

def mean_theta_given_m(kappa, kappa_s):
    con = p_m_given_theta(kappa, kappa_s)
    bayes_mean = np.zeros(len(m))

    for j in range(len(m)):
        posterior = con[:, j] * prior

        bayes_mean[j] = np.arctan2(np.sum(np.sin(x) * posterior), np.sum(np.cos(x) * posterior))
        bayes_mean[j] = bayes_mean[j] - np.floor(bayes_mean[j] / (2 * np.pi)) * 2 * np.pi

    return bayes_mean

def bias(kappa, kappa_s):
    con = p_m_given_theta(kappa, kappa_s)

    bayes_mean = mean_theta_given_m(kappa, kappa_s)

    bias_mean = np.zeros(len(x))

    for i in range(len(x)):
        tem = con[i, :] / np.sum(con[i, :])
        weight = tem * prior
        bias_mean[i] = np.arctan2(np.sum(np.sin(bayes_mean) * weight), np.sum(np.cos(bayes_mean) * weight))
        bias_mean[i] = bias_mean[i] - np.floor(bias_mean[i] / (2 * np.pi)) * 2 * np.pi
        bias_mean[i] = bias_mean[i] - x[i]

    return bias_mean
