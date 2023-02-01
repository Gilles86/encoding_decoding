import numpy as np
import math

x = np.linspace(0, 180, 91) * np.pi / 90
m = np.linspace(0, 180, 91) * np.pi / 90
x = x[:90]
m = m[:90]
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

def bias(kappa, kappa_s):
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

    posterior = np.zeros(len(x))
    bayes_mean = np.zeros(len(m))

    for j in range(len(m)):
        posterior = con[:, j] * prior

        bayes_mean[j] = np.arctan2(np.sum(np.sin(x) * posterior), np.sum(np.cos(x) * posterior))
        bayes_mean[j] = bayes_mean[j] - np.floor(bayes_mean[j] / (2 * np.pi)) * 2 * np.pi

    bias_mean = np.zeros(len(m))

    tem = np.zeros(len(x))
    for i in range(len(x)):
        tem = con[i, :] / np.sum(con[i, :])
        weight = tem * prior
        bias_mean[i] = np.arctan2(np.sum(np.sin(bayes_mean) * weight), np.sum(np.cos(bayes_mean) * weight))
        bias_mean[i] = bias_mean[i] - np.floor(bias_mean[i] / (2 * np.pi)) * 2 * np.pi
        bias_mean[i] = bias_mean[i] - x[i]

    return bias_mean



# import numpy as np
# import math
# import scipy.stats as ss
# from scipy import interpolate
# from scipy import integrate
# from scipy.integrate import simpson, trapezoid, cumulative_trapezoid
#
# x = np.linspace(0, np.pi * 2., 90, False)
# m = np.linspace(0, np.pi * 2., 90, False)
# prior = np.zeros(len(x))
# accu = np.zeros(len(x))
# base = 2
#
#
# def den(x):
#
#     de = (base - np.abs(np.sin(x)))
#     de = de / (2 * np.pi * base)
#
#     return de
#
#
# for i in range(len(x)):
#     prior[i] = den(x[i])
#
# prior = (1/0.001)*prior/sum(prior)
#
# def accum(x):
#
#     if x < np.pi:
#         acc = (-1 + np.cos(x) + base * x) / (2 * np.pi * base - 4)
#     if x >= np.pi:
#         acc = 1 / 2 + (base * (x - np.pi) - 1 + np.cos(x - np.pi)) / (2 * np.pi * base - 4)
#
#     acc = acc * 2. * np.pi
#     return acc
#
#
# # def quan(x, accu):
# #     aaa = np.where(accu > x)
# #     qua = min(aaa)
# #     return qua
#
#
# for i in range(len(x)):
#     accu[i] = accum(x[i])
#
# m = accu
#
#
# def bias(kappa, kappa_s):
#     con = np.zeros((len(x), len(m)))
#     bayes_mean = np.zeros((len(m)))
#     temp = np.zeros(len(x))
#     von = np.zeros(len(x))
#
#     for s in range(len(x)):
#         for j in range(len(x)):
#             for t in range(len(x)):
#                 von[t] = np.exp(kappa * (np.cos(m[j] - accu[t]) - 1))
#                 temp[t] = np.exp(kappa_s * (np.cos(x[s] - x[t]) - 1))
#             con[s, j] = sum(temp * von)
#     con = con / sum(con)
#
#     for j in range(len(m)):
#         posterior = con[:, j] * prior
#         bayes_mean[j] = math.atan2(sum(np.sin(x) * posterior), sum(np.cos(x) * posterior))
#         bayes_mean[j] = bayes_mean[j] - math.floor(bayes_mean[j] / (2. * np.pi)) * 2. * np.pi
#
#     bias_mean = np.zeros(len(m))
#
#     for s in range(len(x)):
#         tem = con[s, :] / sum(con[s, :])
#         weight = tem * prior
#         bias_mean[s] = math.atan2(sum(np.sin(bayes_mean) * weight), sum(np.cos(bayes_mean) * weight))
#         bias_mean[s] = bias_mean[s] - math.floor(bias_mean[s] / (2. * np.pi)) * 2. * np.pi
#         bias_mean[s] = bias_mean[s] - x[s]
#
#     return bias_mean
