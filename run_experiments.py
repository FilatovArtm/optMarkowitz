from oracles import Oracle
from optimization import *
import pandas as pd
import numpy as np
import cvxpy
from datetime import datetime
from xml.etree import ElementTree as ET
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from collections import defaultdict


data = pd.read_csv('stocks_data.csv', delimiter=',')
data = data[1:-1]
data = data.loc[:, data.columns != 'Unnamed: 0']
data = data.fillna(0)

names = list(map(lambda x: x.split('.')[0], list(data.columns)))
n_stocks = len(names)

incomes = np.array(data.values, dtype=np.float32)
mean_income = np.mean(incomes, axis=0)
covariance_income = np.cov(incomes.T)

solutions = []

for i in np.linspace(0, 0.03, 10):
    oracle = Oracle(Sigma=covariance_income, income=mean_income, lambda_1=i,rho=5, mode='efficient')
    solutions.append(AcceleratedProximalGD(oracle, x_0=np.random.uniform(size=len(mean_income)))[0])

sef = np.array(solutions)

for i in range(100):
    plt.plot(np.linspace(0, 0.03, 10), sef[:, i])
plt.ylabel('Share')
plt.xlabel('$\lambda_1$')
plt.title('')
plt.savefig('exp_1.png')
plt.clf()

plt.plot(np.linspace(0, 0.03, 10), np.sum(sef > 1e-6, axis=1))
plt.ylabel('Number of non-zero stocks')
plt.xlabel('$\lambda_1$')
plt.savefig('exp_2.png')
plt.clf()

np.random.seed(4)

results_our = []
for i in range(5):
    ns = np.arange(10, 1000, 80)
    result_our = []
    for n in ns:
        diagonal = np.random.uniform(0, 10, n)
        b = np.random.normal(0,1, n)
        oracle = Oracle(Sigma=np.diag(diagonal), income=b, lambda_1=1, rho=5, mode='efficient')
        start_time = datetime.now()
        [x_star, msg, history] = AcceleratedProximalGD(oracle, np.zeros(n), trace=False, max_iter=100)
        result_our.append((datetime.now() - start_time).total_seconds())

    results_our.append(result_our)

np.random.seed(4)

results_cvx = []
for i in range(5):
    ns = np.arange(10, 1000, 80)
    result_cvx = []
    for n in ns:
        diagonal = np.random.uniform(0, 10, n)
        b = np.random.normal(0,1, n)

        x = cvxpy.Variable(n)
        obj = cvxpy.Minimize(cvxpy.quad_form(P=np.diag(diagonal), x=x)- 5 * x.T @ b + cvxpy.norm1(x))
        cons = [cvxpy.sum_entries(x) == 1]

        start_time = datetime.now()
        cvxpy.Problem(obj, cons).solve(solver='GUROBI')
        result_cvx.append((datetime.now() - start_time).total_seconds())

    results_cvx.append(result_cvx)

for result in results_our:
    plt.plot(ns, result, '-.', alpha=0.1, c='r')
plt.plot(ns, np.mean(results_our, axis=0), c='r', label='ADMM')

for result in results_cvx:
    plt.plot(ns, result, '-.', alpha=0.1, c='b')

plt.plot(ns, np.mean(results_cvx, axis=0), c='b', label='GUROBI')
plt.xlabel('Dimension')
plt.ylabel('Seconds')
plt.legend()
plt.savefig('exp_3.png')
plt.clf()


# experiment_2

data = pd.read_csv('stocks_data.csv', delimiter=',')
data = data[1:-1]
data = data.loc[:, data.columns != 'Unnamed: 0']

data = data.fillna(0)

r = np.mean(data, axis=0).values

n = r.shape[0]
Sigma = np.cov(data.values.astype(float).T) + np.diag(np.ones(n) * 1e-5)
lambd = 0.01

np.random.seed(1)
x0 = np.random.uniform(0, 1, size=n)

orac = Oracle(Sigma, r, lambd, 1, mode='cvxpy')
fista_cvx = FISTA(orac, x0/ np.sum(x0), max_iter=500, trace=True)[2]
acg_cvx = AcceleratedProximalGD(orac, x0/ np.sum(x0), max_iter=500, trace=True)[2]


orac = Oracle(Sigma, r, lambd, 1, mode='efficient')
fista_nocvx = FISTA(orac, x0/ np.sum(x0), max_iter=500, trace=True)[2]
acg_nocvx = AcceleratedProximalGD(orac, x0/ np.sum(x0), max_iter=500, trace=True)[2]

plt.figure(figsize=(12,10))
plt.plot(fista_cvx['time'][:20], fista_cvx['func'][:20], label = 'FISTA (cvx proxy)')
plt.plot(fista_nocvx['time'][:20], fista_nocvx['func'][:20], label = 'FISTA')
plt.plot(acg_nocvx['time'][:20], acg_nocvx['func'][:20], label = 'APCG')
plt.xlabel('Time')
plt.ylabel('Log-scale objective')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('exp_4.png')
plt.clf()

admm = ADMM(orac, x0/np.sum(x0), 0.3, max_iter=5000, trace=True)[2]
plt.figure(figsize=(12,10))
plt.plot(fista_nocvx['time'][:20], fista_nocvx['func'][:20], label = 'FISTA')
plt.plot(acg_nocvx['time'][:20], acg_nocvx['func'][:20], label = 'APCG')
plt.plot(admm['time'][:20], admm['func'][:20], label = 'ADMM')
plt.xlabel('Time')
plt.ylabel('Log-scale objective')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('exp_5.png')
plt.clf()
