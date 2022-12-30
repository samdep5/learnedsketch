import numpy as np 
import scipy
from collections import Counter
from scipy.optimize import minimize_scalar


def gaussianData(N, mean=0, sigma=1): 
    return np.random.normal(mean, sigma, N)

def uniformData(N, a=0, b=10): 
    return np.random.uniform(a, b, N)

def zipfData(N, a=1.74): 
    return np.random.zipf(a, N)

def paretoData(N, a=1.1):
    return np.random.pareto(a, N)

def dataElementFrequencies(data): #'ideal oracle'
    data = np.array(data)
    cts = Counter(data)
    # ctArr = [cts[item] for item in data]
    return cts

zipD = zipfData(N=100)
def neg_zipf_likelihood(s, dataset=zipD):
    freqSet = dataElementFrequencies(dataset)
    n = sum(freqSet)
    # for each word count, find the probability that a random word has such word count
    probas = dataset ** (-s) / np.sum(np.arange(1, n+1) **(-s))
    log_likelihood = sum(np.log(probas) * dataset)
    return -log_likelihood

data = np.array()
v = zipfData(100)
c = dataElementFrequencies(v)
countMin = []


- dual number consistency
- handle multiple functions
- documentation 
- more consistent tests,


# print(dat)
# print(dataElementFrequencies(zipD))
# s_best = minimize_scalar(neg_zipf_likelihood, [0.1, 3.0] )
# print(s_best.x, 1/s_best.x)

