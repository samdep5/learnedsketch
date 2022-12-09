from scipy.optimize import minimize
import numpy as np
import scipy
from math import *
from scipy.stats import zipf 
#import cvxpy as cp

    #to make this work, we need to predict the CDF of the elements, to optimize over G(t) and 

k  = 3 #number of thresholds
N = 1000
epsilon = .01/N
delta = 0.05 #probability bound 
z = 1.5 #zipf parameter

print([zipf.cdf(z, i) for i in range(10)])
def getG(t): 
    print("t: ", t, "G(t): ", zipf.cdf(t, z))
    return zipf.cdf(t, z) #need to define CDF function from prediction oracle; currently a dummy function that assumes Zipf distribution with some param z 

def gettFromGt(Gt):
    pass



def objFunc(x): #change - optimize over G(t) directly, and not t (get t from the CDF of zipf at the end) or the optimizers won't work
    #x is params; first half of x is thresholds, second half is epsilons 

    thresholds = x[:k] #assume these hold G(t), not t 
    epsilons = x[k:]

    sm = 0 
    for i in range(k): 
        t = thresholds[i]
        prevt = thresholds[i-1] if i > 0 else 0
        e = epsilons[i]
        #sm += N*(getG(t) - getG(prevt)) * pow(e, min(1, 1/z))*np.log(1/delta) 
        #print("i: ", i, "e: ", e)

        #todo: fix negative error
        sm += N*(t - prevt) * pow(e, min(1, 1/z))*np.log(1/delta) 

    return sm

#bounds = scipy.optimize.Bounds([(0, 1)]*k + [(0, N)]*k) #G(t) vals bounded between 0 and 1 b/c a CDF; epsilons are bounded between 0 and N
bounds = [(0, 1)]*k + [(0, N)]*k #G(t) vals bounded between 0 and 1 b/c a CDF; epsilons are bounded between 0 and N

constr = [] # linear constraints: -inf < G(t-1) - G(t) <= 0 for all t, aka k constraints; G(0) = 0; G(1) = 1 (2 constraints); 0-inf <= N(sum(gti - gt(i-1)) - 1) - sqrt(Nepsilon) <= 0 

#ineq constraint format: fnc >= 0 
#eq constraint format: fnc = 0

def sum_ineq_constr(x): #the sum less than Nepsilon constriaint
    #x is params; first half of x is thresholds, second half is epsilons 

    thresholds = x[:k] #assume these hold G(t), not t 
    epsilons = x[k:]

    sm = 0 
    for i in range(k): 
        t = thresholds[i]
        prevt = thresholds[i-1] if i > 0 else 0
        e = epsilons[i]
        sm += ((t - prevt) ** 2)*e

    return (epsilon - sm )

constr.append({'type': 'ineq', 'fun': sum_ineq_constr})

for i in range(k):
    def ineq_constr(x): #the G(t) - G(t-1) >= 0 constraints
        #x is params; first half of x is thresholds, second half is epsilons 

        thresholds = x[:k] #assume these hold G(t), not t 

        t = thresholds[i]
        prevt = thresholds[i-1] if i > 0 else 0

        return t - prevt - .0000000001

    constr.append({'type': 'ineq', 'fun': ineq_constr})
    
constr.append({'type': 'eq', 'fun': lambda x: x[0]}) #G(0) = 0
constr.append({'type': 'eq', 'fun': lambda x: x[k-1] - 1}) #G(1) = 1



# #lower constr: 

# lowerConstr = [-np.inf]*k  #G(t-1) - G(t) >= -inf
# lowerConstr.append(0) #G(0) >= 0 
# lowerConstr.append(1) #G(1) >= 1
# lowerConstr.append(-np.inf) #N(sum(gti - gt(i-1)) - 1) >= -inf

# #upper constr:
# upperConstr = [0]*k #G(t-1) - G(t) <= 0
# upperConstr.append(0) #G(0) <= 0
# upperConstr.append(1) #G(1) <= 1
# upperConstr.append(0) #N(sum(gti - gt(i-1)) - 1) <= 0

# #middle constraints, aka array of vars 
# constr = []
# for i in range(k): 
#     constr.append(np.array([0]*i + [1, -1] + [0]*(k-i-1))) #G(t-1) - G(t) <= 0
#     #print(constr)
# constr.append(np.array([1] + [0]*(k-1))) #G(0) <= 0
# constr.append(np.array([0]*(k-1) + [1])) #G(1) <= 1

initGuess = [] #initial guess for thresholds and epsilons
for i in range(k): 
    # assume equal thresholds 
    # t = i*(N/k)
    # initGuess.append(getG(t)) #G(t)
    initGuess.append((i/k)) #t

initGuess += [epsilon]*k #assume equal epsilons, equal to overall epsilon 
print("initGuess: ", initGuess)

#res = minimize(objFunc, initGuess, method='SLSQP', bounds=bounds, constraints=constr, options={'disp': True})
res = minimize(objFunc, initGuess, method='SLSQP', bounds=bounds, constraints=constr, options={'disp': True})
print(res)
