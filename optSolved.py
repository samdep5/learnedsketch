from scipy.optimize import minimize
import numpy as np
import scipy
from math import *
#import cvxpy as cp

    #to make this work, we need to predict the CDF of the elements, to optimize over G(t) and 

k  = 4 #number of thresholds
N = 1000
epsilon = .01/N
delta = 0.05 #probability bound 
z = 2 #zipf parameter

#print("ZipfCDF: ", [scipy.stats.zipf.cdf(z, i) for i in range(10)])
def getG(t): 
    print("t: ", t, "G(t): ", scipy.stats.zipf.cdf(t, z))
    return scipy.stats.zipf.cdf(t, z) #need to define CDF function from prediction oracle; currently a dummy function that assumes Zipf distribution with some param z 

def gettFromGt(Gt):
    return scipy.stats.zipf.ppf(Gt, z)

k += 1 #start and end at 0 and 1, respectively

def objFunc(Gtvals): #change - optimize over G(t) directly, and not t (get t from the CDF of zipf at the end) or the optimizers won't work
    #x is params; first half of x is thresholds, second half is epsilons 


    sm = 0 
    for i in range(k): 
        Gt = Gtvals[i]
        prevGt = Gtvals[i-1] if i > 0 else 0
        G = Gt - prevGt
        if G < 0:
            print("G < 0: ", G, Gt, prevGt, i)
            #TODO include? G=0
        GArr = np.array([ Gtvals[i] - Gtvals[i-1] if i > 0 else 0 for i in range(k) ])
        #sm += N*(getG(t) - getG(prevt)) * pow(e, min(1, 1/z))*np.log(1/delta) 
        #print("i: ", i, "e: ", e)

        #todo: fix negative error
        m = min(1, 1/z)
        powArr = np.power(GArr, 1/(m-1))
        for i in range(len(powArr)):
            if powArr[i] == np.inf:
                powArr[i] = 0
        e = epsilon*G / np.sum([powArr])
        if e < 0: 
            print(e, powArr, G)
            #TODO: fix negative error
            e = abs(e)
        sm += N*(Gt - prevGt) * pow(e, m)*np.log(1/delta) 

    return sm

bounds = [(0, 2)]*k #G(t) vals bounded between 0 and 1 b/c G a CDF; epsilons are bounded between 0 and N

constr = [] # linear constraints: -inf < G(t-1) - G(t) <= 0 for all t, aka k constraints; G(0) = 0; G(1) = 1 (2 constraints); 0-inf <= N(sum(gti - gt(i-1)) - 1) - sqrt(Nepsilon) <= 0 

#ineq constraint format: fnc >= 0 
#eq constraint format: fnc = 0

def sum_ineq_constr(Gtvals): #the sum less than Nepsilon constriaint
    #x is params; first half of x is thresholds, second half is epsilons 

    sm = 0 
    for i in range(k): 
        Gt = Gtvals[i]
        prevGt = Gtvals[i-1] if i > 0 else 0
        GArr = np.array([ Gtvals[i] - Gtvals[i-1] if i > 0 else 0 for i in range(k) ])

        m = min(1, 1/z)
        powArr = np.power(GArr, 1/(m-1))
        for i in range(len(powArr)):
            if powArr[i] == np.inf:
                powArr[i] = 0
        #print("powArr ", GArr, powArr)
        e = epsilon*(Gt - prevGt) / np.sum(powArr)
        sm += ((Gt - prevGt) ** 2)*e

    return (epsilon - sm )

constr.append({'type': 'ineq', 'fun': sum_ineq_constr})

#the G(t) - G(t-1) >= 0 constraints

for i in range(k): 
    def gtconstr(GVals): 
        Gt = GVals[i]
        prevGt = GVals[i-1] if i > 0 else 0
        # if Gt - prevGt < 0: 
        #     print(i, "Gt", Gt, "prevGt", prevGt, "Gt - prevGt", Gt - prevGt)
        # print("Gt", Gt, "prevGt", prevGt, "Gt - prevGt", Gt - prevGt)

        return Gt - prevGt -(1/k)
    
    constr.append({'type': 'ineq', 'fun': gtconstr})

# for i in range(k):
#     def ineq_constr(x): #the G(t) - G(t-1) >= 0 constraints
#         #x is params; first half of x is thresholds, second half is epsilons 

#         thresholds = x #assume these hold G(t), not t 

#         t = thresholds[i]
#         prevt = thresholds[i-1] if i > 0 else 0

#         return t - prevt

#     constr.append({'type': 'ineq', 'fun': ineq_constr})

constr.append({'type': 'eq', 'fun': lambda x: x[0]}) #G(0) = 0
constr.append({'type': 'eq', 'fun': lambda x: x[k-1] - 1}) #G(1) = 1

print("Constraints: ", constr, len(constr))

initGuess = [] #initial guess for thresholds and epsilons
for i in range(k): 
    # assume equal thresholds 
    # t = i*(N/k)
    # initGuess.append(getG(t)) #G(t)
    initGuess.append((i/k)) #t
    
print("initGuess: ", initGuess)

#res = minimize(objFunc, initGuess, method='SLSQP', bounds=bounds, constraints=constr, options={'disp': True})
res = minimize(objFunc, initGuess, bounds=bounds, method='SLSQP', constraints=constr, options={'disp': True})
print(res)
print(res.x)

for gt in res.x: 
    print(gettFromGt(gt))