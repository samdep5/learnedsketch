from scipy.optimize import minimize
import numpy as np
import scipy
import string
from math import *
#import cvxpy as cp

    #to make this work, we need to predict the CDF of the elements, to optimize over G(t) and 

def findBuckets(k=4, N=1000, epsilon=.01/1000, delta=0.05, z=1.51, verbose=False):
    """returns (res, arr) where res is the full optimization result from scipy (including end objective function value) and arr contains the 
    G(t) values defining the borders of each bucket, in order of increasing t, excluding 0 and 1"""
    #k  = 3 #number of thresholds
    #N = 1000
    #epsilon = .01/N
    #delta = 0.05 #probability bound 
    #z = 5 #zipf parameter

    #epsilon * N = .01
    epsilon = .01/N

    m = -1*min(1, 1/z)
    #c = pow(np.log(1/delta)*m, (m-2)/(m-1)) #constant from the paper
    

#print("ZipfCDF: ", [scipy.stats.zipf.cdf(z, i) for i in range(10)])
    def getG(t): 
        print("t: ", t, "G(t): ", scipy.stats.zipf.cdf(t, z))
        return scipy.stats.zipf.cdf(t, z) #need to define CDF function from prediction oracle; currently a dummy function that assumes Zipf distribution with some param z 

    def gettFromGt(Gt):
        return scipy.stats.zipf.ppf(Gt, z)

    k += 1 #start and end at 0 and 1, respectively #DONE fix duplication error

    def objFunc(x): #change - optimize over G(t) directly, and not t (get t from the CDF of zipf at the end) or the optimizers won't work
        #x is params; first half of x is thresholds, second half is epsilons 

        Gtvals = x[:k]
        epsilons = x[k:]
        sm = 0 
        GArr = np.array([ Gtvals[i] - Gtvals[i-1] if i > 0 else 0 for i in range(k) ])
        print(Gtvals, GArr, epsilons)

        for i in range(k): 
            Gt = Gtvals[i]
            prevGt = Gtvals[i-1] if i > 0 else 0
            G = Gt - prevGt
            if G <= 0:#and verbose:
                # if verbose: 
                #     print("G < 0: ", G, i)#, G, Gt, prevGt, i)
                G = 0
            
            #sm += N*(getG(t) - getG(prevt)) * pow(e, min(1, 1/z))*np.log(1/delta) 
            #print("i: ", i, "e: ", e)

            #todo: fix negative error
            
            powArr = np.power(GArr, 1/(m-1))
            for i in range(len(powArr)):
                if powArr[i] == np.inf:
                    powArr[i] = 0
            #e = (k-1)*epsilon*pow(G, 1/(m-1)) / np.sum([powArr])
            e = epsilons[i]
            if e < 0: 
                if verbose:
                    print("NEG e", e, powArr, G)
                #TODO: fix negative error
                #e = 0
            sm += N*G * pow(e, m)*np.log(1/delta) 

        return sm

    bounds = [(0, 2)]*k #G(t) vals bounded between 0 and 1 b/c G a CDF; epsilons are bounded between 0 and N
    bounds += [(0, N)]*k #epsilons 

    constr = [] # linear constraints: -inf < G(t-1) - G(t) <= 0 for all t, aka k constraints; G(0) = 0; G(1) = 1 (2 constraints); 0-inf <= N(sum(gti - gt(i-1)) - 1) - sqrt(Nepsilon) <= 0 

    #ineq constraint format: fnc >= 0 
    #eq constraint format: fnc = 0

    def sum_ineq_constr(x): #the sum less than Nepsilon constriaint
        #x is params; first half of x is thresholds, second half is epsilons 
        Gtvals = x[:k]
        epsilons = x[k:]
        GArr = np.array([ Gtvals[i] - Gtvals[i-1] if i > 0 else 0 for i in range(k) ])
        sm = 0 
        for i in range(k): 
            Gt = Gtvals[i]
            prevGt = Gtvals[i-1] if i > 0 else 0
            
            G = Gt - prevGt
            # if G <= 0: 
            #     # if verbose: 
            #     #     print("G <= 0, ", G, i)
            #     G = 0
            # print("G, 1/(m-1", G, 1/(m-1))
            powArr = np.power(GArr, 1/(m-1))
            for i in range(len(powArr)):
                if powArr[i] == np.inf:
                    powArr[i] = 0
            #print("powArr ", GArr, powArr)
            e = epsilons[i]#(k-1)*epsilon*pow(G, 1/(m-1)) / np.sum(powArr)
            sm += (G ** 2)*e
        if True or (epsilon - sm + .0001) <0: 
            print("FALSE HERE")
            print(epsilon - sm - .00001)
            print((G ** 2))

        return (epsilon - sm - .00001)

    constr.append({'type': 'ineq', 'fun': sum_ineq_constr})

    #the G(t) - G(t-1) >= 0 constraints

    for i in range(k): 
        def gtconstr(x): 
            GVals = x[:k]
            Gt = GVals[i]
            prevGt = GVals[i-1] if i > 0 else 0
            # if Gt - prevGt < 0: 
            #     print(i, "Gt", Gt, "prevGt", prevGt, "Gt - prevGt", Gt - prevGt)
            # print("Gt", Gt, "prevGt", prevGt, "Gt - prevGt", Gt - prevGt)
            if Gt - prevGt <0: 
                print("FALSE HERE")
            return Gt - prevGt #-(1/k)
        
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

    # print("Constraints: ", constr, len(constr))

    initGuess = [] #initial guess for thresholds and epsilons
    for i in range(k): 
        # assume equal thresholds 
        # t = i*(N/k)
        # initGuess.append(getG(t)) #G(t)
        initGuess.append((i/k)) #t 
    initGuess += [epsilon/k]*k #epsilons
        
    print("initGuess: ", initGuess)

    #res = minimize(objFunc, initGuess, method='SLSQP', bounds=bounds, constraints=constr, options={'disp': True})
    res = minimize(objFunc, initGuess, bounds=bounds, method='SLSQP', constraints=constr, options={'disp': verbose})
    arr = []
    for idx in range(len(res.x)):
        gt = res.x[idx] 
        if gt != 0 and gt != 1 and np.round(gt, 2) - np.round(res.x[idx - 1], 2) > 0.01:
            arr.append(gt)
    return (res, arr)

res, Gts = findBuckets(k=3, verbose=True)
print(res)
print(Gts)

def getThresholdsFromGts(Gts, file=None):
    if file != None:
        data = []
        with open(file, 'r') as f:
            fst= True
            ct = 0 
            for line in f:
                if fst:
                    fst = False
                    continue
                line = line.replace(',', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = line.replace(' ', '')
                line = line.replace('\n', '')
                line = line.upper()
                # if ct < 10 and line: 
                #     print(line, float(line))
                #     ct += 1
                if line: 
                    data.append(float(line))

    data.sort()
    #print("len data: ", len(data))
    thresholds = []

    for i in range(len(Gts)):
        gt = Gts[i]
        percentileIdx = int(len(data)*gt)
        #print('percentileIdx: ', percentileIdx, 'len(data)', len(data), 'gt', gt, 'i', i)
        if gt == 0:
            continue
        elif gt == 1:
            continue
        else:
            thresholds.append(data[percentileIdx])

    return thresholds

def getGtsFromThresholds(thresholds, file='test.txt'):
    if file != None:
        data = []
        with open(file, 'r') as f:
            fst= True
            ct = 0 
            for line in f:
                if fst:
                    fst = False
                    continue
                line = line.replace(',', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                line = line.replace(' ', '')
                line = line.replace('\n', '')
                line = line.upper()
                # if ct < 10 and line: 
                #     print(line, float(line))
                #     ct += 1
                if line: 
                    data.append(float(line))

    data.sort()
    data = np.array(data)
    data = np.round(data, 2)
    #print("len data: ", len(data))
    Gts = []

    for i in range(len(thresholds)):
        thres = thresholds[i]
        print(np.where(data == thres))
        idx = np.where(data == thres)[0][0]
        # percentileIdx = int(len(data)*gt)
        # #print('percentileIdx: ', percentileIdx, 'len(data)', len(data), 'gt', gt, 'i', i)
        # if gt == 0:
        #     continue
        # elif gt == 1:
        #     continue
        # else:
        Gts.append(idx/len(data))

    return Gts

def getEpsilonsFromGts(Gts, k=4, epsilon=.01/1000, z=1.5): 
    m = -min(1, 1/z)
    eps = []
    Gts.append(1)
    Garr = [Gts[i] - Gts[i-1] if i > 0 else Gts[i] for i in range(k - 1)]
    for i in range(len(Gts)):
        gt = Gts[i]
        prevgt = Gts[i-1] if i > 0 else 0
        
        G = gt - prevgt

        
        powArr = np.power(Garr, 1/(m-1))
        e = k*epsilon*pow(G, 1/(m-1)) / np.sum(powArr)
        eps.append(e)
    return eps

def getSizeProportionsFromEpsilons(epsilons, z, k=4):
    m = -min(1, 1/z)
    propVals = []
    for i in range(len(epsilons)):
        e = epsilons[i]
        propVals.append(e**m)
    return propVals/np.sum(propVals)



def getSizeProportionsFromThresholds(thresholds, z, k=4):
    Gts = getGtsFromThresholds(thresholds)
    epsilons = getEpsilonsFromGts(Gts, k=k, z=z)
    return getSizeProportionsFromEpsilons(epsilons, z=z, k=k)



#thresholds = getThresholdsFromGts(Gts, file='test.txt')

thresholds = [1.2, 1.9]
Gts = getGtsFromThresholds(thresholds, file='test.txt')
epsilons = getEpsilonsFromGts(Gts, k=3)#, epsilon=.01/1000, z=1.5)
sizes = getSizeProportionsFromEpsilons(epsilons, z=1.75, k=3)




print("Thresholds: ", thresholds, "epsilons: ", epsilons, "sizes: ", sizes)