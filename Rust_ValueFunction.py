#Rust Implementation of value

#parameters from table IX in Rust(1987), p.1021
#params=beta,RC,theta11,theta30,theta31

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

#params from table IX
params=[.9999,11.7270,4.8259,0.3010,0.6884,0.0106]


maxmil=450000
intervals=5000

#utility function for i=0,1
def utility(x,i,params):
    return -0.001*params[2]*x*(1-i)-params[1]*i

#inner value function to loop over 
def innerval(x,params,EV):
    K=max(EV[0])
    value=np.log((np.exp(utility(x,0,params) + params[0]*EV[0,:] - K) + 
                 np.exp(utility(0,1,params) + params[0]*EV[1,:]) - K)) + K
    return value

#value iteration function
def valueiter(params,maxmil,intervals):
    diffs=[]
    p=params[-3:-1] #transition probabilities
    K=int(maxmil/intervals)
    P1=diags(p,[0,1],shape=(K,K)).todense()
    P1=P1.T
    x=np.arange(K)
    EV=np.zeros((2,K))
    tol=1e-08;maxIter=10000;iterNum=0;dif=1
    while dif>tol and iterNum<maxIter:
        EV1=innerval(x,params[:3],EV)
        EVTemp=np.vstack((np.dot(EV1,P1),np.array([np.dot(EV1[:2],p)]*90)))
        dif=np.max(abs(EVTemp-EV))
        EV=np.array(EVTemp)
        iterNum+=1
        diffs.append(dif)
    return EV,iterNum,diffs

def valueiterSeidel(params,maxmil,intervals):
    diffs=[]
    p=params[-3:-1] #transition probabilities
    K=int(maxmil/intervals)
    P1=diags(p,[0,1],shape=(K,K)).todense()
    P1=P1.T
    x=np.arange(K)
    EV=np.zeros((2,K))
    EV1=np.zeros(K)
    tol=1e-08;maxIter=10000;iterNum=0;dif=1
    while dif>tol and iterNum<maxIter:
        C=max(EV[0])
        EV11=np.zeros_like(EV1)
        #loop in Seidel style
        for i in range(len(EV11)):
            EV1[i]=np.log((np.exp(utility(x[i],0,params) + params[0]*EV[0,i] - C) + 
               np.exp(utility(0,1,params) + params[0]*EV[1,i]) - C)) + C
            EV11[i]=np.dot(EV1,P1[:,i])
        EVTemp=np.vstack((EV11,np.array([np.dot(EV1[:2],p)]*90)))
        dif=np.max(abs(EVTemp-EV))
        diffs.append(dif)
        EV=np.array(EVTemp)
        iterNum+=1
    return EV,iterNum,diffs
#%%
#see results    

values,iters,diffs=valueiter(params,maxmil,intervals)
valuesS,itersS,diffsSeidel=valueiterSeidel(params,maxmil,intervals)

plt.plot(diffs)
plt.plot(diffsSeidel)
plt.gca().legend(("Value Iteration","Gauss-Seidel VI"))
plt.show()

print('VI iterations:{}, GS VI iterations:{}'.format(iters,itersS))

# check: results of both algo are same
np.allclose(values,valuesS,1e-05)
