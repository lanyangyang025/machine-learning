import numpy as np
import math
import random


class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self,c,Mu):
        x_history=[]
    	for i in range(10000):
          x_history.append(np.array([Mu[0]+c[0,0]*random.normalvariate(0,1), Mu[1]+ c[1,1]*random.normalvariate(0,1)])) 

    	return x_history


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self,mu,sigma):
    	self.mu=mu
    	self.sigma=sigma
     
    def output(self,x):
    	p=1/(self.sigma * (2*math.pi)**(1/2)) * math.exp(-(x-self.mu)**2/2*self.sigma^2)
    	return p
    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector 
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self,Mu,Sigma):
        self.Mu=Mu
        self.Sigma=Sigma
    def output(self,a):
        D=a.shape[0]      
        p=1.0/(np.linalg.det(self.Sigma)**(0.5) * (2.0*math.pi)**(D/2.0))* math.exp(-0.5*np.dot(np.dot(a.reshape(1,2),np.linalg.inv(self.Sigma)),(a-self.Mu) ))
        #p= np.dot(z,np.linalg.inv(self.Sigma))   +1/(np.linalg.det(self.Sigma)**(1/2) * (2*math.pi)**(D/2))* 
        return p
#1/(np.linalg.det(self.Sigma)**(1/2) * (2*math.pi)**(D/2))* math.exp( -(1/2) *np.dot( np.dot(z,np.linalg.inv(self.Sigma)),y ))



# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self,ap):
    	self.ap=ap
    def output(self):
    	return self.ap



# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):
    
    # Initializes a mixture-model object parameterized by the
    # atomic probabilities vector ap (numpy.array of size k) and by the tuple of 
    # probability models pm
    def __init__(self):
        pass
    def output(self):
        sum=0
        y1=ProbabilityModel().sample(np.eye(2),np.array([1,1]))
        y2=ProbabilityModel().sample(np.eye(2),np.array([1,-1]))
        y3=ProbabilityModel().sample(np.eye(2),np.array([-1,-1]))
        y4=ProbabilityModel().sample(np.eye(2),np.array([-1,1]))
#       p=0.25*MultiVariateNormal(np.array([1,1]),np.eye(2)).output(y1)+0.25*MultiVariateNormal(np.array([1,-1]),np.eye(2)).output(y2)+0.25*MultiVariateNormal(np.array([-1,-1]),np.eye(2)).output(y3)+0.25*MultiVariateNormal(np.array([-1,1]),np.eye(2)).output(y4)
        for j in range (len(y1)):
            z=y1[j]
            if (z[0]-0.1)**2+(z[1]-0.2)**2<=1:
            	sum=sum+1
        for j in range (len(y2)):
            z=y2[j]
            if (z[0]-0.1)**2+(z[1]-0.2)**2<=1:
            	sum=sum+1
        for j in range (len(y3)):
            z=y3[j]
            if (z[0]-0.1)**2+(z[1]-0.2)**2<=1:
            	sum=sum+1
        for j in range (len(y4)):
            z=y4[j]
            if (z[0]-0.1)**2+(z[1]-0.2)**2<=1:
            	sum=sum+1
        return sum/40000.0
        

MixtureModel().output() #mixture 
#x=ProbabilityModel().sample()
#y=UnivariateNormal(x).output(x)
#y=np.array(x)
#z=np.array([x,x])