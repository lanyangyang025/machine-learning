from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
import plot_utils


#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    mu = np.zeros((X.shape[1],))
    sigma = np.ones((X.shape[1],))
    X_norm = np.zeros(X.shape)
    
    for i in range(X.shape[1]):
        mu[i,]=np.mean(X[:,i])
        sigma[i,]=np.std(X[:,i])
    X_norm = (X - mu) / sigma
    ########################################################################
    return X_norm, mu, sigma


#############################################################################
#  Plot the learning curve for training data (X,y) and validation set       #
# (Xval,yval) and regularization lambda reg.                                #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 7 lines of code expected                                                #
    ###########################################################################
    reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
    
    for num1 in range(0,num_examples):
        theta = reglinear_reg1.train(X[0:num1+1,:],y[0:num1+1],reg,num_iters=1000)
        hypothesis=np.dot(X[0:num1+1,:],theta)
        error_train[num1]=np.sum((hypothesis-y[0:num1+1])**2)/(2*(num1+1))#+np.sum(theta**2)*reg/(2*(num1+1))
        hypothesis1=np.dot(Xval,theta)
        error_val[num1]=np.sum((hypothesis1-yval)**2)/(2*len(yval))#+np.sum(theta**2)*reg/(2*len(yval))
    ###########################################################################

    return error_train, error_val

#############################################################################
#  Plot the validation curve for training data (X,y) and validation set     #
# (Xval,yval)                                                               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#                                                                           #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def validation_curve(X,y,Xval,yval):
  
  reg_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
  error_train = np.zeros((len(reg_vec),))
  error_val = np.zeros((len(reg_vec),))

    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 5 lines of code expected                                                #
    ###########################################################################
  reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
    
  for i in range(len(reg_vec)):
    theta = reglinear_reg1.train(X,y,reg_vec[i],num_iters=1000)
    error_train[i]=np.sum((np.dot(X,theta)-y)**2)/(2*(len(y)))
    error_val[i]=np.sum((np.dot(Xval,theta)-yval)**2)/(2*len(yval))
  
  return reg_vec, error_train, error_val

import random

#############################################################################
#  Plot the averaged learning curve for training data (X,y) and             #
#  validation set  (Xval,yval) and regularization lambda reg.               #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     y: vector of length N which are values corresponding to X             #
#     Xval: M x D where N is the number of rows and D is the number of      #
#           features                                                        #
#     yval: vector of length N which are values corresponding to Xval       #
#     reg: regularization strength (float)                                  #
#     Output: error_train: vector of length N-1 corresponding to training   #
#                          error on training set                            #
#             error_val: vector of length N-1 corresponding to error on     #
#                        validation set                                     #
#############################################################################

def averaged_learning_curve(X,y,Xval,yval,reg):
    num_examples,dim = X.shape
    error_train = np.zeros((num_examples,))
    error_val = np.zeros((num_examples,))
    
    ###########################################################################
    # TODO: compute error_train and error_val                                 #
    # 10-12 lines of code expected                                            #
    ###########################################################################
    reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
    times=50 #could change
    
    for num1 in range(0,num_examples):
        for num2 in range(times):
            list=random.sample(np.arange(0,num_examples),num1+1)
            theta = reglinear_reg1.train(X[list,:],y[list],reg,num_iters=1000)
            hypothesis=np.dot(X[list,:],theta)
            error_train[num1]=error_train[num1]+np.sum((hypothesis-y[list])**2)/(2*(num1+1))
            hypothesis1=np.dot(Xval,theta)
            error_val[num1]=error_val[num1]+np.sum((hypothesis1-yval)**2)/(2*len(yval))   
        error_train[num1]=error_train[num1]/times
        error_val[num1]=error_val[num1]/times

    ###########################################################################
    return error_train, error_val


#############################################################################
# Utility functions
#############################################################################
    
def load_mat(fname):
  d = scipy.io.loadmat(fname)
  X = d['X']
  y = d['y']
  Xval = d['Xval']
  yval = d['yval']
  Xtest = d['Xtest']
  ytest = d['ytest']

  # need reshaping!

  X = np.reshape(X,(len(X),))
  y = np.reshape(y,(len(y),))
  Xtest = np.reshape(Xtest,(len(Xtest),))
  ytest = np.reshape(ytest,(len(ytest),))
  Xval = np.reshape(Xval,(len(Xval),))
  yval = np.reshape(yval,(len(yval),))

  return X, y, Xtest, ytest, Xval, yval









