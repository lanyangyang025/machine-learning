import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import cross_validation
import scipy.io

######################################################################################
#   The sigmoid function                                                             #
#     Input: z: can be a scalar, vector or a matrix                                  #
#     Output: sigz: sigmoid of scalar, vector or a matrix                            #
#     TODO: 1 line of code expected                                                  #
######################################################################################

def sigmoid (z):
    sig = np.zeros(z.shape)
    # Your code here
    sig=1/(1+np.e**(-z))
    # End your code

    return sig

######################################################################################
#   The log_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every element x replaced by 1 + log(x)                   #
#     TODO: 1 line of code expected                                                  #
######################################################################################

def log_features(X):
    logf = np.zeros(X.shape)
    # Your code here
    logf=np.log(1+X)
    # End your code
    return logf

######################################################################################
#   The std_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every column with zero mean, unit variance               #
######################################################################################

def std_features(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

######################################################################################
#   The bin_features transform                                                       #
#     Input: X: a data matrix                                                        #
#     Output: a matrix with every element x replaced by 1 if x > 0 else 0            #
#     TODO: 1 line of code expected                                                  #
######################################################################################

def bin_features(X):
    tX = np.zeros(X.shape)
    # your code here
    tX=np.where(X > 0, 1, 0)
    # end your code
    return tX

######################################################################################
#   The select_lambda_crossval function                                              #
#     Inputs: X: a data matrix                                                       #
#             y: a vector of labels                                                  #
#             lambda_low, lambda_high,lambda_step: range of lambdas to sweep         #
#             penalty: 'l1' or 'l2'                                                  #
#     Output: best lambda selected by crossvalidation for input parameters           #
######################################################################################

# Select the best lambda for training set (X,y) by sweeping a range of
# lambda values from lambda_low to lambda_high in steps of lambda_step
# pick sklearn's LogisticRegression with the appropriate penalty (and solver)

# 20-25 lines of code expected

# For each lambda value, divide the data into 10 equal folds
# using sklearn's cross_validation KFold function.
# Then, repeat i = 1:10:
#  1. Retain fold i for testing, and train logistic model on the other 9 folds
#  with that lambda
#  2. Evaluate accuracy of model on left out fold i
# Accuracy associated with that lambda is the averaged accuracy over all 10
# folds.
# Do this for each lambda in the range and pick the best one
#
def select_lambda_crossval(X,y,lambda_low,lambda_high,lambda_step,penalty):

    best_lambda = lambda_low

    # Your code here
    # Implement the algorithm above.
    from logistic_regressor import RegLogisticRegressor
    reg_lr1 = RegLogisticRegressor()
    max_accuracy=0
    reg=lambda_low
    while (reg<lambda_high):
        kf = cross_validation.KFold(len(y),10,shuffle=False)
        accuracy=[]
        for train_index, test_index in kf:
            X_train=y_train=X_test=y_test=[];
            X_train=X[train_index[0],:]
            y_train=np.append(y_train,y[train_index[0]])
            X_test=X[test_index[0],:]
            y_test=np.append(y_test,y[test_index[0]])
            for i in range(1,len(train_index)):
                X_train=np.vstack((X_train,X[train_index[i],:].T))
                y_train=np.append(y_train,y[train_index[i]])
            for i in range(1,len(test_index)):
                X_test=np.vstack((X_test,X[test_index[i],:].T))
                y_test=np.append(y_test,y[test_index[i]])
            
            XX_train = np.vstack([np.ones((X_train.shape[0],)),X_train.T]).T
            XX_test = np.vstack([np.ones((X_test.shape[0],)),X_test.T]).T
            theta_opt = reg_lr1.train(XX_train,y_train,reg,num_iters=1000,norm=False)
            reg_lr1.theta=theta_opt 
            predy = reg_lr1.predict(XX_test)
            num_correct = np.sum(predy == y_test) 
            accuracy.append(float(num_correct)/len(y_test))
        if np.average(accuracy)>max_accuracy:
            max_accuracy=np.average(accuracy)
            best_lambda =reg
        reg=reg+lambda_step
      
    
    # end your code

    return best_lambda
    


######################################################################################

def load_mat(fname):
  d = scipy.io.loadmat(fname)
  Xtrain = d['Xtrain']
  ytrain = d['ytrain']
  Xtest = d['Xtest']
  ytest = d['ytest']

  return Xtrain, ytrain, Xtest, ytest


def load_spam_data():
    data = scipy.io.loadmat('spamData.mat')
    Xtrain = data['Xtrain']
    ytrain1 = data['ytrain']
    Xtest = data['Xtest']
    ytest1 = data['ytest']

    # need to flatten the ytrain and ytest
    ytrain = np.array([x[0] for x in ytrain1])
    ytest = np.array([x[0] for x in ytest1])
    return Xtrain,Xtest,ytrain,ytest

    
