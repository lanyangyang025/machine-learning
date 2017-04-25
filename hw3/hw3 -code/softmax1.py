import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  
  K=grad.shape[1]
  for i in xrange(m):
    X_i =  X[i,:]
    score_i = np.dot(X_i,theta)
    s_max = -score_i.max()
    exp_score = np.exp(score_i+s_max)
    total_score = np.sum(exp_score , axis = 0)
    for k in xrange(K):
      grad[:,k] = grad[:,k] -X_i*(y[i]==k) + (exp_score[k] / total_score) * X_i
    numerator = np.exp(score_i[y[i]]+s_max)
    denom = np.sum(np.exp(score_i+s_max),axis = 0)
    J = J -np.log(numerator / float(denom))

  J = J / float(m) + 0.5 * reg * np.sum(theta**2) / float(m)
  grad = grad / float(m) + reg * theta / float(m)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################

  score_i = np.dot(theta.T,X.T)
  s_max = -np.max(score_i,axis=0)
  exp_score = np.exp(score_i+s_max)

  theta_new=theta[:,y]
  J=np.sum(np.multiply(theta_new,X.T)/float(m))
  J=J-np.sum((np.log(np.sum(exp_score,axis=0))-s_max)/float(m))
#  J=J-np.sum(np.log(np.sum(np.exp(score_i),axis=0))/float(m))
  J=-1.0*J+0.5*reg*np.sum(theta**2)/float(m)

  K=grad.shape[1]
  temp_1=exp_score
  temp_2=np.sum(exp_score,axis=0)
  temp=np.divide(temp_1,temp_2)
  Grad=temp.T
  Grad[np.arange(m),y] += -1.0
  grad = np.dot(X.T,Grad) / float(m) + reg*theta/float(m)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
