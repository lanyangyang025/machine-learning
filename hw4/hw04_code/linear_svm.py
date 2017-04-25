import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  J=0.5*np.sum(theta**2)/m
  J=J+C*np.sum(np.maximum(0,1-np.multiply(y,(np.dot(X,theta)))))/m
 
  grad=theta/m
  temp_1=np.dot(X,theta)
  temp_2=np.multiply(y,temp_1)

  temp_3=y[temp_2<1]
  temp_4=X[temp_2<1,:]
  temp_5=np.dot(temp_4.T,temp_3)
  grad=grad-temp_5*C/m


#  for j in range(d):
#  	grad[j]=float(theta[j]/m)
#  	for k in range(m):
#	  	if (y[k]*(np.dot(theta,X[k,:]))<1):
#	  		grad[j]=grad[j]-float(C*y[k]*X[k,j]/m)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  delta = 1.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero

  # compute the loss function

  K = theta.shape[1]
  m = X.shape[0]
  J = 0.0
  for i in xrange(m):
	scores = X[i,:].dot(theta)
	correct_class_score = scores[y[i]]
	for j in xrange(K):
		if j == y[i]:
			continue
		margin = max(0,scores[j] - correct_class_score + delta)
		J += margin
		if margin > 0 and j!=y[i]:		
			dtheta[:,j] = dtheta[:,j]+X[i,:]
			dtheta[:,y[i]] = dtheta[:,y[i]]-X[i,:]


  # Right now the loss is a sum over all training examples, but we want it
  # To be an average instead so we divide by num_train.
  J /= m
  dtheta = dtheta/m
  # Add regularization to the loss.
  J += 0.5 * reg * np.sum(theta * theta)
  dtheta =dtheta + reg*theta

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  #############################################################################
  
  # compute the loss function

  K = theta.shape[1]
  m = X.shape[0]

  scores = X.dot(theta)
  correct_class_score = scores[np.arange(m),y]
  margin = np.maximum(0, scores - correct_class_score[np.newaxis].T + delta)
  margin[np.arange(m), y] = 0
  J = np.sum(margin)

  J /= m
  J += 0.5 * reg * np.sum(theta * theta)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  num_yi = np.sum(margin>0 , axis=1)
  temp = np.zeros(margin.shape) 
  temp[margin>0] = 1
  temp[np.arange(m), y] = -num_yi
  dtheta = np.dot(X.T,temp)

  dtheta = dtheta/m
  dtheta = dtheta + reg*theta

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta
