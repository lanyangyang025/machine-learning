from sklearn import linear_model
import numpy as np
import utils

class one_vs_allLogisticRegressor:

    def __init__(self,labels):
        self.theta = None
        self.labels = labels

    def train(self,X,y,reg):
        
        """
        Use sklearn LogisticRegression for training K classifiers in one-vs-rest mode
        Read the documentation carefully and choose an appropriate solver. Choose
        the L2 penalty. Remember that the X data has a column of ones prepended to it.
        Set the appropriate flag in logisticRegression to cover that.
        
        X = m X (d+1) array of training data. Assumes X has an intercept column
        y = 1 dimensional vector of length m (with K labels)
        reg = regularization strength

        Computes coefficents for K classifiers: a matrix with (d+1) rows and K columns
           - one theta of length d for each class
       """
        
        m,dim = X.shape
        theta_opt = np.zeros((dim,len(self.labels)))

        ###########################################################################
        # Compute the K logistic regression classifiers                           #
        # TODO: 7-9 lines of code expected                                        #
        ###########################################################################
        for i in range(len(self.labels)):
            y_new1=np.zeros(len(y))+i
            y_new=(y==y_new1)
#            for j in range(len(y)):
#                if y[j]==i:
#                    y_new[j]=1
            sk_logreg = linear_model.LogisticRegression(C=1.0/reg,solver='lbfgs',fit_intercept=False)   
            sk_logreg.fit(X,y_new)
            if i==0:
                theta_1=sk_logreg.coef_[0]
            else:
                theta_1=np.vstack([theta_1,sk_logreg.coef_[0]])
        theta_opt=theta_1.T
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        self.theta = theta_opt


    def predict(self,X):
        """
        Use the trained weights of this linear classifier to predict labels for'l2'        data points.

        Inputs:
        - X: m x (d+1) array of training data. 

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
          array of length m, and each element is a class label from one of the
          set of labels -- the one with the highest probability
        """

        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # Compute the predicted outputs for X                                     #
        # TODO: 1-2 lines of code expected                                        #
        ###########################################################################
        y_pred1=np.dot(X,self.theta)
        for i in range(X.shape[0]):
            y_pred[i]=np.argmax(y_pred1[i,:])
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

