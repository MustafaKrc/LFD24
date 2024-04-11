import numpy as np
np.random.seed(42)
class LogisticRegression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Constructor assumes an x_train matrix in which each column contains an instance.
        Vector y_train contains binary labels for each instance (0 or 1).

        Constructor initializes the weights W and B, alpha, and a one-vector Y containing the labels
        of the training set.
        """
        np.random.seed(42)
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._m = x_train.shape[1]
        
        self._W = np.random.randn(1, x_train.shape[0]) * 0.02
        self._B = 0
        self._Y = y_train.reshape(1, -1)  # Binary labels, no need for one-hot encoding
        self._alpha = 0.05

    def sigmoid(self, Z):
        """
        Computes the sigmoid value for all values in vector Z.
        """
        ##############################################################################
        # TODO: Compute the sigmoid value for all values in input vector             #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        pass
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return sigmoid_val

    def h_theta(self, X):
        """
        Computes the value of the hypothesis according to the logistic regression rule.
        """
        ##############################################################################
        # TODO: Implement Logistic Regression on input X.                            #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        pass
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return h_theta

    def train_binary_classification(self, iterations):
        """
        Performs a number of iterations of gradient descent equal to the parameter passed as input.
        Returns a list with the percentage of instances classified correctly in the training and test sets.
        """
        classified_correctly_train_list = []
        classified_correctly_test_list = []
        ##############################################################################
        # TODO: Implement stochastic gradient descent algorithm on minimizing Mean Squared      #
        # Error cost. You need to obtain partial derivatives before coding up your   # 
        # solutions. This method returns correctly classified train and test samples # 
        # computed at each iteration (Or each N iterations.)                         #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        pass
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################       
        return classified_correctly_train_list, classified_correctly_test_list


    def train_binary_classification_sgd_cross_entropy(self, iterations):
        """
        Performs a number of iterations of stochastic gradient descent (SGD) using cross-entropy loss.
        Returns a list with the percentage of instances classified correctly in the training and test sets.
        """
        classified_correctly_train_list_sgd = []
        classified_correctly_test_list_sgd = []

        ##############################################################################
        # TODO: Implement stochastic gradient descent algorithm on minimizing Cross  #
        # Entropy cost.                                                              #
        # You need to obtain partial derivatives before coding up your               # 
        # solutions. This method returns correctly classified train and test samples # 
        # computed at each iteration (Or each N iterations.)                         #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        pass
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        return classified_correctly_train_list_sgd, classified_correctly_test_list_sgd

