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
        
        sigmoid_val = 1 / (1 + np.exp(-Z))
        
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
        ###########################################5###################################
        
        Z = np.dot(self._W, X) + self._B
        h_theta = self.sigmoid(Z)
        
        ##############################################################################
        ##############################################################################
        return h_theta
    
    def compute_cost_MSE(self, h, y):
        """
        Computes the cost function for all instances.
        """

        return np.mean(np.sum(np.square(h - y)))
    
    def compute_cost_CE(self, h, y):
        """
        Computes the cost function for all instances.
        """
        y = y.reshape(y.shape[0], 1)
        cost0 = y.dot(np.log(h))
        cost1 = (1-y).dot(np.log(1-h))
        cost = -((cost1 + cost0))/y.shape[0]
        return np.mean(np.sum(cost))

    
    def calculate_accuracy(self, h, y):
        """
        Calculates the percentage of correctly classified instances.
        """

        m = len(y)
        h = h.reshape(-1, 1)
        y = y.reshape(-1, 1)
        h[h >= 0.5] = 1
        h[h < 0.5] = 0
        accuracy = (1 / m) * np.sum(h == y)

        return accuracy * 100


    def compute_gradients(self, x, y):
        """
        Computes the gradients of the weights and bias using SGD.
        """
        # same gradient function for both cost functions
        
        x = x.reshape(-1, 1)

        gradient = -1 * (x @ (y - self.h_theta(x))) / x.T.shape[0]

        return gradient


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

        for _ in range(iterations):

            h_train = self.h_theta(self._x_train)
            h_test = self.h_theta(self._x_test)
            
            # Compute the cost
            cost_train = self.compute_cost_MSE(h_train, self._y_train)
            cost_test = self.compute_cost_MSE(h_test, self._y_test)

            print("MSE COST: \n", cost_train)
            #print(cost_train, cost_test)

            # pick random samples
            sample_x = self._x_train[:, np.random.randint(0, self._m)]
            sample_y = self._y_train[np.random.randint(0, self._m)]
            gradient = self.compute_gradients(sample_x, sample_y)
            
            # Update the weights and bias
            self._W -= self._alpha * gradient[1:]
            self._B -= self._alpha * gradient[0]
            
            # Calculate the percentage of correctly classified instances
            train_accuracy = self.calculate_accuracy(h_train, self._y_train)
            test_accuracy = self.calculate_accuracy(h_test, self._y_test)
            
            # Append the accuracies to the lists
            classified_correctly_train_list.append(train_accuracy)
            classified_correctly_test_list.append(test_accuracy)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################       
        return classified_correctly_train_list, classified_correctly_test_list


    def train_binary_classification_cross_entropy(self, iterations): #changed name so that it is compatible with the jupyter notebook
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
        
        for _ in range(iterations):
            
            h_train = self.h_theta(self._x_train)
            h_test = self.h_theta(self._x_test)
            
            # Compute the cost
            cost_train = self.compute_cost_CE(h_train, self._y_train)
            cost_test = self.compute_cost_CE(h_test, self._y_test)

            print("CE COST: \n", cost_train)
            #print(cost_train, cost_test)
            
            #pick random samples
            sample_x = self._x_train[:, np.random.randint(0, self._m)]
            sample_y = self._y_train[np.random.randint(0, self._m)]
            gradient = self.compute_gradients(sample_x, sample_y)
            
            # Update the weights and bias
            self._W -= self._alpha * gradient[1:]
            self._B -= self._alpha * gradient[0]
            
            # Calculate the percentage of correctly classified instances
            train_accuracy = self.calculate_accuracy(h_train, self._y_train)
            test_accuracy = self.calculate_accuracy(h_test, self._y_test)
            
            # Append the accuracies to the lists
            classified_correctly_train_list_sgd.append(train_accuracy)
            classified_correctly_test_list_sgd.append(test_accuracy)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        return classified_correctly_train_list_sgd, classified_correctly_test_list_sgd

