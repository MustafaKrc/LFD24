import numpy as np

np.random.seed(42)  # Set random seed if needed
class SoftmaxPerceptron:
    def __init__(self, weights, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights_ = weights
        
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
    
    def model(self, x):
        a = self.bias + np.dot(x.T, self.weights)
        return a.T
    
    def compute_gradient(self, x, y):
        """
        Computes the gradients of the weights and bias for SoftMax cost.
        """
        xb = np.insert(x, 0, 1).reshape(-1, 1) # calculate the gradient with bias
        x = x.reshape(-1, 1) 
        y = y.reshape(-1, 1)

        gradient = xb @ ((np.exp(-y*self.model(x))/(np.exp(-y*self.model(x)) + 1)) * y)/float(np.size(y))

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return -1 * gradient.T
        
    def fit_sgd(self, X, y):
        np.random.seed(42)

        num_features, num_samples = X.shape  # Get the shape of the input data

        # Initialize weights
        self.weights = self.weights_[:, 1:].reshape(-1, 1)
        self.bias = self.weights_[:, 0]
        
        
        accuracy_history = []  # Initialize loss history

        ##############################################################################
        # TODO: Implement stochastic gradient descent (SGD) to train the logistic    #
        # regression model.                                                          #
        ##############################################################################
        # This function implements stochastic gradient descent (SGD) to train a      #
        # perceptron model. It iterates over the training samples for a              #
        # specified number of iterations (given by max_iter), updating the model     #
        # parameters (weights and bias) after processing each sample. SGD is a       #
        # variant of gradient descent that updates the parameters based on the       #
        # gradient of the loss function computed with respect to each individual     #
        # sample. This makes it computationally more efficient than batch gradient   #
        # descent, particularly for large datasets. The function computes the        #
        # logistic loss and its gradient for each sample, and updates the weights    #
        # and bias accordingly. The accuracy on the training set is computed after   #
        # each iteration and stored in the accuracy_history list. The function       #
        # returns this accuracy history to monitor the training progress.            #
        ##############################################################################

        for _ in range(self.max_iter):
            h_train = self.predict(X)
            
            # Compute the cost
            #cost_train = self.compute_cost(h_train, y)
            #print(cost_train, cost_test)

            # pick random samples
            rand_index = np.random.randint(0, num_samples)
            sample_x = X[:, rand_index]
            sample_y = y[rand_index]
            gradient = self.compute_gradient(sample_x, sample_y)

            # Update the weights and bias
            self.weights -= self.learning_rate * gradient[:, 1:].reshape(-1, 1)
            self.bias -= float(self.learning_rate * gradient[:, 0])
            
            # Calculate the percentage of correctly classified instances
            train_accuracy = self.calculate_accuracy(h_train, y)
            
            # Append the accuracies to the lists
            accuracy_history.append(train_accuracy)

        return accuracy_history

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")

        predictions = np.sign(np.dot(X.T, self.weights).flatten() + self.bias)
        return predictions
       
        
    def fit_gd_regularized(self, X, y, regularization_strength):
        np.random.seed(42)
        num_features, num_samples = X.shape  # Get the shape of the input data
        X_with_bias = X
        self.weights = self.weights_[1:].reshape(-1, 1)
        self.bias = self.weights_[0]

        accuracy_history = []  # Initialize empty list to store accuracy history
        ##############################################################################
        # TODO: Implement regularized gradient descent (GD) to train the softmax      #
        # perceptron model. The regularization_strength parameter controls the        #
        # strength of regularization applied to the model. Regularization helps      #
        # prevent overfitting by penalizing large parameter values. The function      #
        # iterates over the training samples for a specified number of iterations    #
        # (given by max_iter), updating the model parameters (weights and bias)       #
        # after processing each sample. The loss function used is the cross-entropy  #
        # loss with L2 regularization. It computes the loss and its gradient for     #
        # each sample, including a regularization term to penalize large weights.    #
        # The weights excluding the bias term are regularized using L2 regularization.#
        # The accuracy on the training set is computed after each iteration and      #
        # stored in the accuracy_history list. The function returns this accuracy    #
        # history to monitor the training progress.                                  #
        ##############################################################################


