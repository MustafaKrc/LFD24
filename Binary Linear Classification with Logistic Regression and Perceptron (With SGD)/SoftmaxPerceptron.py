np.random.seed(42)  # Set random seed if needed
class SoftmaxPerceptron:
    def __init__(self, weights, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights_ = weights
        

        
    def fit_sgd(self, X, y):
        np.random.seed(42)

        num_features, num_samples = X.shape  # Get the shape of the input data

        # Initialize weights
        self.weights = self.weights_[1:].reshape(-1, 1)
        self.bias = self.weights_[0]
        
        
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


