import numpy as np

np.random.seed(42)  # Set random seed if needed
class SoftmaxPerceptron:
    def __init__(self, weights, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights_ = weights # weights_ is weights with bias
        self.bias = None # why this wasnt even here in init...
        
    def calculate_accuracy(self, h, y):
        """
        Calculates the percentage of correctly classified instances.
        """
        m = len(y)
        h = h.reshape(-1, 1)
        y = y.reshape(-1, 1)
        h[h >= 0] = 1
        h[h < 0] = -1
        tp = np.sum((h == 1) & (y == 1))
        tn = np.sum((h == -1) & (y == -1))
        accuracy = (tp + tn) / m

        return accuracy * 100
    
    def calculate_recall(self, h, y):
        """
        Calculates the recall.
        """
        h = h.reshape(-1, 1)
        y = y.reshape(-1, 1)
        h[h >= 0] = 1
        h[h < 0] = -1
        tp = np.sum((h == 1) & (y == 1))
        fn = np.sum((h == -1) & (y == 1))

        if tp + fn == 0:
            if np.sum(h == 1) == 0 and np.sum(y == 1) == 0:
                return 100  # If there are no positive predictions and no positive actual values, recall is 1 (or 100%)
            else:
                return 0

        recall = tp / (tp + fn)
        return recall * 100
    
    def calculate_precision(self, h, y):
        """
        Calculates the precision.
        """
        h = h.reshape(-1, 1)
        y = y.reshape(-1, 1)
        h[h >= 0] = 1
        h[h < 0] = -1
        tp = np.sum((h == 1) & (y == 1))
        fp = np.sum((h == 1) & (y == -1))

        if tp + fp == 0:
            if np.sum(h == 1) == 0 and np.sum(y == 1) == 0:
                return 100  # If there are no positive predictions and no positive actual values, precision is 1 (or 100%)
            else:
                return 0

        precision = tp / (tp + fp)
        return precision * 100
    
    def calculate_f1_score(self, h, y):
        """
        Calculates the F1 score.
        """
        recall = self.calculate_recall(h, y)
        precision = self.calculate_precision(h, y)

        if recall + precision == 0:
            return 0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

        
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
            sample_x = X[:, rand_index].reshape(-1, 1)
            sample_y = y[rand_index].reshape(-1, 1)
            
            xt_w = np.dot(sample_x.T, self.weights) + self.bias
            sigmoid = -sample_y * (1 - (1 / (1 + np.exp(-sample_y * xt_w))))

            grad_w = np.dot(sample_x, sigmoid) / np.size(sample_y)
            grad_b = np.sum(sigmoid, axis= 0) / np.size(sample_y)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Calculate the percentage of correctly classified instances
            train_accuracy = self.calculate_accuracy(h_train, y)
            train_precision = self.calculate_precision(h_train, y)
            train_recall = self.calculate_recall(h_train, y)
            train_f1 = self.calculate_f1_score(h_train, y)
            metrics = [train_accuracy, train_precision, train_recall, train_f1]

            # Append the accuracies to the lists
            accuracy_history.append(metrics)

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

        y = y.reshape(-1, 1)

        #print(self.weights.shape, self.bias.shape, X.shape, y.shape)
        #print(self.weights)
        #print()
        #print(self.bias)
        for _ in range(self.max_iter):
            xt_w = np.dot(X.T, self.weights) + self.bias
            sigmoid = -y * (1 - (1 / (1 + np.exp(-y * xt_w))))

            grad_w = np.dot(X, sigmoid) / num_samples
            grad_w = grad_w + 2 * regularization_strength * self.weights


            grad_b = np.sum(sigmoid, axis= 0) / num_samples

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            
            
            h_train = self.predict(X)
            # Calculate the percentage of correctly classified instances
            train_accuracy = self.calculate_accuracy(h_train, y)
            train_precision = self.calculate_precision(h_train, y)
            train_recall = self.calculate_recall(h_train, y)
            train_f1 = self.calculate_f1_score(h_train, y)
            metrics = [train_accuracy, train_precision, train_recall, train_f1]
            
            # Append the accuracies to the lists
            accuracy_history.append(metrics)

        return accuracy_history



