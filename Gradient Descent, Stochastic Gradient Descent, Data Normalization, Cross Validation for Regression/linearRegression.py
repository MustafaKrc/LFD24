import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class gradientDescent():
    def __init__(self, x, y, w, lr, num_iters, add_bias=True):
        self.x = np.c_[np.ones((x.shape[0], 1)), x] if add_bias else x.copy()
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w.copy()
        self.weight_history = [self.w]
        self.cost_history = [np.sum(np.square(self.predict(self.x)-self.y))/self.x.shape[0]]

    def cost(self):
        """
            Utitlity method for calculating the MSE for the current set of weights
        """
        return np.sum(np.square(self.predict(self.x) - self.y)) / self.x.shape[0]
        
    def gradient(self):
        gradient = np.zeros_like(self.w)
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code
        
        gradient = -1 * (self.x.T @ (self.y - self.predict(self.x))) / self.x.shape[0]

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return gradient
    
    def fit(self,lr=None, n_iterations=None):
        k = 0
        if n_iterations is None:
            n_iterations = self.num_iters
        if lr != None and lr != "diminishing":
            self.lr = lr
            
        ##############################################################################
        # TODO: Implement gradient descent algorithm.                                #
        #                                                                            #
        # You may not use any built in function which directly calculate             #
        # gradient.                                                                  #
        # Steps of gradient descent algorithm:                                       #
        #   1. Calculate gradient of cost function. (Call gradient() function)       #
        #   2. Update w with gradient.                                               #
        #   3. Log weight and cost for plotting in weight_history and cost_history.  #
        #   4. Repeat 1-3 until the cost change converges to epsilon or achieves     #
        # n_iterations.                                                              #
        # !!WARNING-1: Use Mean Square Error between predicted and  actual y values  #
        # for cost calculation.                                                      #
        #                                                                            #
        # !!WARNING-2: Be careful about lr can takes "diminishing". It will be used  #
        # for further test in the jupyter-notebook. If it takes lr decrease with     #
        # iteration as (lr =(1/k+1), here k is iteration).                           #
        ##############################################################################

        flag = lr == "diminishing"
        for k in range(n_iterations):
            if flag:
                self.w -= (1 / (k + 2)) * self.gradient() # k starts from zero
            else:
                self.w -= self.lr * self.gradient()
            self.weight_history.append(self.w)
            self.cost_history.append(self.cost())
            # Check if the error is less than self.epsilon and then terminate if so
            if abs(self.cost_history[-1] - self.cost_history[-2])< self.epsilon:
                break

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return self.w, k
    
    def predict(self, x):
        
        y_pred = np.zeros_like(self.y)

        y_pred = x.dot(self.w)

        return y_pred
        
    def cross_validate(self, k=5):
        """
        Performs k-fold cross-validation and returns the cross-validated error.

        Args:
        - k (int): Number of folds for cross-validation.

        Returns:
        - float: Cross-validated error.
        """
        np.random.seed(42)
        m = self.x.shape[0]
        fold_size = m // k
        indices = np.arange(m)
        np.random.shuffle(indices)

        cross_val_error = []
        ##############################################################################
        # TODO: The cross_validate function is designed to perform k-fold            #
        # cross-validation, a technique used to evaluate the performance of a        #
        #machine learning model. It takes the number of folds, k, as input and       #
        #splits the dataset into k equal-sized subsets. For each fold, one subset is #
        #used as the validation set, and the remaining k-1 subsets are used as the   #
        #training set. The function iterates through each fold, training a linear    #
        #regression model on the training data and evaluating its performance on     #
        #the validation data. The error of each fold is recorded, and the function   #
        #returns the average cross-validated error, providing a reliable             #
        #estimate of the model's generalization performance.                         #
        ##############################################################################

        for i in range(k):
            # Create the validation set
            start = i * fold_size
            end = (i + 1) * fold_size if i != k - 1 else None
            x_val = self.x[start:end]
            y_val = self.y[start:end]

            # Create the training set
            x_train = np.concatenate((self.x[:start], self.x[end:]))
            y_train = np.concatenate((self.y[:start], self.y[end:]))
            
            # Create the model
            model = gradientDescent(x_train, y_train, self.w, self.lr, self.num_iters, add_bias=False)

            # Train the model
            model.fit()

            # Make predictions on the validation set
            y_pred = model.predict(x_val)

            # Calculate the error
            error = np.sqrt(np.sum(np.square(y_pred - y_val)))

            # Record the error
            cross_val_error.append(error)

        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        avg_cross_val_error = np.mean(cross_val_error)
        
        return avg_cross_val_error
        
class SGD():
    def __init__(self, x, y, w, lr, num_iters, add_bias=True):
        self.x = np.c_[np.ones((x.shape[0], 1)), x] if add_bias else x.copy()
        self.y = y
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = 1e-4
        self.w = w.copy()
        self.weight_history = [self.w]
        self.cost_history = [np.sum(np.square(self.predict(self.x)-self.y))/self.x.shape[0]]
        
    def gradient(self, x, y):
        gradient = np.zeros_like(self.w)
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        #  Replace "pass" statement with your code
        x = x.reshape(-1, 1)

        gradient = -1 * (x @ (y - self.predict(x.T))) / x.T.shape[0]

        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return gradient
    
    def cost(self):
        """
            Utitlity method for calculating the MSE for the current set of weights
        """
        return np.sum(np.square(self.predict(self.x) - self.y)) / self.x.shape[0]
    
    def fit(self,lr=None, n_iterations=None):
        k = 0
        if n_iterations is None:
            n_iterations = self.num_iters
        if lr != None and lr != "diminishing":
            self.lr = lr
            
        ##############################################################################
        # TODO: Implement gradient descent algorithm.                                #
        #                                                                            #
        # You may not use any built in function which directly calculate             #
        # gradient.                                                                  #
        # Steps of stochastic gradient descent algorithm:                            #
        #   1. Pick a sample from the training set.                                  #
        #   2. Calculate gradient of cost function. (Call gradient() function)       #
        #   3. Update w with gradient.                                               #
        #   4. Repeat 1-3 for all samples in the training set.                       #
        #   5. Log weight and cost for plotting in weight_history and cost_history.  #
        #   6. Repeat 1-5 until the cost change converges to epsilon or achieves     #
        # n_iterations.                                                              #
        # !!WARNING-1: Use Mean Square Error between predicted and  actual y values  #
        # for cost calculation.                                                      #
        #                                                                            #
        # !!WARNING-2: Be careful about lr can takes "diminishing". It will be used  #
        # for further test in the jupyter-notebook. If it takes lr decrease with     #
        # iteration as (lr =(1/k+1), here k is iteration).                           #
        ##############################################################################

        flag = lr == "diminishing"

        for k in range(n_iterations):
            # pick a sample from the training set
            random_index = np.random.randint(self.x.shape[0])
            sample_x = np.array(self.x[random_index])
            sample_y = np.array(self.y[random_index])

            if flag:
                self.w -= (1 / (k + 2)) * self.gradient(sample_x, sample_y) # k starts from zero
            else:
                self.w -= self.lr * self.gradient(sample_x, sample_y)

            self.weight_history.append(self.w)
            self.cost_history.append(self.cost())
            # Check if the error is less than self.epsilon and then terminate if so
            if abs(self.cost_history[-1] - self.cost_history[-2]) < self.epsilon:
                break

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return self.w, k
    
    def predict(self, x):
        
        y_pred = np.zeros_like(self.y)

        y_pred = x.dot(self.w)

        return y_pred
    
        
    def cross_validate(self, k=5):
        """
        Performs k-fold cross-validation and returns the cross-validated error.

        Args:
        - k (int): Number of folds for cross-validation.

        Returns:
        - float: Cross-validated error.
        """
        np.random.seed(42)
        m = self.x.shape[0]
        fold_size = m // k
        indices = np.arange(m)
        np.random.shuffle(indices)

        cross_val_error = []

        ##############################################################################
        # TODO: The cross_validate function is designed to perform k-fold            #
        # cross-validation, a technique used to evaluate the performance of a        #
        #machine learning model. It takes the number of folds, k, as input and       #
        #splits the dataset into k equal-sized subsets. For each fold, one subset is #
        #used as the validation set, and the remaining k-1 subsets are used as the   #
        #training set. The function iterates through each fold, training a linear    #
        #regression model on the training data and evaluating its performance on     #
        #the validation data. The error of each fold is recorded, and the function   #
        #returns the average cross-validated error, providing a reliable             #
        #estimate of the model's generalization performance.                         #
        ##############################################################################
        # Replace "pass" statement with your code                                    #
        
        for i in range(k):
            # Create the validation set
            start = i * fold_size
            end = (i + 1) * fold_size if i != k - 1 else None
            x_val = self.x[start:end]
            y_val = self.y[start:end]

            # Create the training set
            x_train = np.concatenate((self.x[:start], self.x[end:]))
            y_train = np.concatenate((self.y[:start], self.y[end:]))
            
            # Create the model
            model = SGD(x_train, y_train, self.w, self.lr, self.num_iters, add_bias=False)

            # Train the model
            model.fit()

            # Make predictions on the validation set
            y_pred = model.predict(x_val)

            # Calculate the error
            error = np.sqrt(np.sum(np.square(y_pred - y_val)))

            # Record the error
            cross_val_error.append(error)
        
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ############################################################################## 

        avg_cross_val_error = np.mean(cross_val_error)
        
        return avg_cross_val_error
