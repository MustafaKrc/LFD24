import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def leastSquares(X, Y, degree):

    """
    Input:
    X and Y are two-dim numpy arrays.
    X dims: (N of samples, feature dims)
    Y dim: (N of samples, response dims)
    degree: Degree of the polynomial
    Output:
    Weight (Coefficient) vector
    Vandermonde Matrix
    """
    # Add a column of ones for the intercept
    if degree < 1:
        raise ValueError("Polynomial degree must be greater than or equal to 1.")

    # Initialize the Vandermonde matrix with the original features
    X_poly = X.copy()
        
    ##############################################################################
    # TODO: Implement least square theorem for polynomial regression.            #
    #                                                                            #
    # You may not use any built in function which directly calculate             #
    # Least squares except matrix operation in numpy.  			    #
    # You need to define and compute Vandermonde matrix first.                   #
    ##############################################################################
    # Replace "pass" statement with your code

    # Form the Vandermonde matrix
    V = np.ones((X.shape[0], 1))
    for i in range(1, degree + 1):
        V = np.concatenate((V, X**i), axis=1)
    # Compute the weights using the normal equation
    W = np.linalg.inv(V.T @ V) @ V.T @ Y
    X_poly = V
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    
    return W, X_poly

class gradientDescent():

    def __init__(self, x, y, w, lr, num_iters):
        self.x = x
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
        N = self.x.shape[0]
        return 1/N * np.sum(np.square(self.predict(self.x) - self.y))
    
    def gradient(self):
        gradient = np.zeros_like(self.w)
        
        ##############################################################################
        # TODO: Calculate gradient of gradient descent algorithm.                    #
        #                                                                            #
        ##############################################################################
        
        # calculate gradient of cost function
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
        # !!WARNING-1: Use Mean Square Error between predicted and  actual y values #
        # for cost calculation.                                                      #
        #                                                                            #
        # !!WARNING-2: Be careful about lr can takes "diminishing". It will be used  #
        # for further test in the jupyter-notebook. If it takes lr decrease with     #
        # iteration as (lr =(1/k+1), here k is iteration).                           #
        ##############################################################################

        # Replace "pass" statement with your code
        
        flag = lr == "diminishing"
        for k in range(n_iterations):
            if flag:
                self.lr = 1/(k+2)

            self.w -= self.lr * self.gradient()
            self.weight_history.append(self.w)
            self.cost_history.append(self.cost())
            if k > 0 and np.abs(self.cost_history[-1] - self.cost_history[-2]) < self.epsilon:
                break
	
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        return self.w, k
    
    def predict(self, x):
        
        y_pred = np.zeros_like(self.y)

        y_pred = x.dot(self.w)

        return y_pred

