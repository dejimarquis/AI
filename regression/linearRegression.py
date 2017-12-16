import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
from Data import Data


class LinearRegression:
    def __init__(self, flag, alpha=1e-2):
        self.flag = flag        # flag can only be either: 'GradientDescent' or 'Analytic'
        self.alpha = alpha      # regularization coefficient
        self.w = np.zeros(2)    # initializing the weights

    def fit(self, X, Y, epochs=10000):
        """
        :param X: input features , shape: (N,2)
        :param Y: target, shape: (N,)
        :return: None

        IMP: Make use of self.w to ensure changes in the weight are reflected globally.
            We will be making use of get_params and set_params to check for correctness
        """
        if self.flag == 'GradientDescent':
            step = 0.001
            for t in range(epochs):
                self.w = self.w - step * self.loss_grad(self.w, X, Y)
                if t%100 == 0:
                    print("Epoch: {} :: loss: {}".format(t, self.loss(self.w, X, y)))

        elif self.flag == 'Analytic':
            A = X
            A_alpha = np.dot(A.T, A) + 2 * self.alpha * np.eye(A.shape[1])
            A_alpha_plus = np.linalg.inv(A_alpha).dot(A.T)
            self.w = np.dot(A_alpha_plus, Y)
            pass
            # Remember: matrix inverse in NOT equal to matrix**(-1)
            # WRITE the required CODE HERE to update the weights using closed form solution

        else:
            raise ValueError('flag can only be either: ''GradientDescent'' or ''Analytic''')

    def predict(self, X):
        """
        :param X: inputs, shape: (N,2)
        :return: predictions, shape: (N,)

        Return your predictions
        """
        return np.dot(X, self.w)

    def loss(self, w, X, Y):
        """
        :param W: weights, shape: (2,)
        :param X: input, shape: (N,2)
        :param Y: target, shape: (N,)
        :return: scalar loss value

        Compute the loss loss. (Function will be tested only for gradient descent)
        """
        w_norm = np.dot(w, w.T)
        sum = 0
        for i in range(X.shape[0]):
            sum = sum + (np.dot(w.T, X[i]) - Y[i]) ** 2
        l = 0.5 * sum + self.alpha*w_norm
        return l

    def loss_grad(self, w, X, y):
        """
        :param W: weights, shape: (2,)
        :param X: input, shape: (N,2)
        :param Y: target, shape: (N,)
        :return: vector of shape: (2,) containing gradients for each weight

        Compute the gradient of the loss. (Function will be tested only for gradient descent)
        """
        A = X
        A_alpha = np.dot(A.T, A) + 2 * self.alpha * np.eye(A.shape[1])
        loss_gradient = np.dot(A_alpha, w) - (np.dot(A.T, y))
        return loss_gradient

    def get_params(self):
        """
        ********* DO NOT EDIT ************
        :return: the current parameters value
        """
        return self.w

    def set_params(self, w):
        """
        ********* DO NOT EDIT ************
        This function Will be used to set the values of weights externally
        :param w:
        :return: None
        """
        self.w = w
        return 0



if __name__ == '__main__':
    # Get data
    data = Data()
    X, y = data.get_linear_regression_data()

    # Linear regression with gradient descent
    model = LinearRegression(flag='GradientDescent')
    # print(model.loss(np.zeros(2),X,y))
    model.fit(X,y)
    y_grad = model.predict(X)
    w_grad = model.get_params()

    # Analytic Linear regression
    model = LinearRegression(flag='Analytic')
    model.fit(X,y)
    y_exact = model.predict(X)
    w_exact = model.get_params()

    # Compare the resultant parameters
    diff = norm(w_exact - w_grad)
    print(diff, w_exact, w_grad)

    # Plot the results
    plt.plot(X[:, 0], y, 'o-', label='true')
    plt.plot(X[:, 0], y_exact, '-', label='pinv')
    plt.plot(X[:, 0], y_grad, '-.', label='grad')
    plt.legend()
    plt.savefig('figures/Q1.png')
    plt.close()
