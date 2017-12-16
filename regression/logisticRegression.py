import numpy as np
import matplotlib.pyplot as plt
from Data import Data


# ===== Helper Function for Plotting - DO NOT EDIT =============
def boundary(w):
    boundary_x1 = np.array([0,1])
    boundary_x2 = -(w[0]*boundary_x1+w[2])/w[1]
    midpt = np.array([0.5,-(w[0]*0.5+w[2])/w[1]])
    vec = (w[:2])/np.linalg.norm(w[:2])
    endpt = midpt + 0.1*vec
    xvec = np.array([midpt[0],endpt[0]])
    yvec = np.array([midpt[1],endpt[1]])
    return (boundary_x1,boundary_x2), (xvec,yvec)

def plot_boundary(w, quiet=True):
    bndry, direction = boundary(w)
    b1,b2 = bndry
    if not quiet:
        color = 'black'
        plt.plot(b1,b2,'-',color=color,label='decision\nboundary')
    else:
        color = 'lightgray'
        plt.plot(b1,b2,'-',color=color)
    xvec,yvec = direction
    plt.plot(xvec,yvec,'o-',color=color)

# =============================================================

class LogisticRegression:
    def __init__(self,  alpha=1e-2):
        self.w = np.array([0,-1,.5]) # You can use these as the initial values for the weights

    def fit(self, X, Y, epochs=10000):
        """
        Make use of self.w to ensure changes in the weight are reflected globally
        :param X: input features, shape:(N,3); Note: Ones for the bias have already been appended for your convenience
        :param Y: target, shape:(N,)
        :return: None

        IMP: Make use of self.w to ensure changes in the weight are reflected globally.
            We will be making use of get_params and set_params to check for correctness
        """
        step = 0.0001 # set the value for step parameter
        for t in range(epochs):

            self.w = self.w - step * self.loss_grad(self.w, X, Y)
            if t % 100 == 0:
                print("Epoch: {} :: loss: {}".format(t, self.loss(self.w, X, y)))
                plot_boundary(self.w)

    def predict(self, X):
        """
        Return your predictions
        :param X: inputs, shape:(N,3)
        :return: predictions, shape:(N,)
        """

        return np.dot(X, self.w)

    def loss(self, w, X, Y):
        """
        :param W: weights, shape:(3,)
        :param X: input, shape:(N,3)
        :param Y: target, shape:(N,)
        :return: scalar loss value
        """

        sum = 0;
        for i in range(X.shape[0]):
            z = np.dot(w.T, X[i])
            sum = sum + self.loss_entropy(Y[i], self.sigmoid(z))
        return sum

    def loss_grad(self, w, X, y):
        """
        Compute the gradient of the loss.
        (Function will be tested only for gradient descent)
        :param W: weights, shape:(3,)
        :param X: input, shape:(N,3)
        :param Y: target, shape:(N,)
        :return: vector of size (3,) containing gradients for each weight
        """
        Z = np.dot(X, w)
        return np.dot((self.sigmoid(Z) - y).T, X)

    def get_params(self):
        """
        :return: the current parameters value
        """
        return self.w

    def set_params(self, w):
        """
        :param w:
        :return: None
        """
        self.w = w
        return 0

    def loss_entropy(self, y, sig):
        # return (sig - y) * x
        return -y*np.log(sig) - (1-y)*np.log(1-sig)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    # Get data
    data = Data()
    X, y, X_train, y_train, X_test, y_test = data.get_logistic_regression_data()
    N = np.shape(X)[0]

    # Logistic regression with gradient descent
    model = LogisticRegression()
    model.fit(X, y)
    y = model.predict(X)
    w_grad = model.get_params()

    # Plot the results
    plt.plot(X[:N // 2, 0], X[:N // 2, 1], 'r+', label='pos')
    plt.plot(X[N // 2:, 0], X[N // 2:, 1], 'b_', label='neg')
    plot_boundary(w_grad, quiet=False)
    plt.legend()
    plt.axis('square')
    plt.axis([0, 1, 0, 1])
    plt.savefig('figures/Q2.png')
    plt.close()
