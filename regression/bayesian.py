from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
from Data import Data



class Posterior:
    def __init__(self, limes, cherries, a=2, b=2):
        self.a = a
        self.b = b
        self.limes = limes          # shape: (N,)
        self.cherries = cherries    # scalar int
        self.N = np.shape(self.limes)[0]

    def get_MAP(self):
        """
        compute MAP estimate
        :return: MAP estimates for diff. values of lime; shape:(N,)
        """
        return 1 - (self.cherries + self.a -1)/(self.cherries + self.limes + self.a + self.b -2)

    def get_finite(self):
        """
        compute posterior with finite hypotheses
        :return: estimates for diff. values of lime; shape:(N,)
        """

        p_h = [.1, .2, .4, .2, .1]
        h = [[1,0], [.75,.25], [.5, .5], [.25, .75], [0,1]]
        p = np.zeros(self.N)
        for n in range(self.N):
            sum = 0
            alpha = 0
            for i in range(len(h)):
                alpha = alpha + h[i][1] ** (n) * p_h[i]
                sum = sum + h[i][1] ** (n +1) * p_h[i]
            p[n] = sum/alpha
        return p

    def get_infinite(self):
        """
        compute posterior with beta prior
        :return: estimates for diff. values of lime; shape:(N,)
        """

        p = np.zeros(self.N)
        for n in range(self.N):
            Z = gamma(self.a) * gamma(self.b + n) / gamma(self.a + self.b + n)
            alpha = 1 / Z
            p[n] = alpha * (gamma(self.a) * gamma(self.b + n +1))/gamma(self.a + self.b + n + 1)
        return p

if __name__ == '__main__':
    # Get data
    data = Data()
    limes, cherries = data.get_bayesian_data()

    # Create class instance
    posterior = Posterior(limes=limes, cherries=cherries)

    # PLot the results
    plt.plot(limes, posterior.get_MAP(), label='MAP')
    plt.plot(limes, posterior.get_finite(), label='5 Hypotheses')
    plt.plot(limes, posterior.get_infinite(), label='Bayesian with Beta Prior')
    plt.legend()
    plt.savefig('figures/Q4.png')
