from matplotlib.pyplot import *
import numpy as np
import scipy.io as io
from mpl_toolkits.mplot3d import Axes3D

class ActiveSubspace(object):
    def __init__(self, alpha, J, dJdalpha, sigma_prior = 0.5, make_plot=True):
        self.alpha = alpha
        self.J = J
        self.dJdalpha = dJdalpha
        self.sigma_prior = sigma_prior
        self.calc()
        if make_plot:
            self.plot()
        
    def calc(self):
        M, m = self.dJdalpha.shape

        # normalize and center alpha
        self.nalpha = (self.alpha - 1.0)/self.sigma_prior
        for i in range(M):
            _norm = np.sqrt(np.sum(self.dJdalpha[i,:]**2))
            self.dJdalpha[i,:] = self.dJdalpha[i,:]/_norm
            
        U, self.sig, V = np.linalg.svd(self.dJdalpha, full_matrices=1, compute_uv=1)
        self.sig = self.sig**2/M
        self.W = V.T

    def plot(self):
        figure()
        semilogy(self.sig[:20], 'ro-')
        ylabel("Eigenvalue")
        figure()
        for i in range(5):
            plot(self.W[:, i])
        legend(["Mode = %i"%i for i in range(5)], loc="best")
        ylabel("Eigenvector")
        figure(3)
        semilogy(np.dot(self.nalpha,self.W[:,0]), self.J, 'r.')
        xlabel("Active variable 1")
        ylabel("Objective function, J")
        
        figure()
        ax = gcf().add_subplot(111, projection='3d')
        ax.scatter(np.dot(self.nalpha,self.W[:,0]), np.dot(self.nalpha,self.W[:,1]), np.dot(self.nalpha,self.W[:,2]), c="r", marker="o")
        show()

if __name__ == "__main__":
    data = io.loadmat("gradz_550.mat")
    alpha, J, dJdalpha = data["alphaz"], data["Jz"], data["dalphaz"]
    active_sub = ActiveSubspace(alpha, J, dJdalpha)
    
