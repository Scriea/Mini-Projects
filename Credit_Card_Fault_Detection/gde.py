import os 
import numpy as np
import pandas as pd

class GDE:
    """
    Gaussian Density Estimation(GDE) is a simple anomaly detection algorithm
    that assumes the features are normally distributed. It calculates the
    mean and variance of each feature in the dataset and uses these parameters
    to estimate the probability density of each example. If the probability
    density of an example is below a certain threshold, it is classified as
    an anomaly.
    """
    def __init__(self,) -> None:
        self.mu = None
        self.var = None
        self.epsilon = None


    def set_gaussian(self, X:np.ndarray)->tuple:
        """
        Calculates mean and variance of all features 
        in the dataset
        
        Args:
            X (ndarray): (m, n) Data matrix
        
        Returns:
            mu (ndarray): (n,) Mean of all features
            var (ndarray): (n,) Variance of all features
        """
        m, n = X.shape
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        self.mu, self.var = mu, var

        return mu, var
    
    def multivariate_gaussian(self, X:np.ndarray)->np.ndarray:
        """
        Computes the probability 
        density function of the examples X under the multivariate gaussian 
        distribution with parameters mu and var. If var is a matrix, it is
        treated as the covariance matrix. If var is a vector, it is treated
        as the var values of the variances in each dimension (a diagonal
        covariance matrix
        """
        # Set mean and variance if not set
        if type(self.mu)!=np.ndarray or type(self.var)!=np.ndarray:         
            self.set_gaussian(X)

        k = len(self.mu)
        
        if self.var.ndim == 1:
            self.var = np.diag(self.var)
            
        X = X - self.mu
        p = (2* np.pi)**(-k/2) * np.linalg.det(self.var)**(-0.5) * \
            np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(self.var)) * X, axis=1))
        
        return p
    
    def fit(self, X:np.ndarray, Y:np.ndarray)->None:
        """
        Fits the model to the data by setting the mean and variance
        of the features in the dataset and finding optimal threshold
        """
        self.set_gaussian(X)
        # print(self.mu.shape, self.var.shape)
        best_epsilon = 0
        best_F1 = 0
        F1 = 0

        p = self.multivariate_gaussian(X)
        
        step_size = (max(p) - min(p)) / 1000
        
        for epsilon in np.arange(min(p), max(p), step_size):
        
            tp = np.sum((p<epsilon) & (Y==1))
            fp = np.sum((p<epsilon) & (Y==0))
            fn = np.sum((p>=epsilon) & (Y==1))
            
            precision = tp/(tp+fp+1e-5)
            recall = tp/(tp+fn+1e-5)
            F1 = 2*precision*recall/(precision + recall+1e-5)
                
            if F1 > best_F1:
                best_F1 = F1
                best_epsilon = epsilon
        
        self.epsilon = best_epsilon

    def predict(self, X):
        """
        Predicts whether the examples in X are outliers
        using the threshold epsilon
        """
        p = self.multivariate_gaussian(X)
        return p < self.epsilon