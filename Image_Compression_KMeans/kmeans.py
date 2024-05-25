import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    KMeans class
    """ 
    def __init__(self,) -> None:
        self.K = None
        self.centroids = None


    def init_centroids(self, X, K):
        """
        Initializes K centroids that are to be used in K-Means on the dataset X
        
        Args:
            X (ndarray): (m, n) Data points
            K (int): number of centroids
        
        Returns:
            centroids (ndarray): (K, n) Randomly initialized centroids
        """
        m, n = X.shape
        centroids = np.zeros((K, n))
        idx = np.random.randint(0, m, K)
        
        for i in range(K):
            centroids[i, :] = X[idx[i], :]
    
        return centroids

    def find_closest_centroids(self, X, centroids):
        """
        Computes the centroid memberships for every example
        
        Args:
            X (ndarray): (m, n) Input values      
            centroids (ndarray): (K, n) centroids
        
        Returns:
            idx (array_like): (m,) closest centroids
        
        """
        m = int(X.shape[0])              ## No of training examples
        K = int(centroids.shape[0])      ## No of centroids
        idx = np.zeros(m, dtype=int)
        
        for i in range(m):
            min_dist = np.inf
            for k in range(K):
                dist = np.sum((X[i,:] - centroids[k,:]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    idx[i] = k
                    
        return idx


    def compute_centroids(self,X, idx, K):
        """
        Returns the new centroids by computing the means of the 
        data points assigned to each centroid.
        
        Args:
            X (ndarray):   (m, n) Data points
            idx (ndarray): (m,) Array containing index of closest centroid for each 
                        example in X. Concretely, idx[i] contains the index of 
                        the centroid closest to example i
            K (int):       number of centroids
        
        Returns:
            centroids (ndarray): (K, n) New centroids computed
        """
        
        m, n = X.shape
        centroids = np.zeros((K, n))

        for k in range(K):
            centroids[k, :] = np.mean(X[idx == k, :], axis=0)
        
        return centroids
    
    def run(self, X, K, max_iters=100, initial_centroids=None):
        """
        Runs the K-Means algorithm on data matrix X, where each row of X
        is a single example
        """
        self.K = K
        if not initial_centroids:
            initial_centroids = self.init_centroids(X, K)
        # Initialize values
        m, n = X.shape
        K = initial_centroids.shape[0]
        centroids = initial_centroids
        previous_centroids = centroids    
        idx = np.zeros(m, dtype=int)
        plt.figure(figsize=(8, 6))

        # Run K-Means
        for i in range(max_iters):
            
            #Output progress
            print("K-Means iteration %d/%d" % (i, max_iters-1))
            
            # For each example in X, assign it to the closest centroid
            idx = self.find_closest_centroids(X, centroids)
            
            # Optionally plot progress
            
                  
            # Given the memberships, compute new centroids
            centroids = self.compute_centroids(X, idx, K)

        return centroids, idx
