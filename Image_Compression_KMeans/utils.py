import numpy as np
import matplotlib.pyplot as plt 

def show_centroid_colors(color_centroids):
    """
    Display the colors of the centroids
    
    Args:
        color_centroids (ndarray): (K, 3) Centroids of the KMeans algorithm
    """
    n_colors = color_centroids.shape[0]  # Number of colors
    
    # Close any previously open figures
    plt.close('all')
    
    
    fig, axes = plt.subplots(1, n_colors, figsize=(n_colors, 2), subplot_kw={'xticks': [], 'yticks': []})
    if n_colors == 1:
        axes = [axes]
    
    # Plot each color
    for ax, color in zip(axes, color_centroids):
        ax.set_facecolor(color / 255.0)  # Normalize the RGB values to [0, 1]
    
    # Set the spacing between subplots to zero
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()