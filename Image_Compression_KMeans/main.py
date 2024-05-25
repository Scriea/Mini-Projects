import os
import argparse
import numpy as np
import matplotlib.pyplot as plt 
from kmeans import *
from utils import *

parser = argparse.ArgumentParser(description='Image Compression using KMeans')
parser.add_argument('--image', type=str, help='Path to the image file')
args = parser.parse_args()

if __name__=="__main__":
    if not os.path.exists(args.image):
        print(f"File {args.image} not found")
        exit(1)

    img = plt.imread(args.image) 
    plt.imshow(img)
    plt.title('Original Image')
    plt.pause(1)
    plt.close()

    X = np.reshape(img, (img.shape[0]*img.shape[1], 3))
    kmeans = KMeans()
    centroids, idx = kmeans.run(X, K=16, max_iters=10)
    show_centroid_colors(centroids)


    # Replace each pixel with the color of the closest centroid
    X_recovered = centroids[idx, :] 
    
    # Reshape image into proper dimensions
    X_recovered = np.reshape(X_recovered, img.shape).astype('uint8') 

    plt.imshow(X_recovered)
    plt.title('Compressed Image')
    plt.pause(1)
    plt.close()


    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(X_recovered)
    axes[1].set_title('Compressed Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()