import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances

# Configuration
input_image_path = "Einstein.jpg"  
sigma_d = 8.0                      
output_dir = "" 

# Cluster specifications
clusters = [
    {'center': (25, 75), 'size': 50, 'color': 'red'},
    {'center': (75, 75), 'size': 50, 'color': 'blue'},
    {'center': (50, 50), 'size': 100, 'color': 'green'},
    {'center': (75, 25), 'size': 100, 'color': 'purple'},
    {'center': (25, 25), 'size': 200, 'color': 'orange'}
]

# Generate points with Gaussian distribution
np.random.seed(20)
points, labels = [], []
for cluster in clusters:
    pts = np.random.normal(cluster['center'], sigma_d, (cluster['size'], 2))
    pts = np.clip(pts, 0, 100)
    points.append(pts)
    labels.extend([cluster['color']] * cluster['size'])
    
points = np.vstack(points) * 5  # Scale to 500x500 coordinates

# Calculate distance matrix
W = euclidean_distances(points)

# Calculate weights for different sigma values
sigmas = [0.1*sigma_d, sigma_d, 10*sigma_d]
weight_matrices = [np.exp(-W**2/(2*(s**2))) for s in sigmas]


#Plot points on input image
def plot_points_on_image():
    plt.figure(figsize=(5, 5), dpi=100)
    img = plt.imread(input_image_path)
    plt.imshow(img)
    plt.scatter(points[:,0], points[:,1], c=labels, s=15, alpha=0.7, edgecolor='w', linewidth=0.3)
    plt.axis('off')
    plt.savefig(f"{output_dir}points_on_image.png", bbox_inches='tight')
    plt.close()

#Plot cluster distribution
def plot_cluster_distribution():
    plt.figure(figsize=(6, 5))
    for cluster in clusters:
        plt.scatter([], [], c=cluster['color'], label=f'Center {cluster["center"]}')
    plt.scatter(points[:,0], points[:,1], c=labels, s=15, alpha=0.7)
    plt.title('Cluster Spatial Distribution')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(f"{output_dir}cluster_distribution.png", bbox_inches='tight')
    plt.close()

#Plot distance histogram
def plot_distance_histogram():
    intra_dist, inter_dist = [], []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if labels[i] == labels[j]:
                intra_dist.append(W[i,j])
            else:
                inter_dist.append(W[i,j])

    plt.figure(figsize=(8, 4))
    plt.hist(intra_dist, bins=50, alpha=0.7, label='Intra-cluster')
    plt.hist(inter_dist, bins=50, alpha=0.7, label='Inter-cluster')
    plt.title('Euclidean Distance Distribution')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{output_dir}distance_histogram.png")
    plt.close()

#Plot weight distributions
def plot_weight_distributions():
    plt.figure(figsize=(8, 4))
    for w, s in zip(weight_matrices, sigmas):
        sns.kdeplot(w.flatten(), label=f'σ={s:.1f}')
    plt.title('Weight Value Distributions')
    plt.xlabel('Weight')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{output_dir}weight_distributions.png")
    plt.close()

#Plot RBF decay curves
def plot_rbf_curves():
    x = np.linspace(0, W.max(), 100)
    plt.figure(figsize=(8, 4))
    for s in sigmas:
        y = np.exp(-x**2/(2*s**2))
        plt.plot(x, y, label=f'σ={s:.1f}')
    plt.title('RBF Weight Decay Patterns')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.savefig(f"{output_dir}rbf_curves.png")
    plt.close()

#Plot heatmap
def plot_heatmap():
    plt.figure(figsize=(5, 5))
    sns.heatmap(weight_matrices[1], cmap='viridis', 
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Weight Value'})
    plt.title(f'Weight Matrix (σ={sigma_d})')
    plt.savefig(f"{output_dir}weight_matrix_heatmap.png")
    plt.close()

#Plot eigenvalues and eigenvectors
def plot_eigen_components(sigmas, weight_matrices, points, output_dir):
    for s, weight_matrix in zip(sigmas, weight_matrices):
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(weight_matrix)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Set dynamic threshold 
        threshold = 0.01 * eigenvalues[0]
        valid_indices = np.where(eigenvalues > threshold)[0]
        
        # Select top 5 eigenvalues or all above threshold
        top_k = min(5, len(valid_indices)) if len(valid_indices) > 0 else 1
        
        
        plt.figure(figsize=(10, 2.5*top_k))
        plt.suptitle(f'Eigen Components Analysis (σ={s:.1f})', y=1.02)
        plt.subplot(top_k+1, 1, 1)
        plt.plot(eigenvalues[:20], 'o-', markersize=4)
        plt.axhline(threshold, color='r', linestyle='--', alpha=0.5)
        plt.title('Top 20 Eigenvalues')
        plt.ylabel('Magnitude')
        plt.yscale('log')
        
        # Plot eigenvectors
        for i in range(top_k):
            plt.subplot(top_k+1, 1, i+2)
            v = eigenvectors[:, valid_indices[i]]
            
            # Normalize for better visualization
            v_norm = (v - v.min()) / (v.max() - v.min())
            
            # Create scatter plot
            sc = plt.scatter(points[:, 0], points[:, 1], c=v_norm,
                            cmap='viridis', s=15, alpha=0.8)
            plt.colorbar(sc, label=f'Eigenvector {i+1} Value')
            plt.title(f'Eigenvector {i+1} (λ={eigenvalues[valid_indices[i]]:.2e})')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}eigen_components_sigma_{s:.1f}.png", bbox_inches='tight')
        plt.close()


plot_points_on_image()
plot_cluster_distribution()
plot_distance_histogram()
plot_weight_distributions()
plot_rbf_curves()
plot_heatmap()
plot_eigen_components(sigmas, weight_matrices, points, output_dir)

print("All plots generated successfully")