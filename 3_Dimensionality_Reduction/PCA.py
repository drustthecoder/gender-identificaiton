import sklearn.datasets
import matplotlib.pyplot
from mlpr import *

# PCA
print("PCA:")
pca_solution = np.load("Solution/IRIS_PCA_matrix_m4.npy")  # Load PCA solution
data, labels = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
principal_component = PCA(data, 4)  # Perform PCA with 4 components
print(pca_solution)
print(principal_component)

colors = ['tab:orange', 'tab:green', 'tab:blue']

for current_label in labels:
    current_data = data[:, labels == current_label]
    current_projection = np.dot(principal_component.T, current_data)  # Compute the projection of the data
    plt.scatter(
        current_projection[0],
        current_projection[1],
        color=colors[current_label]
    )  # Plot the projected data

plt.show()


