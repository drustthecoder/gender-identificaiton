from mlpr import *

print("LDA:")
data, labels = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
lda_solution = np.load("Solution/IRIS_LDA_matrix_m2.npy") # Load LDA solution
# we can compute at most 2 discriminant directions, since we have 3 classes
m = 2
lda_eigenvectors = LDA(data, labels, m) # Perform LDA with 1 dimension
# print(lda_eigenvectors)
# print(lda_solution)

projected_data = np.dot(lda_eigenvectors.T, data) # Compute the projection of the data
# print(projected_data.shape)
colors = ['tab:orange', 'tab:green', 'tab:blue']
for current_label in labels:
    current_data = data[:, labels == current_label]
    current_projection = np.dot(lda_eigenvectors.T, current_data)  # Compute the projection of the data
    plt.scatter(
        current_projection[0],
        current_projection[1],
        color=colors[current_label]
    )  # Plot the projected data
plt.title("LDA")
plt.show()