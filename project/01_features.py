# from dataset import *
from mlpr import *

data, labels = load_data("data/Test.txt")
males = data[:, labels == 0]
females = data[:, labels == 1]

# histogram(data, labels)
# # scatter(data, labels)
#
# heatmap(data, "data", "Greens")
# heatmap(males, "male", "Reds")
# heatmap(females, "female", "Blues")

# lda_eigenvectors = LDA(data, labels, 1)
# print("Calculating projection...")
# males_projection = np.dot(lda_eigenvectors.T, males)
# females_projection = np.dot(lda_eigenvectors.T, females)
# print("Plotting...")
# plt.hist(males_projection.flatten(), color="orange", density=True, bins=100, edgecolor='k', alpha=0.4, label="Male")
# plt.hist(females_projection.flatten(), color="purple", density=True, bins=100, edgecolor='k', alpha=0.4, label="Female")
# plt.legend()
# plt.title("Distribution of Projected Data on LDA Eigenvectors")
# plt.savefig('plots/lda_hist.png')
# plt.show()
# print("Completed!")

min_dim = 0
max_dim = 12

plot_explained_variance_range(data, min_dim, max_dim)

