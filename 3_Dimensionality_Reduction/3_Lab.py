import sklearn.datasets
import matplotlib.pyplot
from mlpr import *

# PCA
print("PCA:")
pca_p_sol = numpy.load("Solution/IRIS_PCA_matrix_m4.npy")
D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
P = PCA(D, 4)
print(pca_p_sol/P)

y = numpy.dot(P.T, D)  # compute the projection of the data
matplotlib.pyplot.scatter(y[0], y[1])
matplotlib.pyplot.show()

# LDA
print("LDA:")
lda_p_sol = numpy.load("Solution/IRIS_LDA_matrix_m2.npy")
m = 1
U = LDA1(D, L, m)
sol = U[:, ::-1][:, 0:m]
print(sol)
print(lda_p_sol)