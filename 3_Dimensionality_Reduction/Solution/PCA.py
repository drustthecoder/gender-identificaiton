import numpy as np
import sklearn

if __name__ == "__main__":
    D = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    p_result = np.load("IRIS_PCA_matrix_m4.npy")
    print("Yeri Saakh Laayaan")