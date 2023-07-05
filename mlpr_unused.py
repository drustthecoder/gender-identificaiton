# def classify_iris():
#     D, L = load_iris()
#     (DTR, LTR), (DTV, LTV) = split_db_2to1(D, L)
#
#     # Trianing: Here we calculate the mean and covariance matrix of each class
#     hCls = {}  # to store parameters for the three classes
#     for lab in [0, 1, 2]:  # for each label/class
#         DCLS = DTR[:, LTR == lab]  # select data of this class
#         hCls[lab] = estimate_mean_and_covariance(DCLS)
#
#     # Classification
#     prior = vcol(np.ones(3) / 3.0)  # one third prior probability for each class
#     S = []  # each row is the class-conditional density for all the test samples given the hypothesised class
#     for hyp in [0, 1, 2]:  # for each class
#         mu, C = hCls[hyp]  # get the model parameters of the class
#
#         # compute exponential of log-density of samples of the class which gives us a column vector
#         fcond = np.exp(logpdf_GAU_ND_fast(DTV, mu, C))
#
#         S.append(vrow(fcond))
#
#     # S was an array. Now it becomes a matrix of class-conditional densities.
#     # Each column of S consists of the densities of a sample in each class (class-conditional density)
#     S = np.vstack(S)
#     S = S * prior  # S is class-conditional * prior
#
#     # S.sum(0) computes the sum of each column of S (each column consists of marginals)
#     # to produce the joint
#     # P is posterior, dividing S by S.sum(0) gives the posterior for each class of each sample
#     P = S / vrow(S.sum(0))
#
#     predicted_labels = np.argmax(P, axis=0)
#     labels = LTV
#     correct_predictions = predicted_labels == labels
#     correct_predictions = np.sum(correct_predictions)
#     total_predictions = labels.size
#     logging.info(f"acc: {correct_predictions/total_predictions}, correct: {correct_predictions}, total: {total_predictions}")
#     return S, P

# def PCA2(data, num_components):
#     # Calculate the mean vector
#     mean_vector = np.mean(data, axis=1)
#     # Calculate the covariance matrix
#     covariance_matrix = np.dot(data - mean_vector, (data - mean_vector).T) / data.shape[1]
#     # Perform singular value decomposition
#     U, _, _ = np.linalg.svd(covariance_matrix)
#     # Extract the principal components
#     principal_components = U[:, 0:num_components]
#     return principal_components

# def LDA2(D, L, m):
#     SB, SW = calculate_scatter_matrices(D, L)
#     U, s, _ = np.linalg.svd(SW)
#     P1 = np.dot(U, vcol(1.0 / s ** 0.5) * U.T)
#     SBTilde = np.dot(P1, np.dot(SB, P1.T))
#     U, _, _ = np.linalg.svd(SBTilde)
#     P2 = U[:, 0:m]
#     return np.dot(P1.T, P2)


# def logpdf_GAU_ND_1Sample(x, mu, C):
#     xc = x - mu  # Centered x
#     M = x.shape[0]  # The size of the feature vector
#     const = - 0.5 * M * np.log(2 * np.pi)
#     logdet = np.linalg.slogdet(C)[1]  # Index 1 is the value of the log determinant
#     L = np.linalg.inv(C)  # Inverse of the covariance = precision matrix
#     v = np.dot(xc.T, np.dot(L, xc)).ravel()  # Ravel so instead of 1x1 matrix it's a 1-d array with one element
#     return const - 0.5 * logdet - 0.5 * v
#
#
# def logpdf_GAU_ND(X, mu, C):
#     Y = []
#     for i in range(X.shape[1]):
#         Y.append(logpdf_GAU_ND_1Sample(X[:, i:i + 1], mu, C))
#     return np.array(Y).ravel()

# class logRegModel():
#     def __init__(self, D, L, l):
#         self.DTR = D
#         self.ZTR = L * 2.0 - 1.0
#         self.l = l
#         self.dim = D.shape[0]
#
#     def logreg_obj(self, v):
#         w = vcol(v[0:self.dim])
#         b = v[-1]
#         scores = np.dot(w.T, self.DTR) + b
#         loss_per_sample = np.logaddexp(0, -self.ZTR * scores)
#         loss = loss_per_sample.mean() + 0.5 * self.l * np.linalg.norm(w)**2
#         return loss
#
#     def train(self):
#         x0 = np.zeros(self.DTR.shape[0]+1)
#         xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0 = x0, approx_grad=True)
#         print(xOpt)
#         print(fOpt)
#         return xOpt