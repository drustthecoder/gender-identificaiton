import logging

import numpy as np
import scipy
import sklearn.datasets


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L


def vrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape(v.size, 1)


def PCA(D, m):
    mu = vcol(D.mean(1))
    C = np.dot(D - mu, (D - mu).T) / D.shape[1]
    s, U = np.linalg.eigh(C)
    U = U[:, ::-1]
    P = U[:, 0:m]

    return P


def PCA2(D, m):
    mu = vcol(D.mean(1))
    C = np.dot(D - mu, (D - mu).T) / D.shape[1]
    U, _, _ = np.linalg.svd(C)
    P = U[:, 0:m]

    return P


def SbSw(D, L):
    SB = 0
    SW = 0
    mu = vcol(D.mean(1))
    for i in range(L.max() + 1):
        DCls = D[:, L == i]
        muCls = vcol(DCls.mean(1))
        SW += np.dot(DCls - muCls, (DCls - muCls).T)
        SB += DCls.shape[1] * np.dot(muCls - mu, (muCls - mu).T)
    SW /= D.shape[1]
    SB /= D.shape[1]
    return SB, SW


def LDA1(D, L, m):
    SB, SW = SbSw(D, L)
    s, U = scipy.linalg.eigh(SB, SW)
    return U[:, ::-1][:, 0:m]


def LDA2(D, L, m):
    SB, SW = SbSw(D, L)
    U, s, _ = np.linalg.svd(SW)
    P1 = np.dot(U, vcol(1.0 / s ** 0.5) * U.T)
    SBTilde = np.dot(P1, np.dot(SB, P1.T))
    U, _, _ = np.linalg.svd(SBTilde)
    P2 = U[:, 0:m]
    return np.dot(P1.T, P2)


def logpdf_GAU_ND_1Sample(x, mu, C):
    xc = x - mu  # Centered x
    M = x.shape[0]  # The size of the feature vector
    const = - 0.5 * M * np.log(2 * np.pi)
    logdet = np.linalg.slogdet(C)[1]  # Index 1 is the value of the log determinant
    L = np.linalg.inv(C)  # Inverse of the covariance = precision matrix
    v = np.dot(xc.T, np.dot(L, xc)).ravel()  # Ravel so instead of 1x1 matrix it's a 1-d array with one element
    return const - 0.5 * logdet - 0.5 * v


def logpdf_GAU_ND(X, mu, C):
    Y = []
    for i in range(X.shape[1]):
        Y.append(logpdf_GAU_ND_1Sample(X[:, i:i + 1], mu, C))
    return np.array(Y).ravel()


def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = - 0.5 * M * np.log(2 * np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC * np.dot(L, XC)).sum(0)
    # The dot product produces Lx1, Lx2, ...
    return const - 0.5 * logdet - 0.5 * v


def mean_cov_estimate(D):
    mu = vcol(D.mean(1))
    C = np.dot(D - mu, (D - mu).T) / D.shape[1]
    return mu, C


def loglikelihood(X, mu, C):
    return np.sum(logpdf_GAU_ND_fast(X, mu, C))


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def classify_iris():
    D, L = load_iris()
    (DTR, LTR), (DTV, LTV) = split_db_2to1(D, L)

    # Trianing: Here we calculate the mean and covariance matrix of each class
    hCls = {}  # to store parameters for the three classes
    for lab in [0, 1, 2]:  # for each label/class
        DCLS = DTR[:, LTR == lab]  # select data of this class
        hCls[lab] = mean_cov_estimate(DCLS)

    # Classification
    prior = vcol(np.ones(3) / 3.0)  # one third prior probability for each class
    S = []  # each row is the class-conditional density for all the test samples given the hypothesised class
    for hyp in [0, 1, 2]:  # for each class
        mu, C = hCls[hyp]  # get the model parameters of the class

        # compute exponential of log-density of samples of the class which gives us a column vector
        fcond = np.exp(logpdf_GAU_ND_fast(DTV, mu, C))

        S.append(vrow(fcond))

    # S was an array. Now it becomes a matrix of class-conditional densities.
    # Each column of S consists of the densities of a sample in each class (class-conditional density)
    S = np.vstack(S)
    S = S * prior  # S is class-conditional * prior

    # S.sum(0) computes the sum of each column of S (each column consists of marginals)
    # to produce the joint
    # P is posterior, dividing S by S.sum(0) gives the posterior for each class of each sample
    P = S / vrow(S.sum(0))

    predicted_labels = np.argmax(P, axis=0)
    labels = LTV
    correct_predictions = predicted_labels == labels
    correct_predictions = np.sum(correct_predictions)
    total_predictions = labels.size
    logging.info(f"acc: {correct_predictions/total_predictions}, correct: {correct_predictions}, total: {total_predictions}")
    return S, P

def classify_log_iris():
    D, L = load_iris()
    (DTR, LTR), (DTV, LTV) = split_db_2to1(D, L)

    # Trianing: Here we calculate the mean and covariance matrix of each class
    hCls = {}  # to store parameters for the three classes
    for lab in [0, 1, 2]:  # for each label/class
        DCLS = DTR[:, LTR == lab]  # select data of this class
        hCls[lab] = mean_cov_estimate(DCLS)

    # Classification
    logprior = np.log(vcol(np.ones(3) / 3.0))  # one third prior probability for each class
    S = []  # each row is the class-conditional density for all the test samples given the hypothesised class
    for hyp in [0, 1, 2]:  # for each class
        mu, C = hCls[hyp]  # get the model parameters of the class

        # compute log-density of samples of the class which gives us a column vector
        fcond = logpdf_GAU_ND_fast(DTV, mu, C)

        S.append(vrow(fcond))

    # S was an array. Now it becomes a matrix of class-conditional log densities.
    # Each column of S consists of the log densities of a sample in each class (class-conditional log density)
    S = np.vstack(S)
    S = S + logprior  # S is class-conditional log density + log prior

    # logsumexp does exponentiatin of S (which is the log-joint),
    # then does the sum of the joint matrix over the rows and then takes the log.
    # logsumexp = log(exp(s).sum(0)) but numerically stable
    # P is posterior
    # logP is log-joint minus the log-marginal
    logP = S - vrow(scipy.special.logsumexp(S, 0))
    P = np.exp(logP)

    predicted_labels = np.argmax(P, axis=0)
    labels = LTV
    correct_predictions = predicted_labels == labels
    correct_predictions = np.sum(correct_predictions)
    total_predictions = labels.size
    logging.info(f"acc: {correct_predictions/total_predictions}, correct: {correct_predictions}, total: {total_predictions}")
    return S, P


def calculate_error_rate(predictions, labels):
    incorrect_predictions = np.sum(predictions != labels)
    error_rate = incorrect_predictions / len(labels)
    return error_rate


class LogisticRegressionModel:
    def __init__(self, training_data, training_labels, evaluation_data, evaluation_labels, regularization_param):
        self.training_data = training_data
        self.training_labels = training_labels * 2.0 - 1.0  # This is ZTR
        self.evaluation_data = evaluation_data
        self.evaluation_labels = evaluation_labels * 2.0 - 1.0
        self.regularization_param = regularization_param  # This is Lambda
        self.num_features = training_data.shape[0]  # This is dim

    def calculate_loss(self, weights_and_bias):
        weights = np.array(weights_and_bias[:self.num_features])
        bias = weights_and_bias[-1]
        scores = np.dot(weights.T, self.training_data) + bias
        loss_per_sample = np.logaddexp(0, -self.training_labels * scores)
        loss = loss_per_sample.mean() + 0.5 * self.regularization_param * np.linalg.norm(weights) ** 2
        return loss

    def train(self):
        initial_weights_and_bias = np.zeros(self.num_features + 1)
        optimized_weights_and_bias, optimal_loss, optimization_info = scipy.optimize.fmin_l_bfgs_b(
            self.calculate_loss,
            x0=initial_weights_and_bias,
            approx_grad=True
        )
        print("Optimized weights and bias:", optimized_weights_and_bias)
        print("Optimal loss:", optimal_loss)

        # Compute predictions on the training data
        weights = optimized_weights_and_bias[:self.num_features]
        bias = optimized_weights_and_bias[-1]

        # compute scores using evaluation data
        scores = np.dot(weights.T, self.evaluation_data) + bias
        predictions = np.sign(scores)

        # Compute the error rate
        error_rate = calculate_error_rate(predictions, self.evaluation_labels)

        return optimized_weights_and_bias, optimal_loss, error_rate


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

if __name__ == "__main__":
    print("This is mlpr.py")
